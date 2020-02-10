from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from .models import Post
from .forms import PostForm

import os
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pandas_datareader as web
from datetime import date, timedelta

# Create your views here.
def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_list.html', {'posts':posts})

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)

    return render(request , 'blog/post_detail.html', {'post':post})

def post_new(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            # post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm()
    return render(request, 'blog/post_edit.html',{'form':form})

def post_edit(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.method == 'POST':
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            # post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm(instance=post)
    return render(request, 'blog/post_edit.html', {'form': form})

def post_draft_list(request):
    posts = Post.objects.filter(published_date__isnull=True).order_by('created_date')
    return render(request, 'blog/post_draft_list.html',{'post':posts})

def post_publish(request, pk):
    post = get_object_or_404(Post, pk=pk)
    post.publish()
    return redirect('post_detail', pk=pk)

def post_remove(request,pk):
    post = get_object_or_404(Post, pk=pk)
    post.delete()
    return redirect('post_list')

def lstm(request):

    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'KS11_s_30_1_2020-02-10.h5')

    model = models.Sequential()
    model = models.load_model(file_path)

    # =============================================================================
    # model.reset_states()
    ## 미래 5일 예측
    TIME_STEP = 30
    CSV_FILE = '^KS11'
    PREDICT_DAYS = 5
    # 최근 일 데이터로 다음날 예측 빠지는 날짜를 대비해서 2배수 요청
    # 오늘은 거래중인 데이터가 최신이 아니기 때문에 어제 까지 요청
    btc_quote = web.DataReader(CSV_FILE, data_source='yahoo', start=date.today() + timedelta(days=-TIME_STEP * 2),
                               end=date.today() + timedelta(days=-1))
    new_df = btc_quote.filter(['Close'])
    # 최근 데이터만 사용..
    new_df = new_df[len(new_df) - TIME_STEP:]
    new_last_df = new_df[-TIME_STEP:]
    last_his_days = new_last_df.values

    #
    scaler = MinMaxScaler()
    last_his_days_scaled = scaler.transform(last_his_days)

    seq_in = last_his_days_scaled
    seq_in_feature = []
    temp_out = []

    seq_in_feature.append(seq_in)
    # volume_mean = np.mean(seq_in, axis=1)[1]

    for i in range(PREDICT_DAYS):
        sample_in = np.array(seq_in_feature)
        sample_in = np.reshape(sample_in, (1, model.input_shape[1], model.input_shape[2]))
        pred_value = model.predict(sample_in)

        # 예측값 특성에 추가, volume은 사전에 구한 평균값으로
        seq_in_feature[0] = np.append(seq_in_feature[0], [[np.squeeze(pred_value)]], axis=0)
        seq_in_feature[0] = seq_in_feature[0][1:]  # 특성 앞쪽 제거

        pred = np.zeros(shape=(len(pred_value), 2))
        pred[:, 0] = pred_value[:, 0]
        pred_price = scaler.inverse_transform(pred)

        # 결과 원복 , 누적
        temp_out.append(pred_price)

    temp_out = np.squeeze(temp_out)

    # new_last_df['Predict'] =
    dates = pd.date_range(date.today() + timedelta(days=1), periods=5)
    predic_df = pd.DataFrame(temp_out[:, 0], index=dates, columns=['Predictions'])
    new_p_df = pd.merge(new_last_df, predic_df, how='outer', left_index=True, right_index=True)

    # predic_df.loc[date.today()+ timedelta(days=-1)] = new_last_df.loc[date.today()+ timedelta(days=-1)]['Close']

    list_index_date = new_last_df.index[-1]
    if list_index_date < date.today():
        predic_df.loc[list_index_date] = new_last_df.loc[list_index_date]['Close']
    else:
        predic_df.loc[date.today() + timedelta(days=-1)] = new_last_df.loc[date.today() + timedelta(days=-1)]['Close']

    predic_df.sort_index(inplace=True)

    plt.figure(figsize=(10, 5))
    plt.title(' {}day Pre'.format(PREDICT_DAYS))
    plt.xlabel('Date', fontsize=5)
    plt.xticks(rotation=40)
    plt.ylabel('Close Price', fontsize=8)
    plt.plot(new_last_df['Close'])
    plt.plot(predic_df)
    plt.legend(['Recents', 'Predictions'], loc='upper left')

    # plt.savefig(savePredictfigPath)
    # plt.show()

    return render(request, 'blog/post_draft_list.html')