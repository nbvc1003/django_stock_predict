from django.urls import path
from django.conf.urls import url
from . import views
from .views import ClassificationCreateView, ClassificationDeleteView, ClassificationListview, ClassificationUpdateView

urlpatterns = [
    # '외부에서콜할수 있는 url주소', function : 불려지는 함수, name= 'name' : {% url 'detail ..과같이 이름으로 url 대신 사용
    path('rnn', ClassificationListview.as_view(), name='list'),
    path('rnn/create/', ClassificationCreateView.as_view(), name='create'),
    path('rnn/update/<int:pk>/', ClassificationUpdateView.as_view(), name='update'),
    path('rnn/delete/<int:pk>/', ClassificationDeleteView.as_view(), name='delete'),

    path('rnn/test1/', ClassificationListview.as_view(), name='test1'),
    path('rnn/create1/', ClassificationCreateView.as_view(), name='create1'),


]