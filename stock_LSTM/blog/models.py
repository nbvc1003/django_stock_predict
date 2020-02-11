from django.db import models
from django.conf import settings
from django.utils import timezone

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.applications import imagenet_utils
from keras import backend as K


# Create your models here.

class Post(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(default=timezone.now)
    published_date = models.DateTimeField(blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title


class Comment(models.Model):
    post = models.ForeignKey('blog.Post', on_delete=models.CASCADE, related_name='comments')
    author = models.CharField(max_length=200)
    text = models.TextField()
    create_date = models.DateTimeField(default=timezone.now)
    approved_comment = models.BooleanField(default=False)

    def approve(self):
        self.approved_comment = True
        self.save()

    def __str__(self):
        return self.text


class Classification(models.Model):
    img = models.ImageField(upload_to='images')
    prediction = models.CharField(max_length=200, blank=True)

    def predict(self):
        K.reset_uids()

        model = 'rnn/model/model_mobilenet.json'
        weights = 'rnn/model/weights_mobilenet.h5'

        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            with open(model, 'r') as f:
                model = model_from_json(f.read())
                model.load_weights(weights)
        img = image.load_img(self.img, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        result = model.predict(x)
        # result_decode = imagenet_utils.decode_predictions(result, top=1)[0][0][1]
        result_decode = imagenet_utils.decode_predictions(result)
        for (i, (predId, pred, prob)) in enumerate(result_decode[0]):
            return "{}.-  {}: {:.2f}%".format(i + 1, pred, prob * 100)

    def save(self, *args, **kwargs):
        self.prediction = self.predict()
        super().save(*args, **kwargs)