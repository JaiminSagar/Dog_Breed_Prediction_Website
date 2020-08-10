from django.db import models
from django.conf import settings



# Create your models here.
class ImagesUploadModel(models.Model):
    image = models.ImageField(blank=False, upload_to='img/')

# class FigureUploadModel(model.Model):
#     figure = models.ImageField(upload_to='fig/', default=)