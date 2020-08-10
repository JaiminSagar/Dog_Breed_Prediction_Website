from django.urls import path
from . import views

app_name = 'dog_breed'

urlpatterns = [
    path('', views.index, name='index'),
    path('return_breeds/', views.return_breeds, name='return_breeds'),
    #path('delete_images/', views.delete_images, name='delete_images'), 
]
