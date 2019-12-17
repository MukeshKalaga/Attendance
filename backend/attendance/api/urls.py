from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('incr/', views.incrnum, name='random'),
    path('recog/', views.recog2, name='recognition'),
]