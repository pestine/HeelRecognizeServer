from django.urls import path,re_path
from . import views
from . import tests

urlpatterns = [
    path('uploadUrl', views.uploadUrl, name='uploadUrl'),
    path('uploadImage', views.uploadImage, name='uploadImage'),
    path('test', tests.test, name='test'),

]