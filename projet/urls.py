
from django.urls import path
from . import views
from  . import apps


urlpatterns = [
    path('', views.index, name='index'),
    path('tooo', views.ind, name='testo'),
    path('JJ', apps.hd, name='hd'),
    path('test', views.test, name='test'),
    path('inter', views.update, name='up'),

]