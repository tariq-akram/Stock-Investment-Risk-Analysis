from django.urls import path
from . import views

urlpatterns = [
    path('', views.stock_analysis_view, name='index'),
]
