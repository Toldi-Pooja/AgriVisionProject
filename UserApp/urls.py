from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),

    path('entry/', views.user_entry, name='user_entry'),
    path('register/', views.user_register, name='user_register'),
    path('login/', views.user_login, name='user_login'),
    path('logout/', views.user_logout, name='user_logout'),

    path('dashboard/', views.user_dashboard, name='user_dashboard'),
    path('profile/', views.user_profile, name='user_profile'),

    path('weed/', views.weed_detection, name='weed_detection'),
    path('disease/', views.disease_detection, name='disease_detection'),
    path('quality/', views.crop_quality, name='crop_quality'),

    path('result/', views.result_page, name='result_page'),
]