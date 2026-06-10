from django.urls import path
from . import views

urlpatterns = [
    path("login/", views.admin_login, name="admin_login"),
    path("dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("view-dataset/", views.view_dataset, name="view_dataset"),
    path("preprocess/", views.preprocess_page, name="preprocess_page"),
    path("train/", views.train_models, name="train_models"),
    path("compare/", views.compare_models, name="compare_models"),
    path("logout/", views.admin_logout, name="admin_logout"),
]