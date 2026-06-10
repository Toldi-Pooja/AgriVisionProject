from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from UserApp import views as user_views

urlpatterns = [
    path("admin/", admin.site.urls),

    # Public home
    path("", user_views.home, name="home"),

    # UserApp
    path("user/", include("UserApp.urls")),

    # AdminApp
    path("adminapp/", include("AdminApp.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)