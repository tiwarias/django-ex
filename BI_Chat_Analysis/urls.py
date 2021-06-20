"""BI_Chat_Analysis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
import os, requests, uuid
from . import views
obj_admin_view = views.main_admin()

urlpatterns = [
    path('admin/', admin.site.urls),
    path('admin_index/', obj_admin_view.admin_index),
    path('', include('translate.urls')),
    path('', obj_admin_view.home),
    path('home', include('translate.urls')),
    path('login', views.loginPage, name = 'login'),
    path('logout', views.logoutUser, name = 'logout'),
    #path('register', views.registerPage, name = 'register'),
    path('selection', obj_admin_view.selection, name = 'selection'),
    path('synonym_add', obj_admin_view.synonym_add, name = 'synonym_add'),
    path('synonym_del', obj_admin_view.synonym_del, name = 'synonym_del'),
    path('intent_function', obj_admin_view.intent_function, name = 'intent_function'),
    path('category_function', obj_admin_view.category_function, name = 'category_function'),
    path('fetch_keywords', obj_admin_view.fetch_keywords, name = 'fetch_keywords')
   
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
