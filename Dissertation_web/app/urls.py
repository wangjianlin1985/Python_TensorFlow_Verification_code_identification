from django.conf.urls import url,include
from django.contrib import admin
from app import views
admin.autodiscover()

urlpatterns = [
    url(r'^index/$',views.index),
    url(r'^login/$',views.login),
    url(r'^regist/$',views.regist),

]