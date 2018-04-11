from django.conf.urls import url, include

from . import views

urlpatterns = [
    url(r'', views.searchView.as_view()),
    url(r'testing/', views.filterView.as_view()),
    
]

