from django.urls import path
from .views import summarize_text, home

urlpatterns = [
    path("", home, name="home"),
    path("summarize/", summarize_text, name="summarize"),
]
