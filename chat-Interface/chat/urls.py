from django.urls import path
from chat import views

urlpatterns = [
    path('save_vectorization/', views.save_vectorization, name='save_vectorization'),
]
