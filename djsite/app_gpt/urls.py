from django.urls import path

from .views import start, test, chatbot

app_name = 'chatbot'

urlpatterns = [
    path('', start, name='start'),
    path('test/', test, name='test'),
    path('chat/', chatbot, name='chat'),
]
