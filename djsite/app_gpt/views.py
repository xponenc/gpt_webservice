from django.shortcuts import render
from django.http import HttpResponse


def start(request):
    return render(request, 'app_gpt/start.html')


def test(request):
    return HttpResponse('Тестовая страница')


def chatbot(request):
    return render(request, 'app_gpt/chatbot.html')
