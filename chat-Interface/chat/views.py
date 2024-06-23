import requests  # Añade esto al inicio para hacer solicitudes HTTP
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_exempt
from .models import Conversation, User
import json

# Página de inicio que puede incluir el formulario de inicio de sesión
def index(request):
    return render(request, 'index.html')

# API para recibir mensajes y responder
@csrf_exempt
def api_view(request):
    if request.method == "POST":
        user_input = request.POST.get('mensaje', '')  # Asegúrate de enviar el mensaje como 'mensaje' en la solicitud POST desde el frontend
        # Hacer una solicitud al servicio FastAPI
        response = requests.post('http://localhost:8800/chat/', json={'user_input': user_input})
        if response.status_code == 200:
            response_data = response.json()
            return JsonResponse({'mensaje': response_data['response']})
        else:
            return JsonResponse({'error': 'Error con el servicio de chat'}, status=500)
    return JsonResponse({'error': 'Método no permitido'}, status=405)

# Registro de usuarios
@csrf_exempt
def register(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
        email = data.get('email', '')
        if not User.objects.filter(username=username).exists():
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            return JsonResponse({'status': 'success', 'message': 'Usuario registrado exitosamente.'})
        else:
            return JsonResponse({'status': 'error', 'message': 'El nombre de usuario ya existe.'})
    return JsonResponse({'error': 'Método no permitido'}, status=405)

# Autenticación de usuarios
@csrf_exempt
def user_login(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({'status': 'success', 'message': 'Inicio de sesión exitoso.'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Usuario o contraseña incorrecta.'})
    return JsonResponse({'error': 'Método no permitido'}, status=405)

# Cierre de sesión de usuarios
@csrf_exempt
def user_logout(request):
    logout(request)
    return JsonResponse({'status': 'success', 'message': 'Sesión cerrada exitosamente.'})
