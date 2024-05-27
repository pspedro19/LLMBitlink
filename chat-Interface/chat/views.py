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
def api_view(request, mensaje):
    if request.method == "POST":
        user = request.user
        user_input = mensaje
        # Simula la respuesta del asistente (integra tu lógica de modelo aquí)
        response_text = f"Respondido: {mensaje}"
        # Guardar en la base de datos
        Conversation.objects.create(user=user, input=user_input, output=response_text)
        return JsonResponse({'mensaje': response_text})
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
