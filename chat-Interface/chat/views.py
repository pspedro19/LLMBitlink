import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_protect, csrf_exempt  # Cambio de csrf_exempt a csrf_protect
from .models import Chunk  # Asegúrate de importar Chunk
import json
import logging

logger = logging.getLogger(__name__)

def index(request):
    return render(request, 'index.html')

@csrf_protect  # Protección CSRF habilitada
def api_view(request):
    if request.method == "POST":
        user_input = request.POST.get('mensaje', '')
        try:
            response = requests.post('http://localhost:8800/chat/', json={'user_input': user_input})
            if response.status_code == 200:
                response_data = response.json()
                if 'response' in response_data:
                    return JsonResponse({'mensaje': response_data['response']})
                else:
                    logger.error('Respuesta inesperada desde FastAPI: %s', response_data)
                    return JsonResponse({'error': 'Respuesta inesperada desde el servicio de chat'}, status=500)
            else:
                logger.error('Error de estado desde FastAPI: %s', response.status_code)
                return JsonResponse({'error': 'Error con el servicio de chat'}, status=response.status_code)
        except requests.exceptions.RequestException as e:
            logger.error('Excepción al conectar con FastAPI: %s', e)
            return JsonResponse({'error': 'Error de conexión con el servicio de chat'}, status=500)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)

@csrf_protect
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

@csrf_protect
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

@csrf_protect
def user_logout(request):
    logout(request)
    return JsonResponse({'status': 'success', 'message': 'Sesión cerrada exitosamente.'})

@csrf_exempt
def save_vectorization(request):
    if request.method == 'POST':
        try:
            logger.info(f"Request body: {request.body}")
            data = json.loads(request.body)
            logger.info(f"Data received: {data}")
            for item in data:
                Chunk.objects.create(
                    document_id=item['document_id'],
                    content=item['content'],
                    embedding=item['embedding']
                )
            return JsonResponse({"status": "success"})
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return JsonResponse({"status": "fail", "error": str(e)}, status=400)
    return JsonResponse({"status": "fail"}, status=400)
