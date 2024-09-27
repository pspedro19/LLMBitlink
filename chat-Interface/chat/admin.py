from django.contrib import admin
from .models import VectorChunk

# Registrar el modelo VectorChunk en el panel de administración
admin.site.register(VectorChunk)