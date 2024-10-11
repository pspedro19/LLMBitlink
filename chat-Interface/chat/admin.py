from django.contrib import admin
from import_export.admin import ImportExportModelAdmin  # Para importar/exportar
from .models import Conversation, Chunk  # Asegúrate de importar el modelo Chunk

# Registrar el modelo Conversation en el admin
@admin.register(Conversation)
class ConversationAdmin(ImportExportModelAdmin):
    list_display = ('user', 'input', 'output', 'timestamp')  # Columnas a mostrar
    search_fields = ('user__username', 'input', 'output')  # Campos por los que se puede buscar
    list_filter = ('user', 'timestamp')  # Filtros laterales
    ordering = ('-timestamp',)  # Ordenar por las conversaciones más recientes

# Registrar el modelo Chunk en el admin
@admin.register(Chunk)
class ChunkAdmin(ImportExportModelAdmin):
    list_display = ('document_id', 'content_summary')  # Mostrar parte del contenido
    search_fields = ('document_id', 'content')  # Permitir búsqueda por ID y contenido
    ordering = ('document_id',)  # Ordenar por el ID del documento
    def content_summary(self, obj):
        return obj.content[:50]  # Mostrar los primeros 50 caracteres del contenido
    content_summary.short_description = 'Content Summary'