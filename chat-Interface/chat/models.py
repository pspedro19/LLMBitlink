from django.db import models
from django.contrib.auth.models import User

# Modelo para el historial de conversaciones
class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    input = models.TextField()
    output = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

class Chunk(models.Model):
    document_id = models.IntegerField()
    content = models.TextField()
    embedding = models.JSONField()

    def __str__(self):
        return f"Document {self.document_id}: {self.content[:50]}"
