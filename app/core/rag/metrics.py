# core/rag/metrics.py
from prometheus_client import Counter, Histogram, start_http_server

class RAGMetrics:
    def __init__(self):
        self.document_counter = Counter(
            'rag_documents_total',
            'Total de documentos procesados en el sistema RAG'
        )
        self.processing_time = Histogram(
            'rag_processing_seconds',
            'Tiempo de procesamiento de documentos'
        )
        self.search_latency = Histogram(
            'rag_search_latency_seconds',
            'Latencia en búsquedas semánticas'
        )

# Inicia el servidor de métricas en el puerto 8000 (única instancia)
start_http_server(8000)
