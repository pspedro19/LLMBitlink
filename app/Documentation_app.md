# 🤖 LLMBitlink

## Descripción General
LLMBitlink es una aplicación FastAPI que implementa un sistema conversacional inteligente para recomendaciones turísticas en Curazao. El sistema utiliza procesamiento de lenguaje natural avanzado y la API de OpenAI para proporcionar recomendaciones personalizadas y mantener conversaciones naturales con los usuarios.

## 📋 Tabla de Contenidos
1. [Tecnologías Utilizadas](#tecnologías-utilizadas)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Configuración y Despliegue](#configuración-y-despliegue)
5. [Módulos Principales](#módulos-principales)
6. [Integración con OpenAI](#integración-con-openai)
7. [Gestión de Datos](#gestión-de-datos)
8. [Sistema de Logging](#sistema-de-logging)
9. [API Endpoints](#api-endpoints)

## Tecnologías Utilizadas

### 1. Procesamiento de Lenguaje Natural (NLP)

#### SpaCy
- **Modelo Base**: `es_core_news_sm` para procesamiento en español
- **Implementación**: `ImprovedNLPProcessor` en `nlp_processor.py`
- **Características principales**:
  ```python
  - Extracción de entidades nombradas (NER)
  - Análisis sintáctico y semántico
  - Procesamiento de texto multilingüe
  - Sistema de caché para optimización
  ```

#### Análisis de Intenciones
- **Patrones de Interés**:
  - Cultural (museos, historia, arte)
  - Aventura (senderismo, deportes)
  - Naturaleza (parques, playas)
  - Gastronomía
  - Actividades acuáticas
- **Sistema de Puntuación**:
  ```python
  intent_scores = {
      'activity_search': 1.5,
      'food_search': 1.2,
      'cultural_interest': 1.3,
      'nature_adventure': 1.4,
      'planning_logistics': 1.1
  }
  ```

### 2. Sistema de Recomendaciones

#### Motor de Recomendaciones
- **Clase Principal**: `RecommendationEngine` en `recommendation_engine.py`
- **Características**:
  ```python
  - Scoring ponderado multicriteria
  - Balanceo de categorías
  - Diversificación de resultados
  - Cache de recomendaciones
  ```

#### Sistema de Puntuación
- **Implementación**: `RecommendationScoring` en `scoring.py`
- **Pesos de Criterios**:
  ```python
  scoring_weights = {
      'interest_match': 2.0,
      'location_match': 1.5,
      'budget_match': 1.3,
      'rating_bonus': 1.2,
      'diversity_bonus': 1.1
  }
  ```
- **Características**:
  - Matching de intereses contextual
  - Bonificación por diversidad
  - Puntuación basada en presupuesto
  - Factor de calificación

### 3. Sistema de Logging

#### Configuración Centralizada
- **Implementación**: `get_logger()` en `logger.py`
- **Características**:
  ```python
  - Logging multinivel
  - Rotación de archivos
  - Formato personalizado
  - Separación por módulos
  ```
- **Estructura**:
  ```
  /logs/
  ├── core.analyzer.nlp.processor.log
  ├── core.recommender.full_service.log
  ├── core.analyzer.query.log
  └── ...
  ```

### 4. FastAPI y Componentes Web
- **FastAPI**: Framework principal para la API REST
  - Implementación de endpoints asíncronos
  - Sistema de validación con Pydantic
  - Middleware CORS para manejo cross-origin
  - Documentación automática OpenAPI
  - Manejo de errores HTTP personalizado

- **HTML Templates**
  - Sistema de templates para respuestas formateadas
  - Componentes HTML dinámicos para recomendaciones
  - Estilos CSS integrados

#### Procesamiento de Lenguaje Natural
- **OpenAI API (GPT-3.5 Turbo)**
  - Generación de respuestas conversacionales
  - Análisis de intenciones del usuario
  - Formateo de recomendaciones en lenguaje natural
  - Sistema de roles para contextualización
  - Manejo de temperaturas para variedad en respuestas

- **SpaCy**
  - Modelo es_core_news_sm para español
  - Análisis sintáctico y semántico
  - Extracción de entidades nombradas (NER)
  - Procesamiento multilingüe

#### Gestión de Datos
- **Excel Engine**
  - Lectura y procesamiento de archivos XLSX
  - Cache de datos para optimización
  - Sistema de consultas estructurado
  - Validación de datos de entrada

- **Base de Datos en Memoria**
  - Caché de respuestas frecuentes
  - Sistema TTL para expiración de datos
  - Optimización de rendimiento

#### Infraestructura
- **Docker**
  - Containerización de la aplicación
  - Gestión de dependencias
  - Configuración de entorno aislado
  - Scripts de despliegue automatizado

- **Python 3.9+**
  - Async/await para operaciones asíncronas
  - Type hints para seguridad de tipos
  - F-strings para formateo eficiente
  - Context managers para recursos

### Bibliotecas y Frameworks Auxiliares

#### Sistema de Logging
- Registro multinivel
- Rotación de archivos
- Formateo personalizado
- Integración con sistemas de monitoreo

#### Validación y Modelado
- **Pydantic**
  - Modelos de datos validados
  - Conversión automática de tipos
  - Serialización JSON
  - Validación de configuración

#### Utilitarios
- **CORS Middleware**
  - Configuración de orígenes permitidos
  - Manejo de headers personalizados
  - Métodos HTTP permitidos

- **Datetime Utilities**
  - Manejo de zonas horarias
  - Formateo de fechas
  - Cálculos temporales

### Integraciones Externas
- **OpenAI API Client**
  - Gestión de tokens
  - Reintentos automáticos
  - Manejo de rate limiting
  - Gestión de errores

## Arquitectura del Sistema

El sistema está diseñado con una arquitectura modular que separa claramente las responsabilidades:

```
[Cliente] ←→ [FastAPI Server] ←→ [Core Services]
                   ↓               ↙     ↓     ↘
              [OpenAI API]  [Analyzer] [Data] [Recommender]
```

### Flujo de Datos Principal
1. El cliente envía una consulta en lenguaje natural
2. El servidor procesa la solicitud a través del sistema NLP
3. Se extraen preferencias y se analizan intenciones
4. El motor de recomendaciones genera sugerencias personalizadas
5. Se formatea la respuesta (HTML/JSON) y se envía al cliente

## Estructura del Proyecto

### Core
- **analyzer/**: Procesamiento de lenguaje natural y análisis de consultas
  - `nlp_processor.py`: Procesamiento de texto y extracción de preferencias
  - `preference.py`: Gestión de preferencias de usuario
  - `query.py`: Análisis de consultas

- **data/**: Gestión de base de datos
  - `database.py`: Conexión y operaciones con datos
  - `models.py`: Modelos de datos

- **recommender/**: Motor de recomendaciones
  - `full_service.py`: Servicio principal de recomendaciones
  - `recommendation_engine.py`: Lógica de recomendaciones
  - `formatter.py`: Formateo de respuestas
  - `scoring.py`: Sistema de puntuación
  - `validator.py`: Validación de recomendaciones

### Utils
- `openai_helper.py`: Integración con OpenAI
- `logger.py`: Configuración de logging
- `config.py`: Configuración global

## Configuración y Despliegue

### Variables de Entorno
```env
OPENAI_API_KEY=your-api-key
```

### Requisitos del Sistema
- Python 3.9+
- Docker (opcional)
- 2GB RAM mínimo
- Espacio en disco: 500MB mínimo

### Instalación

1. Clonar el repositorio:
```bash
git clone [repository-url]
cd LLMBitlink
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
```bash
export OPENAI_API_KEY=your-api-key
```

### Despliegue con Docker
```bash
docker build -t llmbitlink .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-api-key llmbitlink
```

## Integración con OpenAI

La integración con OpenAI se realiza a través de la clase `OpenAIHelper` en `utils/openai_helper.py`. Esta clase proporciona métodos para:

- Generación de introducciones para guías turísticos
- Respuestas conversacionales
- Manejo de errores y respuestas de fallback

### Ejemplo de Uso
```python
from utils.openai_helper import OpenAIHelper

openai_helper = OpenAIHelper(api_key)
response = openai_helper.generate_tour_guide_response(
    user_text="¿Qué actividades recomiendas en Willemstad?",
    system_message="Eres un guía turístico experto en Curazao"
)
```

## Gestión de Datos

El sistema utiliza archivos Excel para almacenar información sobre:

### Estructura de Datos
- **Tourist Spots** (63 registros)
  - Tipos: historic site, museum, park
  - Ubicaciones: Westpunt, Punda, Willemstad, Otrobanda, etc.
  - Ratings: 3.5 - 4.9

- **Activities** (67 registros)
  - Tipos: kayaking, tours, snorkeling, hiking
  - Duración: 1-8 horas
  - Costos: $15-500

- **Nightclubs** (61 registros)
  - Tipos de música: reggaeton, electronic, salsa
  - Rangos de precio: low, medium, high
  - Horarios: 22:00-04:00 típicamente

- **Restaurants** (63 registros)
  - Tipos de cocina: international, fusion, local
  - Rangos de precio: $25-125 por persona
  - Ratings: 3.1 - 4.9

### Acceso a Datos
Los datos se cargan y procesan a través del módulo `core.data.database`, que proporciona una capa de abstracción para acceder a la información almacenada en los archivos Excel.

## Sistema de Logging

El sistema implementa un logging comprehensivo configurado en `utils/logger.py`:

### Niveles de Log
- INFO: Operaciones normales
- ERROR: Errores y excepciones
- DEBUG: Información de depuración

### Archivos de Log
Los logs se almacenan en el directorio `/logs/` con archivos separados para cada módulo:
- `core.analyzer.nlp.processor.log`
- `core.recommender.full_service.log`
- etc.

## API Endpoints

### Endpoints Principales

#### POST `/recommendations/`
- **Descripción**: Obtiene recomendaciones turísticas personalizadas
- **Input**: RecommendationRequest (query + preferences)
- **Output**: JSONResponse con recomendaciones

#### POST `/recommendations/chat`
- **Descripción**: Endpoint conversacional con análisis de intenciones
- **Input**: ChatRequest (text)
- **Output**: HTMLResponse con respuesta formateada

#### POST `/recommendations/full`
- **Descripción**: Combina análisis NLP y generación de HTML
- **Input**: NLPRequest (text)
- **Output**: HTMLResponse con recomendaciones completas

### Formatos de Request

```python
class Preferences(BaseModel):
    interests: List[str]
    locations: List[str]
    budget_per_day: Optional[float]
    trip_duration: int
    group_size: int
    activity_types: List[str]
    specific_sites: Optional[List[str]]
    cuisine_preferences: Optional[List[str]]
```

Para más detalles técnicos y ejemplos de uso, consulta la documentación de la API en `/docs` cuando el servidor esté en ejecución.