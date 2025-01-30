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
```json
  {
    "query": "Discover the colonial architecture and vibrant local culture of Willemstad. Join a photography-focused walking tour through colorful streets, UNESCO sites, and bustling markets. End the day savoring local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Trip duration: 3 days. Budget: $50 per day.",
    "preferences": {
      "interests": [
        "colonial architecture",
        "photography",
        "local culture"
      ],
      "locations": [
        "unesco sites",
        "bustling markets",
        "willemstad"
      ],
      "budget_per_day": 50,
      "trip_duration": 3,
      "group_size": 1,
      "activity_types": [
        "walking tour"
      ],
      "specific_sites": [
        "curaçaoan restaurant"
      ],
      "cuisine_preferences": [
        "keshi yena",
        "stoba"
      ]
    }
  }
```

- **Output**: JSONResponse con recomendaciones
```json
  {
    "status": "success",
    "recommendations": [
      {
        "name": "SPICE GARDEN",
        "type": "unknown",
        "location": "Willemstad",
        "cost": 0,
        "rating": {
          "value": 4.8,
          "display": "★★★★☆"
        },
        "description": "Authentic Indian food with freshly ground spices.",
        "relevance_score": 0.6552
      },
      {
        "name": "SPICE GARDEN",
        "type": "unknown",
        "location": "Willemstad",
        "cost": 0,
        "rating": {
          "value": 4.8,
          "display": "★★★★☆"
        },
        "description": "Authentic Indian food with freshly ground spices.",
        "relevance_score": 0.6552
      },
      {
        "name": "CURAÇAO CARNIVAL DANCE WORKSHOP",
        "type": "culture",
        "location": "Willemstad",
        "cost": 50,
        "rating": {
          "value": 4.7,
          "display": "★★★★☆"
        },
        "description": "Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.",
        "relevance_score": 0.544
      },
      {
        "name": "CURAÇAO CARNIVAL DANCE WORKSHOP",
        "type": "culture",
        "location": "Willemstad",
        "cost": 50,
        "rating": {
          "value": 4.7,
          "display": "★★★★☆"
        },
        "description": "Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.",
        "relevance_score": 0.544
      },
      {
        "name": "HISTORIC WILLEMSTAD WALKING TOUR",
        "type": "tour",
        "location": "Willemstad",
        "cost": 40,
        "rating": {
          "value": 4.6,
          "display": "★★★★☆"
        },
        "description": "Discover UNESCO-listed architecture, colorful street art, and hidden gems in Willemstad.",
        "relevance_score": 0.6504
      },
      {
        "name": "HISTORIC WILLEMSTAD WALKING TOUR",
        "type": "tour",
        "location": "Willemstad",
        "cost": 40,
        "rating": {
          "value": 4.6,
          "display": "★★★★☆"
        },
        "description": "Discover UNESCO-listed architecture, colorful street art, and hidden gems in Willemstad.",
        "relevance_score": 0.6504
      },
      {
        "name": "WILLEMSTAD ART WALK",
        "type": "art",
        "location": "Willemstad",
        "cost": 30,
        "rating": {
          "value": 4.5,
          "display": "★★★★☆"
        },
        "description": "Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.",
        "relevance_score": 0.54
      },
      {
        "name": "WILLEMSTAD ART WALK",
        "type": "art",
        "location": "Willemstad",
        "cost": 30,
        "rating": {
          "value": 4.5,
          "display": "★★★★☆"
        },
        "description": "Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.",
        "relevance_score": 0.54
      },
      {
        "name": "PROGRESSIVE 5THGENERATION APPLICATION",
        "type": "park",
        "location": "Willemstad",
        "cost": 0,
        "rating": {
          "value": 4,
          "display": "★★★★☆"
        },
        "description": "An ideal location for families to relax.",
        "relevance_score": 0.636
      },
      {
        "name": "REACTIVE 24HOUR MORATORIUM",
        "type": "park",
        "location": "Willemstad",
        "cost": 0,
        "rating": {
          "value": 4,
          "display": "★★★★☆"
        },
        "description": "An ideal location for families to relax.",
        "relevance_score": 0.53
      }
    ],
    "metadata": {
      "query_time": "2025-01-30T15:42:28.030981",
      "processing_time": 0.014071,
      "total_results": 10,
      "query_understanding": 1,
      "currency": "USD",
      "timestamp": "2025-01-30T15:42:28.031050"
    },
    "validation": {
      "location_match": 1,
      "budget_match": 0.9333333333333335,
      "interest_match": 1,
      "diversity_score": 0.4666666666666667,
      "preference_coverage": 0.3333333333333333
    }
  }
```

#### POST `/recommendations/html`
- **Descripción**: Obtiene recomendaciones turísticas personalizadas, endpoint basado en `/recommendations/` que mejora la salida mostrando el resultado en cards contruidas con html
- **Input**: RecommendationRequest (query + preferences)
```json
  {
    "query": "Discover the colonial architecture and vibrant local culture of Willemstad. Join a photography-focused walking tour through colorful streets, UNESCO sites, and bustling markets. End the day savoring local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Trip duration: 3 days. Budget: $50 per day.",
    "preferences": {
      "interests": [
        "colonial architecture",
        "photography",
        "local culture"
      ],
      "locations": [
        "unesco sites",
        "bustling markets",
        "willemstad"
      ],
      "budget_per_day": 50,
      "trip_duration": 3,
      "group_size": 1,
      "activity_types": [
        "walking tour"
      ],
      "specific_sites": [
        "curaçaoan restaurant"
      ],
      "cuisine_preferences": [
        "keshi yena",
        "stoba"
      ]
    }
  }
```
- **Output**: HTMLResponse con recomendaciones
```html

  <div>
      <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">¡Bienvenidos a Curaçao, la joya del Caribe! Como guía turístico experto, me complace presentarles un emocionante itinerario de 3 días para explorar esta encantadora isla. Durante nuestro recorrido, visitaremos sitios reconocidos por la UNESCO, vibrantes mercados locales y la pintoresca ciudad de Willemstad. Este exclusivo viaje está diseñado para una persona que disfruta de la arquitectura colonial, la fotografía y la inmersión en la cultura local. Con un presupuesto diario de $50.0, garantizamos una experiencia inolvidable llena de descubrimientos y momentos únicos. ¡Prepárate para vivir Curaçao de una manera única y auténtica! ¡Comencemos esta aventura juntos!</p>
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px;">
          
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Spice Garden" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Spice Garden</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Authentic Indian food with freshly ground spices.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.8/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $45.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 11:00-22:00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> spice enthusiasts</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Spice Garden" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Spice Garden</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Authentic Indian food with freshly ground spices.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.8/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $45.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 11:00-22:00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> spice enthusiasts</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Curaçao Carnival Dance Workshop" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Curaçao Carnival Dance Workshop</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.7/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $50.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> culture enthusiasts, groups</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Curaçao Carnival Dance Workshop" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Curaçao Carnival Dance Workshop</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.7/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $50.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> culture enthusiasts, groups</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Historic Willemstad Walking Tour" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Historic Willemstad Walking Tour</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Discover UNESCO-listed architecture, colorful street art, and hidden gems in Willemstad.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.6/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $40.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> history lovers</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cash, Credit Card</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Historic Willemstad Walking Tour" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Historic Willemstad Walking Tour</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Discover UNESCO-listed architecture, colorful street art, and hidden gems in Willemstad.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.6/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $40.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> history lovers</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cash, Credit Card</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Willemstad Art Walk" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Willemstad Art Walk</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.5/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $30.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> art lovers, families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Willemstad Art Walk" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Willemstad Art Walk</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.5/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $30.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> art lovers, families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Progressive 5thgeneration application" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Progressive 5thgeneration application</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">An ideal location for families to relax.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.0/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $5.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 09:00 - 18:00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> No Parking | <strong>Formas de pago:</strong> Credit Card Only</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Reactive 24hour moratorium" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Reactive 24hour moratorium</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">An ideal location for families to relax.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.0/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $5.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 09:00 - 18:00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> No Parking | <strong>Formas de pago:</strong> Credit Card Only</p></div>
      </div>
  </div>

      </div>
  </div>
```
#### POST `/recommendations/html-pro`
- **Descripción**: Obtiene recomendaciones turísticas personalizadas, endpoint basado en `/recommendations/` que mejora la salida mostrando el resultado en cards contruidas con html
- **Input**: RecommendationRequest (query + preferences)
```json
  {
    "query": "Discover the colonial architecture and vibrant local culture of Willemstad. Join a photography-focused walking tour through colorful streets, UNESCO sites, and bustling markets. End the day savoring local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Trip duration: 3 days. Budget: $50 per day.",
    "preferences": {
      "interests": [
        "colonial architecture",
        "photography",
        "local culture"
      ],
      "locations": [
        "unesco sites",
        "bustling markets",
        "willemstad"
      ],
      "budget_per_day": 50,
      "trip_duration": 3,
      "group_size": 1,
      "activity_types": [
        "walking tour"
      ],
      "specific_sites": [
        "curaçaoan restaurant"
      ],
      "cuisine_preferences": [
        "keshi yena",
        "stoba"
      ]
    }
  }
```
- **Output**: HTMLResponse con recomendaciones
```html
  <div>
  <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">¡Bienvenidos a Curaçao, la joya del Caribe! Como guía turístico experto, me complace ofrecerte un emocionante itinerario de 3 días para explorar los sitios de la UNESCO, los animados mercados y la encantadora ciudad de Willemstad. Este viaje personalizado está diseñado para un viajero en busca de la arquitectura colonial, la fotografía y la auténtica cultura local. Con un presupuesto de $50.0 por día, te invito a sumergirte en la rica historia y la belleza de esta increíble isla. ¡Prepárate para una experiencia inolvidable en Curaçao!</p>
  <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px;">
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Spice Garden" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Spice Garden</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Authentic Indian food with freshly ground spices.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.8/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $45.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 11:00-22:00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> spice enthusiasts</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cards</p></div>
  </div>
  <div class="description-content"><p class="description">Spice Garden in Curaçao holds historical and cultural significance as a place where visitors can explore the island's rich culinary traditions and the influence of various cultures on its cuisine. For travelers interested in colonial architecture and local culture, a photography-focused walking tour through Willemstad's colorful streets and UNESCO sites is a must. End the day by indulging in local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Practical tip: Try to visit during local food festivals to experience a variety of flavors and culinary delights unique to Curaçao.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to Spice Garden in Curaçao is to try the local spice-infused cocktails offered at the bar. The bartenders are skilled at incorporating traditional island flavors like tamarind, lemongrass, and cactus into their creations, providing a unique and refreshing taste of the island's culinary heritage. It's a great way to immerse yourself in the local culture while enjoying a delicious drink in a beautiful garden setting.</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Spice Garden" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Spice Garden</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Authentic Indian food with freshly ground spices.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.8/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $45.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 11:00-22:00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> spice enthusiasts</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cards</p></div>
  </div>
  <div class="description-content"><p class="description">Spice Garden in Curaçao holds historical and cultural significance as a place where visitors can explore the island's rich culinary traditions and the influence of various cultures on its cuisine. For travelers interested in colonial architecture and local culture, a photography-focused walking tour through Willemstad's colorful streets and UNESCO sites is a must. End the day by indulging in local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Practical tip: Try to visit during local food festivals to experience a variety of flavors and culinary delights unique to Curaçao.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to Spice Garden in Curaçao is to try the local spice-infused cocktails offered at the bar. The bartenders are skilled at incorporating traditional island flavors like tamarind, lemongrass, and cactus into their creations, providing a unique and refreshing taste of the island's culinary heritage. It's a great way to immerse yourself in the local culture while enjoying a delicious drink in a beautiful garden setting.</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Curaçao Carnival Dance Workshop" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Curaçao Carnival Dance Workshop</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.7/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $50.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> culture enthusiasts, groups</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
  </div>
  <div class="description-content"><p class="description">Immerse yourself in the vibrant spirit of Curaçao Carnival with a dance workshop, a cultural tradition rich in history and celebration. Connect with the local culture by exploring Willemstad's colonial architecture and colorful streets on a photography-focused walking tour. End your day by savoring authentic Curaçaoan dishes like stoba and keshi yena at a traditional restaurant. Don't forget to pack comfortable shoes and clothing for the workshop, and be prepared to embrace the lively rhythms of the island's Carnival festivities.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to the Curaçao Carnival Dance Workshop is to come prepared to immerse yourself fully in the vibrant and energetic atmosphere of traditional Curaçaoan dance. Be ready to let go, move your body, and embrace the infectious rhythms of the music. Don't be shy to join in and follow the lead of the local instructors, as they will help you experience the true essence of Curaçao's rich cultural heritage through dance. Remember to wear comfortable clothing and shoes to fully enjoy the workshop!</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Curaçao Carnival Dance Workshop" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Curaçao Carnival Dance Workshop</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.7/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $50.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> culture enthusiasts, groups</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
  </div>
  <div class="description-content"><p class="description">Immerse yourself in the vibrant spirit of Curaçao Carnival with a dance workshop, a cultural tradition rich in history and celebration. Connect with the local culture by exploring Willemstad's colonial architecture and colorful streets on a photography-focused walking tour. End your day by savoring authentic Curaçaoan dishes like stoba and keshi yena at a traditional restaurant. Don't forget to pack comfortable shoes and clothing for the workshop, and be prepared to embrace the lively rhythms of the island's Carnival festivities.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to the Curaçao Carnival Dance Workshop is to come prepared to immerse yourself fully in the vibrant and energetic atmosphere of traditional Curaçaoan dance. Be ready to let go, move your body, and embrace the infectious rhythms of the music. Don't be shy to join in and follow the lead of the local instructors, as they will help you experience the true essence of Curaçao's rich cultural heritage through dance. Remember to wear comfortable clothing and shoes to fully enjoy the workshop!</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Historic Willemstad Walking Tour" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Historic Willemstad Walking Tour</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Discover UNESCO-listed architecture, colorful street art, and hidden gems in Willemstad.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.6/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $40.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> history lovers</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cash, Credit Card</p></div>
  </div>
  <div class="description-content"><p class="description">Immerse yourself in Curaçao's rich history and vibrant culture on a Historic Willemstad Walking Tour. Explore the UNESCO sites and colonial architecture through a photography-focused experience. Don't miss the opportunity to savor local delicacies like keshi yena and stoba at traditional Curaçaoan restaurants. Remember to wear comfortable walking shoes and stay hydrated while enjoying the colorful streets and bustling markets of Willemstad.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One expert tip for visitors to the Historic Willemstad Walking Tour is to make sure to wear comfortable walking shoes and stay hydrated. The tour involves exploring the colorful streets and historic sites of Willemstad, which can involve quite a bit of walking under the Caribbean sun. Staying comfortable and hydrated will ensure you can fully enjoy the beauty and history of this charming city. Additionally, don't forget your camera to capture the picturesque architecture and vibrant atmosphere of Willemstad!</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Historic Willemstad Walking Tour" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Historic Willemstad Walking Tour</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Discover UNESCO-listed architecture, colorful street art, and hidden gems in Willemstad.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.6/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $40.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> history lovers</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cash, Credit Card</p></div>
  </div>
  <div class="description-content"><p class="description">Immerse yourself in Curaçao's rich history and vibrant culture on a Historic Willemstad Walking Tour. Explore the UNESCO sites and colonial architecture through a photography-focused experience. Don't miss the opportunity to savor local delicacies like keshi yena and stoba at traditional Curaçaoan restaurants. Remember to wear comfortable walking shoes and stay hydrated while enjoying the colorful streets and bustling markets of Willemstad.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One expert tip for visitors to the Historic Willemstad Walking Tour is to make sure to wear comfortable walking shoes and stay hydrated. The tour involves exploring the colorful streets and historic sites of Willemstad, which can involve quite a bit of walking under the Caribbean sun. Staying comfortable and hydrated will ensure you can fully enjoy the beauty and history of this charming city. Additionally, don't forget your camera to capture the picturesque architecture and vibrant atmosphere of Willemstad!</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Willemstad Art Walk" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Willemstad Art Walk</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.5/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $30.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> art lovers, families</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
  </div>
  <div class="description-content"><p class="description">Embark on the Willemstad Art Walk to explore the historical significance of the colorful colonial architecture and vibrant local culture. This photography-focused walking tour takes you through UNESCO sites and bustling markets, ending with a taste of traditional Curaçaoan cuisine like stoba and keshi yena. For practical tips, wear comfortable shoes, stay hydrated, and immerse yourself in the rich cultural elements of the island during this 3-day trip within a $50 per day budget.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to the Willemstad Art Walk is to take your time exploring the vibrant street art and murals that adorn the city. Many of these pieces are not only visually stunning but also carry important messages about the history, culture, and social issues of Curaçao. By taking the time to appreciate and understand the significance of these artworks, you can gain a deeper insight into the rich tapestry of the island's artistic expression.</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Willemstad Art Walk" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Willemstad Art Walk</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.5/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $30.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> art lovers, families</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
  </div>
  <div class="description-content"><p class="description">Embark on the Willemstad Art Walk to explore the historical significance of the colorful colonial architecture and vibrant local culture. This photography-focused walking tour takes you through UNESCO sites and bustling markets, ending with a taste of traditional Curaçaoan cuisine like stoba and keshi yena. For practical tips, wear comfortable shoes, stay hydrated, and immerse yourself in the rich cultural elements of the island during this 3-day trip within a $50 per day budget.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to the Willemstad Art Walk is to take your time exploring the vibrant street art and murals that adorn the city. Many of these pieces are not only visually stunning but also carry important messages about the history, culture, and social issues of Curaçao. By taking the time to appreciate and understand the significance of these artworks, you can gain a deeper insight into the rich tapestry of the island's artistic expression.</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Progressive 5thgeneration application" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Progressive 5thgeneration application</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">An ideal location for families to relax.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.0/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $5.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 09:00 - 18:00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> families</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> No Parking | <strong>Formas de pago:</strong> Credit Card Only</p></div>
  </div>
  <div class="description-content"><p class="description">Experience the rich historical and cultural significance of Curaçao by exploring Willemstad's colonial architecture and local culture through a photography-focused walking tour. Immerse yourself in the vibrant atmosphere of UNESCO sites and bustling markets, then indulge in traditional Curaçaoan dishes like keshi yena and stoba at a local restaurant. For a 3-day trip on a $50 per day budget, this itinerary offers a perfect blend of discovery and culinary delights. Remember to wear comfortable shoes and carry a reusable water bottle to stay hydrated while exploring the colorful streets of Willemstad. Don't miss the chance to attend local events or try other traditional foods like keshi yena and stoba for a complete cultural experience.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to Progressive 5thgeneration application in Curaçao is to make sure you download the application before your trip and familiarize yourself with its features. This will allow you to easily navigate the island, discover hidden gems, and access real-time information on events, activities, and local businesses. Additionally, be sure to enable notifications to stay updated on any special offers or promotions available through the app during your stay in Curaçao.</p></div>
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
  <img alt="Imagen de Reactive 24hour moratorium" src="https://via.placeholder.com/400" style="width: 100%; border-radius: 8px 8px 0 0;"/>
  <div style="padding: 12px;">
  <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Reactive 24hour moratorium</h3>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
  <p style="margin: 0 0 8px; color: #555;">An ideal location for families to relax.</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.0/5)</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $5.00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 09:00 - 18:00</p>
  <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> families</p>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> No Parking | <strong>Formas de pago:</strong> Credit Card Only</p></div>
  </div>
  <div class="description-content"><p class="description">Experience the Reactive 24hour moratorium in Curaçao by immersing yourself in the historical and cultural significance of Willemstad. Delve into the colonial architecture and vibrant local culture through a photography-focused walking tour, exploring UNESCO sites and bustling markets. End each day with a taste of traditional Curaçaoan cuisine like stoba and keshi yena. For practical tips, wear comfortable shoes, stay hydrated, and be respectful of local customs. Don't miss the opportunity to savor the unique flavors of keshi yena and stoba during your 3-day trip with a budget of $50 per day.</p></div><p class="expert-tip" style="font-style: italic; color: #666;">🎯 Pro Tip: One specific expert tip for visitors to Reactive 24hour moratorium is to make sure to wear comfortable shoes as you explore the area. The terrain can be uneven in some parts, so having sturdy footwear will ensure you can navigate the surroundings comfortably and safely. Additionally, be sure to bring plenty of water and sunscreen, as the sun can be strong in Curaçao. Enjoy your visit!</p></div>
  </div>
  </div>
  <div class="intro"><p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">Welcome to Willemstad, Curaçao! Immerse yourself in the UNESCO World Heritage-listed charm of this vibrant city as you wander through its colorful streets and bustling markets on a photography-focused walking tour. Indulge in the rich flavors of Curaçaoan cuisine, savoring dishes like keshi yena and stoba at a traditional local restaurant. Let's embark on a 3-day journey filled with colonial architecture, cultural delights, and unforgettable experiences, all within a budget of $50 per day.</p></div>
```

#### POST `/recommendations/nlp`
- **Descripción**: Endpoint conversacional con análisis de intenciones, endpoint basado en `/recommendations/` que permite identificar las preferencias a partor de un texto natural, dando como salida un JSONResponse
- **Input**: ChatRequest (text)
```json
  {
    "text": "Discover the colonial architecture and vibrant local culture of Willemstad. Join a photography-focused walking tour through colorful streets, UNESCO sites, and bustling markets. End the day savoring local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Trip duration: 3 days. Budget: $50 per day."
  }
```
- **Output**: JSONResponse con recomendaciones (query + preferences)
```json
  {
    "query": "Discover the colonial architecture and vibrant local culture of Willemstad. Join a photography-focused walking tour through colorful streets, UNESCO sites, and bustling markets. End the day savoring local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Trip duration: 3 days. Budget: $50 per day.",
    "preferences": {
      "interests": [
        "colonial architecture",
        "photography",
        "vibrant local culture"
      ],
      "locations": [
        "unesco sites",
        "willemstad",
        "bustling markets"
      ],
      "budget_per_day": 50,
      "trip_duration": 3,
      "group_size": 1,
      "activity_types": [
        "walking tour"
      ],
      "specific_sites": [
        "unesco sites",
        "willemstad",
        "bustling markets"
      ],
      "cuisine_preferences": [
        "stoba",
        "local dishes",
        "traditional curaçaoan cuisine",
        "keshi yena"
      ]
    }
  }
```


#### POST `/recommendations/full`
- **Descripción**: Combina análisis NLP y generación de HTML, endpoint basado en `/recommendations/npl` y `/recommendations/html`, que permite procesar texto natural convertirlo en un JSONResponse con recomendaciones (query + preferences) el cual es usado para dar las recomendaciones segun la entra del modelo interno.
- **Input**: NLPRequest (text)
```json
  {
    "text": "Discover the colonial architecture and vibrant local culture of Willemstad. Join a photography-focused walking tour through colorful streets, UNESCO sites, and bustling markets. End the day savoring local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Trip duration: 3 days. Budget: $50 per day."
  }
```
- **Output**: HTMLResponse con recomendaciones completas
```html

  <div>
      <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">¡Bienvenidos a Curaçao, la joya del Caribe! Como guía turístico experto, me complace acompañarlos en esta emocionante aventura de 3 días por la fascinante ciudad de Willemstad. Durante este tiempo, exploraremos la rica herencia cultural, disfrutaremos de la deliciosa gastronomía local, nos sumergiremos en experiencias especiales y viviremos auténticas experiencias locales. Todo esto, sin salirnos de su presupuesto de $0 por día. ¡Prepárense para descubrir los encantos de Curaçao de la mano de un guía apasionado por compartir lo mejor de este destino con ustedes! ¡Comencemos nuestra inolvidable travesía juntos!</p>
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px;">
          
  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Spice Garden" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Spice Garden</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Authentic Indian food with freshly ground spices.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.8/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $45.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 11:00-22:00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> spice enthusiasts</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Spice Garden" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Spice Garden</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Authentic Indian food with freshly ground spices.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.8/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $45.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 11:00-22:00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> spice enthusiasts</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Culinary Exploration of Curaçao" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Culinary Exploration of Curaçao</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Savor local dishes like keshi yena and discover the island's vibrant culinary scene.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.6/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $70.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> food enthusiasts, families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Art and Culture Kayaking Tour" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Art and Culture Kayaking Tour</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Kayak along the coastline, stopping at cultural landmarks and galleries.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.6/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $60.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Media</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> art lovers, adventurers</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Moderate | <strong>Estacionamiento:</strong> Street | <strong>Formas de pago:</strong> Cash, Credit Card</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Culinary Exploration of Curaçao" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Culinary Exploration of Curaçao</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Savor local dishes like keshi yena and discover the island's vibrant culinary scene.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.6/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $70.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Alta</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> food enthusiasts, families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Curaçao Carnival Dance Workshop" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Curaçao Carnival Dance Workshop</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.7/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $50.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Baja</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> culture enthusiasts, groups</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Curaçao Carnival Dance Workshop" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Curaçao Carnival Dance Workshop</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Learn the vibrant dance moves of Curaçao's Carnival in a fun and energetic workshop.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.7/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $50.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Baja</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> culture enthusiasts, groups</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Willemstad Art Walk" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Willemstad Art Walk</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.5/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $30.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Baja</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> art lovers, families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Willemstad Art Walk" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Willemstad Art Walk</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">Discover colorful murals and meet local artists during this guided walking tour through Willemstad's art scene.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.5/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $30.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Baja</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> No disponible</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> art lovers, families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Easy | <strong>Estacionamiento:</strong> Paid Parking | <strong>Formas de pago:</strong> All Cards</p></div>
      </div>
  </div>

  <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
      <img src="https://via.placeholder.com/400" alt="Imagen de Progressive 5thgeneration application" style="width: 100%; border-radius: 8px 8px 0 0;">
      <div style="padding: 12px;">
          <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">Progressive 5thgeneration application</h3>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> Willemstad</p>
          <p style="margin: 0 0 8px; color: #555;">An ideal location for families to relax.</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> ★★★★☆ (4.0/5)</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> $5.00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> Baja</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> 09:00 - 18:00</p>
          <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> families</p>
          <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;"><p style="margin: 0 0 8px; color: #555;"><strong>Accesibilidad:</strong> Wheelchair Accessible | <strong>Estacionamiento:</strong> No Parking | <strong>Formas de pago:</strong> Credit Card Only</p></div>
      </div>
  </div>

      </div>
  </div>
```

#### POST `/recommendations/chat`
- **Descripción**: Endpoint conversacional con análisis de intenciones, esta es la version final del proceso de chatbor de recomendaciones turisticas que integra todas la mejoras de los otros endpoints.
- **Input**: ChatRequest (text)
```json
  {
    "text": "Discover the colonial architecture and vibrant local culture of Willemstad. Join a photography-focused walking tour through colorful streets, UNESCO sites, and bustling markets. End the day savoring local dishes like stoba and keshi yena at a traditional Curaçaoan restaurant. Trip duration: 3 days. Budget: $50 per day."
  }
```
- **Output**: HTMLResponse con respuesta formateada
```html
  
  <div class="message message-bot">
      <div class="message-content">
          <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">
              ¡Hola! ¡Bienvenido a Curazao! Me alegra saber que estás interesado en descubrir la arquitectura colonial y la vibrante cultura local de Willemstad. Un recorrido a pie enfocado en la fotografía suena fascinante, te permitirá capturar la belleza de nuestras calles coloridas, sitios de la UNESCO y animados mercados. 

Además, probar platos locales como stoba y keshi yena en un restaurante tradicional de Curazao es una experiencia que definitivamente no te puedes perder. 

Si necesitas alguna recomendación adicional o ayuda para planificar tu viaje de 3 días con un presupuesto diario de $50, ¡aquí estoy para ayudarte! ¡No dudes en preguntar!
          </p>
      </div>
  </div>
```

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