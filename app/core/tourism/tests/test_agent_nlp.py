import pytest
from types import SimpleNamespace
import json
from unittest.mock import AsyncMock, patch, MagicMock
import importlib
import logging

# Importamos los módulos y clases necesarios
from app.core.tourism.agents.agent_nlp import NLPAgent
from app.core.recommender.text_formatter import TextFormattedRequest
import app.core.tourism.agents.agent_nlp as nlp_module
from app.utils.logger import get_logger

# Configuración para mostrar mensajes de diagnóstico durante las pruebas
VERBOSE_TESTING = True

def print_test_info(message):
    """Función auxiliar para imprimir información de test"""
    if VERBOSE_TESTING:
        print(f"\n{'=' * 80}\n{message}\n{'=' * 80}")

# Configuramos logging para capturar los logs durante el test
@pytest.fixture
def mock_logger():
    """Fixture que proporciona un logger mockeado para capturar los logs"""
    with patch('app.utils.logger.get_logger') as mock_get_logger:
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        yield logger_mock

# Ejemplo de respuesta real formateada (basada en el JSON proporcionado)
def generate_realistic_recommendation(query_type):
    """Genera una recomendación realista basada en el tipo de consulta"""
    if "playa" in query_type.lower():
        return """
        RECOMENDACIONES TURÍSTICAS
        ==========================

        PLAYA KENEPA (GROTE KNIP)
        -------------------------
        Type: Beach
        Location: West Side
        Cost: $0.00
        Rating: ★★★★★ (4.9/5)

        Description:
        Considerada una de las más hermosas playas de Curazao, con aguas
        cristalinas de color turquesa y arena blanca. Perfecta para nadar
        y hacer snorkel. Ofrece vistas panorámicas impresionantes de la costa.

        ========================================

        PLAYA CAS ABAO
        -------------
        Type: Beach
        Location: West Side
        Cost: $6.00
        Rating: ★★★★☆ (4.8/5)

        Description:
        Playa de arena blanca con aguas tranquilas y cristalinas. Cuenta con
        instalaciones como hamacas, sombrillas, restaurante y servicios. Ideal
        para familias y práctica de snorkel por sus coloridos arrecifes.

        ========================================

        PLAYA PORTO MARI
        ---------------
        Type: Beach
        Location: Banda Bou
        Cost: $5.00
        Rating: ★★★★☆ (4.7/5)

        Description:
        Hermosa playa con infraestructura completa que incluye restaurante,
        bar, duchas y alquiler de equipos de snorkel. Hogar del famoso doble
        arrecife, perfecto para observar peces tropicales a poca profundidad.
        """
    
    elif "hotel" in query_type.lower():
        return """
        RECOMENDACIONES DE ALOJAMIENTO
        ==============================

        BAOASE LUXURY RESORT
        -------------------
        Type: Luxury Resort
        Location: Willemstad
        Cost: $650.00
        Rating: ★★★★★ (4.9/5)

        Description:
        Resort de lujo con acceso directo a una playa privada de arena blanca.
        Ofrece suites elegantes, algunas con piscinas privadas, y villas de lujo.
        Restaurante gourmet, spa y servicio personalizado de primer nivel.

        ========================================

        AVILA BEACH HOTEL
        ---------------
        Type: Beach Hotel
        Location: Pietermaai
        Cost: $220.00
        Rating: ★★★★☆ (4.6/5)

        Description:
        El hotel más antiguo de Curazao, con encanto colonial y modernas comodidades.
        Ubicado junto al mar con dos playas privadas. Destacan sus habitaciones con
        vistas al mar y su muelle privado perfecto para nadar.

        ========================================

        RENAISSANCE CURAÇAO RESORT
        -------------------------
        Type: Urban Resort
        Location: Willemstad
        Cost: $190.00
        Rating: ★★★★☆ (4.3/5)

        Description:
        Ubicado junto al centro histórico de Willemstad, ofrece la exclusiva
        playa infinity con arena de Curazao. Cuenta con casino, centro comercial,
        piscinas y acceso directo a las atracciones del centro.
        """
    
    elif "actividades" in query_type.lower() or "familia" in query_type.lower():
        return """
        PLAN FAMILIAR DE 5 DÍAS EN CURAZAO
        ==================================

        DÍA 1: EXPLORACIÓN DE WILLEMSTAD
        --------------------------------
        Type: Cultural
        Location: Willemstad
        Cost: $40.00
        Rating: ★★★★☆ (4.6/5)

        Description:
        Mañana: Visita al Museo Marítimo para aprender sobre la historia naval de la isla.
        Mediodía: Almuerzo en un restaurante local con vistas al canal.
        Tarde: Paseo por el puente flotante Queen Emma y exploración de los barrios
        históricos de Punda y Otrobanda con sus coloridas fachadas.

        ========================================

        DÍA 2: PLAYA PORTO MARI
        ----------------------
        Type: Beach Day
        Location: Banda Bou
        Cost: $35.00
        Rating: ★★★★★ (4.8/5)

        Description:
        Día completo en Playa Porto Mari, ideal para niños por sus aguas tranquilas
        y poca profundidad cerca de la orilla. Incluye snorkel en el doble arrecife,
        observación de peces tropicales y posibilidad de ver cerdos salvajes en el camino.
        Restaurante en la playa con menú infantil disponible.

        ========================================

        DÍA 3: NATURALEZA Y AVENTURA
        ---------------------------
        Type: Nature & Adventure
        Location: Christoffel Park / Shete Boka
        Cost: $55.00
        Rating: ★★★★☆ (4.7/5)

        Description:
        Mañana: Visita al Parque Nacional Christoffel. Senderos fáciles para niños y
        posibilidad de ver la flora y fauna local, incluidos venados y lagartos.
        Tarde: Recorrido por Shete Boka para observar el espectáculo natural de las
        olas contra las formaciones rocosas y descubrir las pequeñas calas escondidas.
        """
    
    elif "restaurante" in query_type.lower():
        return """
        RECOMENDACIONES GASTRONÓMICAS
        ============================

        KOME
        ----
        Type: Restaurant
        Location: Pietermaai District
        Cost: $65.00
        Rating: ★★★★★ (4.8/5)

        Description:
        Restaurante contemporáneo con enfoque en ingredientes locales y sabores
        internacionales. Ambiente elegante pero relajado, con un menú creativo
        que cambia según la temporada. Excelente selección de vinos y cócteles.

        ========================================

        BLESSING BY SELENE
        -----------------
        Type: Caribbean Fusion
        Location: Jan Thiel
        Cost: $70.00
        Rating: ★★★★★ (4.9/5)

        Description:
        Una de las joyas culinarias de Curazao, ofrece cocina caribeña de autor
        con influencias internacionales. Ambiente íntimo y elegante, perfecto para
        cenas especiales. No te pierdas sus creaciones a base de pescado fresco.

        ========================================

        PLASA BIEU (OLD MARKET)
        ---------------------
        Type: Local Food Court
        Location: Punda, Willemstad
        Cost: $15.00
        Rating: ★★★★☆ (4.5/5)

        Description:
        Auténtico mercado de comida local donde puedes probar verdaderos platos
        curazoleños como kadushi (sopa de cactus), kabritu (cabra estofada) y
        funchi (similar a la polenta). Experiencia gastronómica y cultural genuina.
        """
    
    elif not query_type:  # Entrada vacía
        return "Por favor, proporcione alguna información sobre su interés en Curazao para poder ofrecerle recomendaciones personalizadas."
    
    else:  # Consulta sobre información general
        return """
        INFORMACIÓN SOBRE CURAZAO
        ========================

        CURAZAO (CURAÇAO)
        ----------------
        Type: Caribbean Island
        Location: Southern Caribbean, near Venezuela
        Currency: Florin antillano / US Dollar
        Languages: Papiamento, Dutch, English, Spanish

        Description:
        Curazao es una isla caribeña de origen holandés, famosa por sus playas
        de aguas cristalinas, sus coloridas construcciones coloniales y su rica
        mezcla cultural. La capital, Willemstad, fue declarada Patrimonio de la
        Humanidad por la UNESCO por su arquitectura única que combina estilos
        holandeses y caribeños.

        La isla ofrece excelentes opciones para el buceo y snorkel gracias a
        sus arrecifes de coral bien conservados, así como una vibrante escena
        gastronómica que fusiona influencias europeas, caribeñas y latinoamericanas.

        El clima es tropical durante todo el año, con una temporada seca de
        enero a septiembre y una temporada más lluviosa de octubre a diciembre,
        aunque las lluvias suelen ser breves. ¡Disfrútala!
        """

# Clase para simular diferentes tipos de respuestas
class MockResponse:
    def __init__(self, content, as_bytes=True, has_body=True):
        """
        Inicializa una respuesta simulada
        
        Args:
            content: El contenido de la respuesta
            as_bytes: Si es True, el contenido se almacena como bytes, sino como string
            has_body: Si es True, la respuesta tiene un atributo body, sino es un objeto simple
        """
        if as_bytes and has_body:
            self.body = content if isinstance(content, bytes) else content.encode('utf-8')
        elif has_body:
            self.body = content
        else:
            # Si no tiene body, el objeto entero es la respuesta
            self.content = content
    
    def __str__(self):
        """Representación en string para el caso en que no tiene body"""
        if hasattr(self, 'content'):
            return self.content
        return "MockResponse object"

# Lista de casos de prueba para parametrizar
TOURISM_TEST_CASES = [
    # Caso básico: pregunta simple sobre playas
    {
        "name": "basic_beaches_query",
        "user_input": "¿Cuáles son las mejores playas en Curazao para visitar?",
        "query_type": "playa",
        "response_type": {"as_bytes": True, "has_body": True},
        "should_fail": False
    },
    # Caso complejo: consulta detallada sobre actividades
    {
        "name": "complex_activities_query",
        "user_input": "Estoy planeando un viaje a Curazao de 5 días con mi familia incluyendo "
                     "niños de 8 y 10 años. Nos gustaría combinar actividades de playa, "
                     "naturaleza y cultura. ¿Qué nos recomiendas?",
        "query_type": "actividades familia",
        "response_type": {"as_bytes": True, "has_body": True},
        "should_fail": False
    },
    # Caso de respuesta sin body
    {
        "name": "no_body_response",
        "user_input": "¿Qué hoteles recomiendas en Curazao?",
        "query_type": "hotel",
        "response_type": {"as_bytes": False, "has_body": False},
        "should_fail": False
    },
    # Caso de entrada vacía
    {
        "name": "empty_input",
        "user_input": "",
        "query_type": "",
        "response_type": {"as_bytes": True, "has_body": True},
        "should_fail": False
    },
    # Caso de error en el servicio
    {
        "name": "service_error",
        "user_input": "Cuéntame sobre restaurantes en Curazao",
        "query_type": "restaurante",
        "response_type": {"as_bytes": True, "has_body": True},
        "should_fail": True
    }
]

@pytest.mark.parametrize("test_case", TOURISM_TEST_CASES, ids=[tc["name"] for tc in TOURISM_TEST_CASES])
@pytest.mark.asyncio
async def test_nlp_agent_with_tourism_queries(test_case, mock_logger, monkeypatch):
    """
    Test parametrizado que prueba el NLPAgent con diferentes consultas turísticas
    
    Args:
        test_case: Diccionario con los datos del caso de prueba
        mock_logger: Logger mockeado para verificar los logs
        monkeypatch: Fixture de pytest para parchear funciones
    """
    # Imprimimos información sobre el caso de prueba
    print_test_info(f"EJECUTANDO CASO: {test_case['name']}")
    
    # Extraemos datos del caso de prueba
    user_input = test_case["user_input"]
    query_type = test_case["query_type"]
    as_bytes = test_case["response_type"]["as_bytes"]
    has_body = test_case["response_type"]["has_body"]
    should_fail = test_case["should_fail"]
    
    # Generamos una respuesta realista basada en el tipo de consulta
    response_content = generate_realistic_recommendation(query_type)
    
    print(f"ENTRADA: '{user_input}'")
    print(f"TIPO DE CONSULTA: {query_type}")
    print(f"CONFIGURACIÓN: as_bytes={as_bytes}, has_body={has_body}, should_fail={should_fail}")
    
    # Mostramos un resumen de la respuesta (no toda para evitar texto excesivo)
    print(f"RESPUESTA ESPERADA (resumen): '{response_content.strip()[:100]}...'")
    
    # Creamos una función mock que simula get_text_formatted_recommendations
    async def mock_get_text_formatted_recommendations(request: TextFormattedRequest):
        # Verificamos que el request contiene el texto esperado
        assert request.text == user_input
        print(f"SERVICIO RECIBIÓ: '{request.text}'")
        
        if should_fail:
            print("SIMULANDO ERROR DEL SERVICIO")
            raise Exception("Simulated service error")
        
        print(f"SERVICIO DEVUELVE RECOMENDACIÓN DE TIPO: '{query_type}'")
        return MockResponse(response_content, as_bytes=as_bytes, has_body=has_body)
    
    # Reemplazamos la función real con nuestra mock
    monkeypatch.setattr(
        "app.core.recommender.text_formatter.get_text_formatted_recommendations",
        mock_get_text_formatted_recommendations
    )
    
    # Recargamos el módulo para que tome la función mockeada
    importlib.reload(nlp_module)
    
    # Creamos el agente
    agent = nlp_module.NLPAgent()
    
    # Creamos un estado con la entrada del usuario
    state = {
        "user_input": user_input,
        "memory": SimpleNamespace(preferences={}),
        "preferences": {},
    }
    
    print(f"ESTADO INICIAL: user_input='{state['user_input']}'")
    
    # Procesamos el estado con el agente
    updated_state = await agent.process(state)
    
    # Verificaciones según si debería fallar o no
    if should_fail:
        # Verificamos que se registró el error
        mock_logger.error.assert_called_once()
        
        # Verificamos que el estado contiene mensajes de error
        error_msg = "Error generating recommendations."
        assert updated_state["formatted_recommendations"] == error_msg
        assert updated_state["response"] == error_msg
        print(f"RESULTADO (ERROR): '{error_msg}'")
    else:
        # Verificamos que la respuesta está en el estado
        assert "formatted_recommendations" in updated_state
        
        # Si la respuesta tiene body y es bytes, decodificamos para comparar
        expected_response = response_content
        if as_bytes and has_body:
            if isinstance(response_content, bytes):
                expected_response = response_content.decode('utf-8')
        elif not has_body:
            expected_response = response_content
            
        assert updated_state["formatted_recommendations"] == expected_response
        assert updated_state["response"] == expected_response
        
        # Mostramos un resumen de la respuesta real
        print(f"RESULTADO (resumen): '{updated_state['formatted_recommendations'].strip()[:100]}...'")
        print("\nRESPUESTA COMPLETA:")
        print(f"{updated_state['formatted_recommendations']}")
        
        # Verificamos que se registró el debug log
        if user_input:  # Solo verificamos esto si hay entrada de usuario
            mock_logger.debug.assert_called_once()

@pytest.mark.asyncio
async def test_nlp_agent_with_real_preferences():
    """
    Test que prueba el NLPAgent con preferencias reales en el estado
    para asegurar que el agente las maneja correctamente aunque no las use
    """
    print_test_info("EJECUTANDO CASO: test_nlp_agent_with_real_preferences")
    
    # Preparamos la respuesta mockeada antes de crear el mock
    content = generate_realistic_recommendation("playa")
    mock_response = MockResponse(content, as_bytes=True, has_body=True)
    
    print(f"RESPUESTA ESPERADA (con preferencias, resumen): '{content.strip()[:100]}...'")
    
    # Creamos un mock para la función que devuelve nuestra respuesta preparada
    mock_service = AsyncMock(return_value=mock_response)
    
    # Aplicamos el patch directamente al módulo correcto
    with patch("app.core.tourism.agents.agent_nlp.get_text_formatted_recommendations", mock_service):
        # Creamos el agente después de aplicar el mock
        agent = NLPAgent()
        
        # Creamos un estado con preferencias reales
        preferences = {
            "activities": ["beach", "snorkeling", "culture"],
            "budget": "medium",
            "travelDuration": 7,
            "travelWith": "family",
            "cuisine": ["local", "seafood"]
        }
        
        state = {
            "user_input": "Necesito recomendaciones para Curazao",
            "memory": SimpleNamespace(preferences=preferences),
            "preferences": preferences,
        }
        
        print(f"ENTRADA: '{state['user_input']}'")
        print(f"PREFERENCIAS: {json.dumps(preferences, indent=2)}")
        
        # Procesamos el estado
        updated_state = await agent.process(state)
        
        # Verificamos que se llamó al mock correctamente
        mock_service.assert_called_once()
        print("SERVICIO LLAMADO CORRECTAMENTE")
        
        # Verificamos que las preferencias se mantienen en el estado
        assert updated_state["preferences"] == preferences
        assert updated_state["memory"].preferences == preferences
        print("PREFERENCIAS PRESERVADAS CORRECTAMENTE")
        
        # Y que la respuesta está presente
        assert updated_state["formatted_recommendations"] == content
        assert updated_state["response"] == content
        print(f"RESULTADO (resumen): '{updated_state['formatted_recommendations'].strip()[:100]}...'")
        print("\nRESPUESTA COMPLETA:")
        print(f"{updated_state['formatted_recommendations']}")

@pytest.mark.asyncio
async def test_nlp_agent_unicode_handling():
    """Test específico para verificar el manejo de caracteres Unicode en las respuestas"""
    print_test_info("EJECUTANDO CASO: test_nlp_agent_unicode_handling")
    
    # Usamos el ejemplo con caracteres Unicode
    special_text = generate_realistic_recommendation("general")
    mock_response = MockResponse(special_text.encode('utf-8'), as_bytes=True, has_body=True)
    
    print(f"RESPUESTA ESPERADA (con caracteres Unicode, resumen): '{special_text.strip()[:100]}...'")
    
    # Usamos un AsyncMock para manejar correctamente la función asíncrona
    mock_service = AsyncMock(return_value=mock_response)
    
    # Aplicamos el patch directamente al módulo que usa el agente
    with patch("app.core.tourism.agents.agent_nlp.get_text_formatted_recommendations", mock_service):
        agent = NLPAgent()
        
        user_input = "Háblame sobre Curazao y su patrimonio"
        state = {
            "user_input": user_input,
            "memory": SimpleNamespace(preferences={}),
            "preferences": {},
        }
        
        print(f"ENTRADA (con caracteres Unicode): '{user_input}'")
        
        updated_state = await agent.process(state)
        
        # Verificamos que se llamó al mock correctamente
        mock_service.assert_called_once()
        print("SERVICIO LLAMADO CORRECTAMENTE")
        
        # Verificamos que los caracteres especiales se mantienen intactos
        assert updated_state["formatted_recommendations"] == special_text
        assert "Curaçao" in updated_state["formatted_recommendations"]
        assert "¡Disfrútala!" in updated_state["formatted_recommendations"]
        
        print(f"RESULTADO (resumen): '{updated_state['formatted_recommendations'].strip()[:100]}...'")
        print("\nRESPUESTA COMPLETA:")
        print(f"{updated_state['formatted_recommendations']}")
        print("CARACTERES ESPECIALES PRESERVADOS: 'Curaçao' y '¡Disfrútala!'")