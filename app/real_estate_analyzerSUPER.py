import sqlite3
import json
import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import openai


class RealEstateAnalyzer:
    
    def __init__(self, db_path: str, log_path: str = "logs"):
        self._setup_logging(log_path)
        self._validate_and_connect_db(db_path)
        self.schema = self._get_schema()
        self._init_query_mappings()


    def _validate_and_connect_db(self, db_path: str):
        if not os.path.exists(db_path):
            self.logger.error(f"Base de datos no encontrada en: {db_path}")
            raise FileNotFoundError(f"Base de datos no encontrada en: {db_path}")
        
        self.conn = sqlite3.connect(db_path)
        self.logger.info(f"Conexión exitosa a la base de datos: {db_path}")
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_property';")
        
        if not cursor.fetchone():
            raise Exception("La tabla chat_property no existe en la base de datos")


    def _setup_logging(self, log_path: str):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        log_file = os.path.join(log_path, f'real_estate_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)


    def _get_schema(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%';
        """)
        schema_str = ""
        for (table_name,) in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            schema_str += f"\nTabla: {table_name}\nColumnas:\n"
            for col in columns:
                schema_str += f"  - {col[1]} ({col[2]})\n"
        return schema_str


    # def _init_query_mappings(self):
    #     self.categorias = {
    #         "identificacion_localizacion": ["ubicacion", "localización", "ciudad", "zona"],
    #         "analisis_precio": ["precio", "valor", "costo", "mercado"],
    #         "detalles_propiedad": ["habitaciones", "características", "espacios"],
    #         "clasificacion_categoria": ["categoría", "tipo", "proyecto"],
    #         "cronologia": ["tiempo", "fecha", "histórico", "tendencias"]
    #     }

    #     self.consultas = {
    #         "identificacion_localizacion": {
    #             "info_basica": """
    #                 SELECT id, location, property_type, ROUND(price, 2) as price
    #                 FROM chat_property WHERE {condition}
    #             """
    #         },
    #         "analisis_precio": {
    #             "comparativa_precios": """
    #                 SELECT location, property_type, 
    #                 ROUND(AVG(price), 2) as precio_promedio
    #                 FROM chat_property
    #                 GROUP BY location, property_type
    #             """
    #         }
    #     }

    # def _init_query_mappings(self):
    #     self.categorias = {
    #         "identificacion_localizacion": [
    #             "ubicacion", "localización", "ciudad", "zona", "país", "provincia",
    #             "región", "barrio", "dirección", "geografía",
    #             "donde", "cerca", "próximo", "alrededores", "sector", "vecindario",
    #             "manzana", "calle", "avenida", "distrito", "código postal", "área",
    #             "comuna", "municipio", "departamento", "estado", "territorio"
    #         ],
    #         "analisis_precio": [
    #             "precio", "valor", "costo", "mercado", "tasación", "valoración",
    #             "inversión", "presupuesto", "cotización", "metro cuadrado",
    #             "dólares", "usd", "euros", "moneda", "financiación", "hipoteca",
    #             "cuotas", "entrada", "inicial", "descuento", "oferta", "negociable",
    #             "rebaja", "precio por metro", "rentabilidad", "roi", "retorno",
    #             "ganancia", "apreciación", "valorización"
    #         ],
    #         "detalles_propiedad": [
    #             "habitaciones", "características", "espacios", "dormitorios", 
    #             "baños", "ambientes", "metros", "superficie", "amenities",
    #             "comodidades",
    #             "garage", "estacionamiento", "piscina", "jardín", "terraza",
    #             "balcón", "patio", "sótano", "ático", "equipamiento", "amoblado",
    #             "antigüedad", "año", "construcción", "materiales", "acabados",
    #             "pisos", "techos", "ventanas", "orientación", "vista", "luz",
    #             "natural", "seguridad", "vigilancia", "acceso"
    #         ],
    #         "clasificacion_categoria": [
    #             "categoría", "tipo", "proyecto", "residencial", "comercial",
    #             "clasificación", "nivel", "segmento", "calidad"
    #         ],
    #         "cronologia": [
    #             "tiempo", "fecha", "histórico", "tendencias", "evolución",
    #             "trayectoria", "período", "temporal", "estacional"
    #         ],
    #         "comparativo_mercado": [
    #             "comparación", "benchmark", "competencia", "similar", "equivalente",
    #             "mercado", "oferta", "demanda"
    #         ],
    #         "analisis_geografico": [
    #             "distribución", "concentración", "densidad", "cluster", "zona",
    #             "territorial", "geográfico", "regional"
    #         ],
    #     }

    #     self.consultas = {
    #         "identificacion_localizacion": {
    #             "info_basica": """
    #                 SELECT 
    #                     cp.id, 
    #                     cp.location, 
    #                     cp.property_type, 
    #                     cp.url, 
    #                     cp.square_meters, 
    #                     cp.description, 
    #                     cp.image, 
    #                     cp.num_bedrooms AS promedio_ambientes,
    #                     cp.num_rooms AS promedio_dormitorios,
    #                     ROUND(cp.price, 2) as price,
    #                     cc.name as country_name,
    #                     cd.name as province_name,
    #                     cct.name as city_name
    #                 FROM chat_property cp
    #                 LEFT JOIN chat_country cc ON cp.country_id = cc.id
    #                 LEFT JOIN chat_province cd ON cp.province_id = cd.id
    #                 LEFT JOIN chat_city cct ON cp.city_id = cct.id
    #                 WHERE {condition}
    #             """,
    #             "distribucion_geografica": """
    #                 SELECT cc.name as country_name, 
    #                     COUNT(*) as total_properties,
    #                     ROUND(AVG(cp.price), 2) as avg_price
    #                 FROM chat_property cp
    #                 JOIN chat_country cc ON cp.country_id = cc.id
    #                 GROUP BY cc.name
    #                 ORDER BY total_properties DESC
    #             """
    #         },
    #         "analisis_precio": {
    #             "comparativa_precios": """
    #                 SELECT cp.location, cp.property_type, 
    #                     ROUND(AVG(cp.price), 2) as precio_promedio,
    #                     ROUND(MIN(cp.price), 2) as precio_minimo,
    #                     ROUND(MAX(cp.price), 2) as precio_maximo,
    #                     ROUND(AVG(cp.price/cp.square_meters), 2) as precio_m2
    #                 FROM chat_property cp
    #                 GROUP BY cp.location, cp.property_type
    #             """,
    #             "analisis_por_pais": """
    #                 SELECT cc.name as country_name,
    #                     cp.property_type,
    #                     ROUND(AVG(cp.price), 2) as precio_promedio,
    #                     COUNT(*) as total_propiedades
    #                 FROM chat_property cp
    #                 JOIN chat_country cc ON cp.country_id = cc.id
    #                 GROUP BY cc.name, cp.property_type
    #                 ORDER BY cc.name, precio_promedio DESC
    #             """
    #         },
    #         "detalles_propiedad": {
    #             "estadisticas_tipos": """
    #                 SELECT property_type,
    #                     COUNT(*) as cantidad,
    #                     ROUND(AVG(square_meters), 2) as promedio_m2,
    #                     ROUND(AVG(num_bedrooms), 1) as promedio_dormitorios,
    #                     ROUND(AVG(num_rooms), 1) as promedio_ambientes
    #                 FROM chat_property
    #                 GROUP BY property_type
    #             """,
    #             "distribucion_tamanos": """
    #                 SELECT 
    #                     CASE 
    #                         WHEN square_meters <= 50 THEN 'Pequeño (<=50m2)'
    #                         WHEN square_meters <= 100 THEN 'Mediano (51-100m2)'
    #                         WHEN square_meters <= 200 THEN 'Grande (101-200m2)'
    #                         ELSE 'Muy grande (>200m2)'
    #                     END as categoria_tamano,
    #                     COUNT(*) as cantidad,
    #                     ROUND(AVG(price), 2) as precio_promedio
    #                 FROM chat_property
    #                 GROUP BY categoria_tamano
    #             """
    #         },
    #         "clasificacion_categoria": {
    #             "distribucion_categorias": """
    #                 SELECT project_category,
    #                     COUNT(*) as cantidad,
    #                     ROUND(AVG(price), 2) as precio_promedio
    #                 FROM chat_property
    #                 GROUP BY project_category
    #             """,
    #             "tipos_residencia": """
    #                 SELECT residence_type,
    #                     COUNT(*) as cantidad,
    #                     ROUND(AVG(square_meters), 2) as promedio_m2,
    #                     ROUND(AVG(price), 2) as precio_promedio
    #                 FROM chat_property
    #                 GROUP BY residence_type
    #             """
    #         },
    #         "cronologia": {
    #             "tendencia_temporal": """
    #                 SELECT strftime('%Y-%m', created_at) as mes,
    #                     COUNT(*) as nuevas_propiedades,
    #                     ROUND(AVG(price), 2) as precio_promedio
    #                 FROM chat_property
    #                 GROUP BY mes
    #                 ORDER BY mes
    #             """,
    #             "evolucion_precios": """
    #                 SELECT 
    #                     strftime('%Y-%m', created_at) as mes,
    #                     property_type,
    #                     ROUND(AVG(price/square_meters), 2) as precio_m2_promedio
    #                 FROM chat_property
    #                 GROUP BY mes, property_type
    #                 ORDER BY mes, property_type
    #             """
    #         },
    #         "comparativo_mercado": {
    #             "benchmark_precios": """
    #                 SELECT cp.property_type,
    #                     cc.name as country_name,
    #                     ROUND(AVG(cp.price/cp.square_meters), 2) as precio_m2,
    #                     COUNT(*) as muestra
    #                 FROM chat_property cp
    #                 JOIN chat_country cc ON cp.country_id = cc.id
    #                 GROUP BY cp.property_type, cc.name
    #                 HAVING muestra >= 5
    #             """,
    #             "analisis_competencia": """
    #                 SELECT location,
    #                     property_type,
    #                     COUNT(*) as total_propiedades,
    #                     ROUND(AVG(price), 2) as precio_promedio,
    #                     ROUND(STDDEV(price), 2) as desviacion_precio
    #                 FROM chat_property
    #                 GROUP BY location, property_type
    #                 HAVING total_propiedades >= 3
    #             """
    #         },
    #         "analisis_geografico": {
    #             "concentracion_por_pais": """
    #                 SELECT cc.name as country_name,
    #                     COUNT(*) as total_propiedades,
    #                     COUNT(DISTINCT cp.property_type) as tipos_distintos,
    #                     ROUND(AVG(cp.price), 2) as precio_promedio
    #                 FROM chat_property cp
    #                 JOIN chat_country cc ON cp.country_id = cc.id
    #                 GROUP BY cc.name
    #             """,
    #             "densidad_tipos": """
    #                 SELECT cc.name as country_name,
    #                     cp.property_type,
    #                     COUNT(*) as cantidad,
    #                     ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY cc.name), 2) as porcentaje
    #                 FROM chat_property cp
    #                 JOIN chat_country cc ON cp.country_id = cc.id
    #                 GROUP BY cc.name, cp.property_type
    #             """
    #         }
    #     }
    
    def _init_query_mappings(self):
        self.categorias = {
            "identificacion_localizacion": [
                # Básicos
                "ubicacion", "localización", "ciudad", "zona", "país", 
                "provincia", "región", "barrio", "dirección",
                # Especificadores
                "donde", "cerca", "próximo", "alrededores", "sector",
                # Términos administrativos
                "distrito", "departamento", "estado", "capital", "central",
                "municipalidad", "centro", "periferia", "suburbio"
            ],

            "analisis_precio": [
                # Términos de precio
                "precio", "valor", "costo", "tasación", "valoración",
                "dólares", "usd", "euros", "cuanto", "cuesta",
                # Análisis financiero
                "metro cuadrado", "m2", "precio por metro", "inversión",
                "economico", "barato", "costoso", "caro", "ganga",
                # Rangos
                "menor a", "mayor a", "entre", "máximo", "mínimo",
                "desde", "hasta", "rango de precios"
            ],

            "detalles_propiedad": [
                # Espacios principales
                "habitaciones", "dormitorios", "baños", "ambientes",
                "cuartos", "recamaras", "alcobas",
                # Medidas
                "metros", "m2", "superficie", "tamaño", "dimensiones",
                "grande", "pequeño", "amplio",
                # Características 
                "nuevo", "usado", "moderno", "antiguo", "remodelado"
            ],

            "comparativo_mercado": [
                # Comparaciones
                "comparar", "similar", "parecido", "equivalente",
                "versus", "mejor", "peor", "diferencia",
                # Mercado
                "oferta", "demanda", "disponible", "stock",
                "tendencia", "promedio", "media"
            ]
        }

        self.consultas = {
            "identificacion_localizacion": {
                "info_basica": """
                    SELECT 
                        cp.id,
                        cp.location,
                        cp.property_type,
                        cp.url,
                        cp.square_meters,
                        cp.description,
                        cp.image,
                        cp.num_bedrooms as promedio_ambientes,
                        cp.num_rooms as promedio_dormitorios,
                        ROUND(cp.price, 2) as price,
                        ROUND(cp.price/cp.square_meters, 2) as price_per_m2,
                        cc.name as country_name,
                        cd.name as province_name,
                        cct.name as city_name,
                        cp.project_category,
                        cp.project_type,
                        cp.residence_type
                    FROM chat_property cp
                    LEFT JOIN chat_country cc ON cp.country_id = cc.id
                    LEFT JOIN chat_province cd ON cp.province_id = cd.id
                    LEFT JOIN chat_city cct ON cp.city_id = cct.id
                    WHERE {condition}
                    ORDER BY cp.price/cp.square_meters ASC
                """,

                "analisis_ubicacion": """
                    SELECT 
                        cc.name as country_name,
                        cd.name as province_name,
                        cct.name as city_name,
                        COUNT(*) as total_properties,
                        ROUND(AVG(cp.price), 2) as avg_price,
                        ROUND(AVG(cp.price/cp.square_meters), 2) as avg_price_per_m2,
                        ROUND(MIN(cp.price), 2) as min_price,
                        ROUND(MAX(cp.price), 2) as max_price
                    FROM chat_property cp
                    LEFT JOIN chat_country cc ON cp.country_id = cc.id
                    LEFT JOIN chat_province cd ON cp.province_id = cd.id
                    LEFT JOIN chat_city cct ON cp.city_id = cct.id
                    GROUP BY cc.name, cd.name, cct.name
                    HAVING total_properties > 0
                    ORDER BY avg_price_per_m2 ASC
                """
            },

            "analisis_precio": {
                "comparativa_precios": """
                    SELECT 
                        cp.property_type,
                        cc.name as country_name,
                        cd.name as province_name,
                        COUNT(*) as total_properties,
                        ROUND(AVG(cp.price), 2) as avg_price,
                        ROUND(MIN(cp.price), 2) as min_price,
                        ROUND(MAX(cp.price), 2) as max_price,
                        ROUND(AVG(cp.price/cp.square_meters), 2) as avg_price_per_m2,
                        ROUND(AVG(cp.square_meters), 2) as avg_size,
                        ROUND(AVG(cp.num_bedrooms), 1) as promedio_ambientes
                    FROM chat_property cp
                    LEFT JOIN chat_country cc ON cp.country_id = cc.id
                    LEFT JOIN chat_province cd ON cp.province_id = cd.id
                    GROUP BY cp.property_type, cc.name, cd.name
                    HAVING total_properties > 0
                    ORDER BY avg_price_per_m2 ASC
                """
            },

            "detalles_propiedad": {
                "analisis_detallado": """
                    SELECT 
                        cp.property_type,
                        cp.project_category,
                        cp.project_type,
                        cp.residence_type,
                        COUNT(*) as total,
                        ROUND(AVG(cp.square_meters), 2) as avg_size,
                        ROUND(AVG(cp.num_bedrooms), 1) as promedio_ambientes,
                        ROUND(AVG(cp.num_rooms), 1) as promedio_dormitorios,
                        ROUND(AVG(cp.price), 2) as avg_price,
                        ROUND(AVG(cp.price/cp.square_meters), 2) as avg_price_per_m2
                    FROM chat_property cp
                    GROUP BY 
                        cp.property_type,
                        cp.project_category,
                        cp.project_type,
                        cp.residence_type
                    HAVING total > 0
                    ORDER BY avg_price_per_m2 ASC
                """
            },

            "comparativo_mercado": {
                "analisis_mercado": """
                    SELECT 
                        cc.name as country_name,
                        cp.property_type,
                        COUNT(*) as total_properties,
                        ROUND(AVG(cp.price), 2) as avg_price,
                        ROUND(MIN(cp.price), 2) as min_price,
                        ROUND(MAX(cp.price), 2) as max_price,
                        ROUND(AVG(cp.square_meters), 2) as avg_size,
                        ROUND(AVG(cp.price/cp.square_meters), 2) as avg_price_per_m2,
                        ROUND(AVG(cp.num_bedrooms), 1) as promedio_ambientes
                    FROM chat_property cp
                    LEFT JOIN chat_country cc ON cp.country_id = cc.id
                    GROUP BY cc.name, cp.property_type
                    HAVING total_properties > 0
                    ORDER BY avg_price_per_m2 ASC
                """
            }
        }
    
    def generar_respuesta(self, pregunta: str) -> str:
        try:
            categoria = self._identificar_categoria(pregunta)
            datos = self._obtener_datos(pregunta, categoria)
            return self._generar_respuesta_gpt(pregunta, datos, categoria)
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            return f"Error al procesar la pregunta: {str(e)}"


    def _identificar_categoria(self, pregunta: str) -> str:
        pregunta = pregunta.lower()
        mejor_match = "general"
        max_coincidencias = 0
        
        for categoria, palabras_clave in self.categorias.items():
            coincidencias = sum(1 for palabra in palabras_clave if palabra in pregunta)
            if coincidencias > max_coincidencias:
                max_coincidencias = coincidencias
                mejor_match = categoria
        
        return mejor_match


    def _obtener_datos(self, pregunta: str, categoria: str, condition: str = "1=1") -> Dict[str, Any]:
        resultados = {}
        if categoria in self.consultas:
            for nombre_consulta, consulta in self.consultas[categoria].items():
                try:
                    if "{condition}" in consulta:
                        consulta = consulta.replace("{condition}", condition)
                    df = pd.read_sql_query(consulta, self.conn)
                    
                    # Procesar las URLs de las imágenes y los detalles
                    for index, row in df.iterrows():
                        # Procesar imagen
                        if 'image' in row:
                            image_name = row['image'].replace('property_images/', '')
                            df.at[index, 'image'] = f"/media/{image_name}"
                        
                        # Procesar URL de detalles
                        if 'url' in row:
                            url = row['url']
                            # Limpiar URL
                            url = self._clean_url(url)
                            df.at[index, 'url'] = url
                    
                    resultados[nombre_consulta] = df.to_dict('records')
                except Exception as e:
                    self.logger.error(f"Error en consulta {nombre_consulta}: {str(e)}")
                    resultados[nombre_consulta] = {"error": str(e)}
        return resultados

    def _clean_url(self, url: str) -> str:
        """Limpia una URL de atributos HTML y caracteres no deseados"""
        if not url:
            return '#'
        
        # Lista de atributos HTML a remover
        attrs_to_remove = ['class=', 'target=', 'rel=']
        cleaned_url = url
        
        for attr in attrs_to_remove:
            if attr in cleaned_url:
                cleaned_url = cleaned_url.split(attr)[0]
        
        return cleaned_url.replace('"', '').strip()

    # def _generar_respuesta_gpt(self, pregunta: str, datos: Dict[str, Any], categoria: str) -> str:
    #     prompt = f"""
    #     Pregunta: {pregunta}
    #     Categoría: {categoria}
    #     Datos disponibles: {json.dumps(datos, ensure_ascii=False)}
        
    #     Proporciona un análisis detallado basado en los datos disponibles.
    #     """
        
    #     try:
    #         response = openai.ChatCompletion.create(
    #             model="gpt-4",
    #             messages=[
    #                 {"role": "system", "content": "Eres un experto en análisis inmobiliario."},
    #                 {"role": "user", "content": prompt}
    #             ]
    #         )
    #         return response.choices[0].message["content"]
    #     except Exception as e:
    #         return f"Error al generar respuesta con GPT: {str(e)}"


    def cerrar(self):
        self.conn.close()
        self.logger.info("Conexión cerrada correctamente")