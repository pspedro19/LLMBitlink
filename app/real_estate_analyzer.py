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


    def _init_query_mappings(self):
        self.categorias = {
            "identificacion_localizacion": ["ubicacion", "localización", "ciudad", "zona"],
            "analisis_precio": ["precio", "valor", "costo", "mercado"],
            "detalles_propiedad": ["habitaciones", "características", "espacios"],
            "clasificacion_categoria": ["categoría", "tipo", "proyecto"],
            "cronologia": ["tiempo", "fecha", "histórico", "tendencias"]
        }

        self.consultas = {
            "identificacion_localizacion": {
                "info_basica": """
                    SELECT id, location, property_type, ROUND(price, 2) as price
                    FROM chat_property WHERE {condition}
                """
            },
            "analisis_precio": {
                "comparativa_precios": """
                    SELECT location, property_type, 
                    ROUND(AVG(price), 2) as precio_promedio
                    FROM chat_property
                    GROUP BY location, property_type
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


    def _obtener_datos(self, pregunta: str, categoria: str) -> Dict[str, Any]:
        resultados = {}
        if categoria in self.consultas:
            for nombre_consulta, consulta in self.consultas[categoria].items():
                try:
                    if "{condition}" in consulta:
                        consulta = consulta.replace("{condition}", "1=1")
                    df = pd.read_sql_query(consulta, self.conn)
                    resultados[nombre_consulta] = df.to_dict('records')
                except Exception as e:
                    resultados[nombre_consulta] = {"error": str(e)}
        return resultados


    def _generar_respuesta_gpt(self, pregunta: str, datos: Dict[str, Any], categoria: str) -> str:
        prompt = f"""
        Pregunta: {pregunta}
        Categoría: {categoria}
        Datos disponibles: {json.dumps(datos, ensure_ascii=False)}
        
        Proporciona un análisis detallado basado en los datos disponibles.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis inmobiliario."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"Error al generar respuesta con GPT: {str(e)}"


    def cerrar(self):
        self.conn.close()
        self.logger.info("Conexión cerrada correctamente")