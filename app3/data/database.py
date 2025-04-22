from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
import os
import traceback
from utils.logger import get_logger
from utils.config import Config

logger = get_logger(__name__)

class DatabaseManager:
    """
    Gestiona las operaciones de base de datos y acceso a fuentes de datos estructurados.
    """
    def __init__(self):
        """Inicializa el gestor de base de datos"""
        self.config = Config()
        self.dataframes = {}
        
        # Definir mapeo de columnas para cada tabla
        self.column_mappings = {
            "activities": {
                "id_activity": "id",
                "name": "name",
                "type": "type",
                "location": "location",
                "duration_hours": "duration_hours",
                "cost": "cost",
                "rating": "rating",
                "description": "description",
                "recommended_for": "recommended_for",
                "contact_info": "contact_info",
                "website": "website",
                "social_media": "social_media",
                "accessibility": "accessibility",
                "parking": "parking",
                "payment_options": "payment_options",
                "languages_spoken": "languages",
                "season": "season",
                "accessible": "accessible",
                "languages": "languages",
                "contact_number": "contact_number"
            },
            "tourist_spots": {
                "id_spot": "id",
                "name": "name",
                "type": "type",
                "location": "location",
                "opening_hours": "opening_hours",
                "entry_fee": "entry_fee",
                "rating": "rating",
                "description": "description",
                "ideal_for": "ideal_for",
                "contact_info": "contact_info",
                "website": "website",
                "social_media": "social_media",
                "accessibility": "accessibility",
                "parking": "parking",
                "payment_options": "payment_options",
                "languages_spoken": "languages",
                "season": "season",
                "accessible": "accessible",
                "languages": "languages",
                "contact_number": "contact_number"
            },
            "restaurants": {
                "id_restaurant": "id",
                "name": "name",
                "cuisine_type": "cuisine_type",
                "average_person_expense": "average_person_expense",
                "price_range": "price_range",
                "location": "location",
                "opening_hours": "opening_hours",
                "rating": "rating",
                "recommended_for": "recommended_for",
                "description": "description",
                "contact_info": "contact_info",
                "website": "website",
                "social_media": "social_media",
                "accessibility": "accessibility",
                "parking": "parking",
                "payment_options": "payment_options",
                "languages_spoken": "languages",
                "season": "season",
                "accessible": "accessible",
                "languages": "languages"
            },
            "nightclubs": {
                "id_nightclub": "id",
                "name": "name",
                "music_type": "music_type",
                "average_person_expense": "average_person_expense",
                "price_range": "price_range",
                "location": "location",
                "opening_hours": "opening_hours",
                "rating": "rating",
                "recommended_for": "recommended_for",
                "description": "description",
                "contact_info": "contact_info",
                "website": "website",
                "social_media": "social_media",
                "accessibility": "accessibility",
                "parking": "parking",
                "payment_options": "payment_options",
                "languages_spoken": "languages",
                "season": "season",
                "accessible": "accessible",
                "languages": "languages",
                "dress_code": "dress_code"
            },
            "tourism_packages": {
                "id_package": "id",
                "name": "name",
                "description": "description",
                "price": "price",
                "duration_days": "duration_days",
                "includes": "includes",
                "categories": "categories",
                "contact_info": "contact_info",
                "website": "website",
                "social_media": "social_media",
                "accessibility": "accessibility",
                "parking": "parking",
                "payment_options": "payment_options",
                "languages_spoken": "languages",
                "season": "season",
                "accessible": "accessible",
                "languages": "languages",
                "contact_number": "contact_number"
            }
        }
        
        # Cargar los datos
        self._load_data_sources()
        
    def _load_data_sources(self) -> None:
        """Carga todas las fuentes de datos en memoria"""
        try:
            data_sources = {
                "activities": {
                    "path": self.config.DATABASE_PATHS["activities"],
                    "required_columns": [
                        "id", "name", "type", "location", "duration_hours", 
                        "cost", "rating", "description", "recommended_for"
                    ]
                },
                "tourist_spots": {
                    "path": self.config.DATABASE_PATHS["tourist_spots"],
                    "required_columns": [
                        "id", "name", "type", "location", "entry_fee", 
                        "rating", "description", "ideal_for"
                    ]
                },
                "restaurants": {
                    "path": self.config.DATABASE_PATHS["restaurants"],
                    "required_columns": [
                        "id", "name", "cuisine_type", "location", "average_person_expense", 
                        "price_range", "rating", "recommended_for"
                    ]
                },
                "nightclubs": {
                    "path": self.config.DATABASE_PATHS["nightclubs"],
                    "required_columns": [
                        "id", "name", "music_type", "location", "average_person_expense", 
                        "price_range", "rating", "recommended_for"
                    ]
                },
                "tourism_packages": {
                    "path": self.config.DATABASE_PATHS["tourism_packages"],
                    "required_columns": [
                        "id", "name", "description", "price", "duration_days", 
                        "includes", "categories"
                    ]
                }
            }
            
            # Cargar cada fuente de datos
            for name, info in data_sources.items():
                self._load_dataframe(name, info["path"], info["required_columns"])
                
            logger.info(f"Cargadas {len(self.dataframes)} fuentes de datos correctamente")
            
        except Exception as e:
            logger.error(f"Error cargando fuentes de datos: {e}")
            logger.error(traceback.format_exc())
            # Continuar con dataframes vacíos
            for name in ["activities", "tourist_spots", "restaurants", "nightclubs", "tourism_packages"]:
                if name not in self.dataframes:
                    self.dataframes[name] = pd.DataFrame()
    
    def _load_dataframe(self, name: str, path: Path, required_columns: List[str]) -> None:
        """
        Carga un único dataframe desde su fuente y maneja nombres de columnas diferentes
        
        Args:
            name (str): Nombre del dataframe
            path (Path): Ruta al archivo Excel/CSV
            required_columns (List[str]): Lista de columnas requeridas
        """
        try:
            # Verificar si el archivo existe
            if not os.path.exists(path):
                logger.error(f"Archivo no encontrado: {path}")
                # Crear un DataFrame vacío con las columnas requeridas
                empty_df = pd.DataFrame(columns=required_columns)
                self.dataframes[name] = empty_df
                return
                    
            # Cargar dataframe
            logger.info(f"Intentando cargar {name} desde {path}")
            df = pd.read_excel(path, engine='openpyxl')
            
            # Normalizar nombres de columnas
            df.columns = df.columns.str.lower().str.strip()
            
            # Detectar columnas originales para depuración
            logger.info(f"Columnas originales en {name}: {df.columns.tolist()}")
            
            # Verificar columnas de ID específicas antes del mapeo
            id_columns = [col for col in df.columns if col.startswith('id_')]
            if id_columns and 'id' not in df.columns:
                logger.info(f"Detectadas columnas ID en {name}: {id_columns}")
            
            # Aplicar mapeo de columnas
            if name in self.column_mappings:
                # Renombrar columnas según el mapeo
                column_map = {k: v for k, v in self.column_mappings[name].items() if k in df.columns}
                if column_map:
                    df = df.rename(columns=column_map)
                    logger.info(f"Columnas renombradas en {name}: {column_map}")
            
            # Verificar columnas requeridas después del mapeo
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Columnas faltantes en {name}: {missing_columns}")
                for col in missing_columns:
                    # Si la columna es 'id', generar ids automáticos o buscar alternativas
                    if col == 'id':
                        # Verificar si existe id_[name]
                        id_column = f"id_{name.rstrip('s')}"
                        if id_column in df.columns:
                            df[col] = df[id_column]
                            logger.info(f"Usando {id_column} como {col} en {name}")
                        else:
                            # Generar ids automáticos
                            df[col] = [f"{name}_{i}" for i in range(len(df))]
                            logger.info(f"Generados IDs automáticos para {name}")
                    else:
                        df[col] = None
            
            # Convertir columnas numéricas
            numeric_columns = {
                'cost': 'float64',
                'entry_fee': 'float64',
                'price': 'float64',
                'rating': 'float64',
                'duration_hours': 'float64',
                'duration_days': 'int64',
                'average_person_expense': 'float64',
                'price_range': 'float64'
            }

            for col, dtype in numeric_columns.items():
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"No se pudo convertir {col} a {dtype}: {e}")
            
            # Asegurar que todas las columnas de texto sean strings y manejar NaN
            text_cols = df.select_dtypes(include=['object']).columns
            for col in text_cols:
                df[col] = df[col].astype(str).replace('nan', '')
                    
            # Añadir valores predeterminados para campos importantes
            if 'rating' in df.columns and df['rating'].isna().any():
                df['rating'] = df['rating'].fillna(3.5)  # Rating predeterminado neutral
                
            # Almacenar el dataframe
            self.dataframes[name] = df
            logger.info(f"Cargado {name} con {len(df)} filas")
            
        except Exception as e:
            logger.error(f"Error cargando {name} desde {path}: {e}")
            logger.error(traceback.format_exc())
            # Crear un DataFrame vacío como fallback
            empty_df = pd.DataFrame(columns=required_columns)
            self.dataframes[name] = empty_df
    
    def execute_query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ejecuta una consulta en la base de datos de manera robusta
        
        Args:
            query_params (Dict[str, Any]): Parámetros de consulta estructurados
            
        Returns:
            List[Dict[str, Any]]: Resultados de la consulta
        """
        try:
            results = []
            
            # Determinar qué tablas consultar
            tables = query_params.get('tables', list(self.dataframes.keys()))
            
            # Si no hay tablas especificadas, usar todas
            if not tables:
                tables = list(self.dataframes.keys())
                
            # Log para depuración
            logger.info(f"Ejecutando consulta en tablas: {tables}")
            logger.info(f"Filtros aplicados: {query_params.get('filters', {})}")
            
            for table in tables:
                if table not in self.dataframes:
                    logger.warning(f"Tabla {table} no encontrada")
                    continue
                    
                # Obtener dataframe
                df = self.dataframes[table].copy()
                
                # Si el dataframe está vacío, continuar con la siguiente tabla
                if df.empty:
                    logger.warning(f"Tabla {table} está vacía")
                    continue
                    
                # Aplicar filtros - proteger contra posibles errores en el filtrado
                try:
                    filtered_df = self._apply_filters(df, query_params.get('filters', {}), table)
                    
                    # Verificar si hay resultados después de filtrar
                    if filtered_df is None:
                        # Este es un caso crítico - corrección fundamental
                        logger.error(f"El método _apply_filters devolvió None para la tabla {table}")
                        filtered_df = df  # Usar el dataframe original como fallback
                    
                    if filtered_df.empty:
                        logger.info(f"No hay resultados en la tabla {table} después de filtrar")
                        continue
                        
                    # Ordenar resultados
                    sorted_df = self._apply_sorting(filtered_df, query_params.get('sort', {}))
                    
                    # Convertir a diccionarios y añadir a resultados
                    table_results = sorted_df.to_dict('records')
                    
                    # Añadir origen de datos
                    for record in table_results:
                        record['data_source'] = table
                    
                    results.extend(table_results)
                except Exception as filter_error:
                    logger.error(f"Error aplicando filtros a la tabla {table}: {filter_error}")
                    logger.error(traceback.format_exc())
                    # Intentar continuar con la siguiente tabla en lugar de fallar completamente
                    continue
                        
            # Log para depuración
            if results:
                logger.info(f"Consulta obtuvo {len(results)} resultados")
            else:
                logger.warning("La consulta no obtuvo resultados")
                
            # Ordenar resultados combinados si existe ordenamiento global
            if results and query_params.get('sort'):
                sort_key = query_params['sort'].get('key', 'rating')
                ascending = query_params['sort'].get('ascending', False)
                
                def safe_sort_key(x):
                    value = x.get(sort_key)
                    # Manejar cualquier tipo de valor o nulos
                    if value is None:
                        return 0 if not ascending else float('inf')
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        # Si no se puede convertir a número, usar representación string
                        return str(value)
                
                try:
                    results = sorted(results, key=safe_sort_key, reverse=not ascending)
                except Exception as e:
                    logger.warning(f"Error ordenando resultados: {e}")
                        
            # Aplicar límite global
            limit = query_params.get('limit', 10)
            limited_results = results[:limit]
            
            # Añadir información de debug
            logger.info(f"Devolviendo {len(limited_results)} resultados de {len(results)} después de aplicar límite {limit}")
                
            return limited_results
                
        except Exception as e:
            logger.error(f"Error ejecutando consulta: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any], table_name: str) -> pd.DataFrame:
        """
        Aplica filtros a un dataframe de manera más robusta
        
        Args:
            df (pd.DataFrame): Dataframe a filtrar
            filters (Dict[str, Any]): Filtros a aplicar
            table_name (str): Nombre de la tabla
            
        Returns:
            pd.DataFrame: Dataframe filtrado
        """
        if not filters:
            return df
                
        # Mappings para nombres de columnas por tabla
        column_mappings = {
            'budget': {
                'activities': 'cost',
                'tourist_spots': 'entry_fee',
                'restaurants': 'average_person_expense',
                'nightclubs': 'average_person_expense',
                'tourism_packages': 'price'
            }
        }
        
        # Copia para no modificar el original
        filtered_df = df.copy()
        
        # Registrar estado inicial para debug
        initial_rows = len(filtered_df)
        
        # Filtrar por ubicación de manera más flexible
        if 'locations' in filters and filters['locations']:
            locations = [str(loc).lower() for loc in filters['locations']]
            logger.info(f"Filtrando por ubicaciones: {locations}")
            
            # Si 'curacao' o 'curaçao' está entre las ubicaciones, tratar como ubicación general
            has_curacao = any(loc in ['curacao', 'curaçao'] for loc in locations)
            
            # Verificar si tenemos columna de ubicación
            if 'location' in filtered_df.columns:
                # Si buscamos Curaçao en general, no filtramos estrictamente por ubicación
                if has_curacao and len(locations) <= 2:
                    # No aplicamos filtro estricto, asumimos que todo está en Curaçao
                    logger.info(f"Búsqueda general de Curaçao detectada, manteniendo todas las ubicaciones")
                else:
                    # Para ubicaciones específicas, filtrar con coincidencia parcial
                    location_mask = filtered_df['location'].fillna('').astype(str).str.lower().apply(
                        lambda x: any(loc in x or x in loc for loc in locations) or 
                                any(x.startswith(loc) or x.endswith(loc) for loc in locations)
                    )
                    
                    # Si no encontramos ninguna coincidencia, ser más flexible
                    if not location_mask.any() and has_curacao:
                        logger.info(f"No se encontraron coincidencias exactas, manteniendo todas las ubicaciones")
                    else:
                        filtered_df = filtered_df[location_mask]
                        logger.info(f"Después de filtrar por ubicación: {len(filtered_df)} filas (de {initial_rows})")
            else:
                logger.warning(f"Columna 'location' no encontrada en {table_name}")
        
        # Filtrar por presupuesto
        if 'budget' in filters and filters['budget'] is not None:
            try:
                budget = float(filters['budget'])
                # Determinar columna de presupuesto
                budget_column = None
                if table_name in column_mappings['budget']:
                    budget_column = column_mappings['budget'][table_name]
                    
                # Verificar si la columna existe
                if budget_column and budget_column in filtered_df.columns:
                    # Aplicar filtro de presupuesto con más flexibilidad (50% extra)
                    budget_limit = budget * 1.5
                    filtered_df = filtered_df[
                        (filtered_df[budget_column].isna()) | 
                        (filtered_df[budget_column] <= budget_limit)
                    ]
                    logger.info(f"Después de filtrar por presupuesto ({budget_column} <= {budget_limit}): {len(filtered_df)} filas")
                else:
                    logger.warning(f"Columna de presupuesto {budget_column} no encontrada en {table_name}")
                    
            except (ValueError, TypeError):
                logger.warning(f"Valor de presupuesto no válido: {filters['budget']}")
            
        # Filtrar por intereses de manera más flexible
        if 'interests' in filters and filters['interests']:
            # Mapear intereses a categorías y tipos
            interest_types = self._map_interests_to_types(filters['interests'])
            logger.info(f"Filtrando por tipos de interés: {interest_types[:10]}...") # Log truncado para evitar spam
            
            # Diferentes campos según la tabla
            fields_to_check = ['type', 'description', 'recommended_for', 'ideal_for', 'categories', 'name']
            if table_name == 'restaurants':
                fields_to_check.extend(['cuisine_type'])
            elif table_name == 'nightclubs':
                fields_to_check.extend(['music_type'])
                
            # Crear máscara combinada para intereses
            interest_mask = pd.Series(False, index=filtered_df.index)
            
            # Verificar cada campo relevante
            fields_checked = 0
            for field in fields_to_check:
                if field in filtered_df.columns:
                    fields_checked += 1
                    field_mask = filtered_df[field].fillna('').astype(str).str.lower().apply(
                        lambda x: any(interest_type in x for interest_type in interest_types) or
                                any(x in interest_type for interest_type in interest_types)
                    )
                    interest_mask = interest_mask | field_mask
            
            # Si no encontramos campos para buscar intereses, no filtrar
            if fields_checked == 0:
                logger.warning(f"No se encontraron campos para filtrar intereses en {table_name}")
            # Aplicar filtro solo si tenemos alguna coincidencia
            elif interest_mask.any():
                filtered_df = filtered_df[interest_mask]
                logger.info(f"Después de filtrar por intereses: {len(filtered_df)} filas")
            else:
                logger.warning(f"No se encontraron coincidencias de interés en {table_name}")
                # Si no hay coincidencias pero es un tipo de tabla relevante para estos intereses,
                # evitamos filtrar completamente para mantener algunos resultados
                interest_to_table_relevance = {
                    'cultural': ['tourist_spots', 'activities'],
                    'naturaleza': ['tourist_spots', 'activities'],
                    'gastronomía': ['restaurants'],
                    'actividades_acuáticas': ['activities'],
                    'aventura': ['activities'],
                    'vida_nocturna': ['nightclubs'],
                    'romántico': ['restaurants', 'tourist_spots']
                }
                
                for interest in filters['interests']:
                    if interest in interest_to_table_relevance:
                        relevant_tables = interest_to_table_relevance[interest]
                        if table_name in relevant_tables:
                            logger.info(f"Manteniendo resultados para tabla relevante {table_name} para interés {interest}")
                            # No aplicar filtro, mantener todos los resultados
                            break
                        
        # Filtrar por calificación mínima
        if 'min_rating' in filters and filters['min_rating'] is not None:
            try:
                min_rating = float(filters['min_rating'])
                if 'rating' in filtered_df.columns:
                    # Ser más flexible con el rating mínimo (0.5 menos)
                    flexible_min_rating = max(0, min_rating - 0.5)
                    filtered_df = filtered_df[
                        (filtered_df['rating'].isna()) | 
                        (filtered_df['rating'] >= flexible_min_rating)
                    ]
                    logger.info(f"Después de filtrar por rating (>= {flexible_min_rating}): {len(filtered_df)} filas")
            except (ValueError, TypeError):
                logger.warning(f"Valor de calificación mínima no válido: {filters['min_rating']}")
            
        # Filtro para viajes familiares
        if 'family_friendly' in filters and filters.get('family_friendly') is True:
            # Verificar en campos relevantes
            family_keywords = ['familia', 'niño', 'family', 'kid', 'child']
            
            family_mask = pd.Series(False, index=filtered_df.index)
            for field in ['recommended_for', 'ideal_for', 'description']:
                if field in filtered_df.columns:
                    field_mask = filtered_df[field].fillna('').astype(str).str.lower().apply(
                        lambda x: any(keyword in x for keyword in family_keywords)
                    )
                    family_mask = family_mask | field_mask
            
            # Aplicar filtro si hay coincidencias
            if family_mask.any():
                filtered_df = filtered_df[family_mask]
                logger.info(f"Después de filtrar por family_friendly: {len(filtered_df)} filas")
            else:
                # Si no hay coincidencias específicas de familia, no filtramos
                logger.warning(f"No se encontraron actividades explícitamente familiares, manteniendo todas las opciones")
                
        # Como último recurso, si después de todos los filtros no queda nada,
        # intentamos ser menos estrictos con los filtros de ubicación
        if filtered_df.empty and 'locations' in filters:
            logger.warning(f"No quedaron resultados después de filtrar. Intentando búsqueda más flexible.")
            return self._apply_filters_relaxed(df, filters, table_name)
                
        # Este return es crítico - estaba faltando en la versión original
        return filtered_df

    def _apply_filters_relaxed(self, df: pd.DataFrame, filters: Dict[str, Any], table_name: str) -> pd.DataFrame:
        """
        Versión relajada del filtrado para casos donde los filtros estrictos no dan resultados
        
        Args:
            df (pd.DataFrame): DataFrame original
            filters (Dict[str, Any]): Filtros a aplicar de forma relajada
            table_name (str): Nombre de la tabla
            
        Returns:
            pd.DataFrame: DataFrame filtrado con criterios relajados
        """
        # Copia para no modificar el original
        filtered_df = df.copy()
        
        # Filtrar solo por rating mínimo si existe
        if 'min_rating' in filters and filters['min_rating'] is not None:
            try:
                min_rating = float(filters['min_rating'])
                if 'rating' in filtered_df.columns:
                    # Usar un rating mínimo muy bajo
                    filtered_df = filtered_df[
                        (filtered_df['rating'].isna()) | 
                        (filtered_df['rating'] >= min_rating - 1.0)  # 1.0 menos que el original
                    ]
            except (ValueError, TypeError):
                pass
        
        # Aplicar alguna coincidencia parcial de intereses si es posible
        if 'interests' in filters and filters['interests'] and len(filtered_df) > 10:
            # Obtener solo palabras clave principales
            main_keywords = []
            for interest in filters['interests']:
                if interest == 'cultural':
                    main_keywords.extend(['museum', 'history', 'culture'])
                elif interest == 'naturaleza':
                    main_keywords.extend(['nature', 'beach', 'park'])
                elif interest == 'gastronomía':
                    main_keywords.extend(['food', 'restaurant'])
                elif interest == 'actividades_acuáticas':
                    main_keywords.extend(['water', 'beach', 'dive'])
                elif interest == 'aventura':
                    main_keywords.extend(['adventure', 'outdoor'])
            
            if main_keywords:
                # Buscar coincidencias en nombre o descripción
                interest_mask = pd.Series(False, index=filtered_df.index)
                for field in ['name', 'description', 'type']:
                    if field in filtered_df.columns:
                        field_mask = filtered_df[field].fillna('').astype(str).str.lower().apply(
                            lambda x: any(keyword in x for keyword in main_keywords)
                        )
                        interest_mask = interest_mask | field_mask
                
                # Aplicar filtro solo si no reduce demasiado los resultados
                if interest_mask.sum() >= 5:
                    filtered_df = filtered_df[interest_mask]
        
        # Si aún no tenemos suficientes resultados, devolver todo
        if len(filtered_df) < 5:
            logger.warning(f"Búsqueda relajada produjo pocos resultados. Devolviendo todo para {table_name}.")
            return df
        
        return filtered_df
        
    def _apply_sorting(self, df: pd.DataFrame, sort_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Aplica ordenamiento a un dataframe
        
        Args:
            df (pd.DataFrame): Dataframe a ordenar
            sort_params (Dict[str, Any]): Parámetros de ordenamiento
            
        Returns:
            pd.DataFrame: Dataframe ordenado
        """
        if not sort_params:
            # Ordenamiento predeterminado por calificación
            if 'rating' in df.columns:
                return df.sort_values('rating', ascending=False, na_position='last')
            return df
            
        sort_key = sort_params.get('key', 'rating')
        ascending = sort_params.get('ascending', False)
        
        if sort_key in df.columns:
            return df.sort_values(sort_key, ascending=ascending, na_position='last')
        
        return df
    
    def _map_interests_to_types(self, interests: List[str]) -> List[str]:
        """
        Mapea intereses a tipos específicos para filtrado de manera más exhaustiva
        
        Args:
            interests (List[str]): Lista de intereses
            
        Returns:
            List[str]: Lista de tipos mapeados
        """
        interest_type_mapping = {
            'cultural': [
                'museo', 'histórico', 'cultural', 'history', 'arts', 'architecture', 'monumento',
                'museum', 'heritage', 'monument', 'art', 'culture', 'historical', 'colonial', 
                'patrimonio', 'architectural', 'tour', 'guía', 'guide', 'site', 'sitio',
                'heritage', 'patrimonio', 'world', 'unesco', 'tradicional', 'traditional'
            ],
            'naturaleza': [
                'playa', 'parque', 'nature', 'beach', 'park', 'natural', 'flora', 'fauna',
                'reserva', 'island', 'isla', 'jardín', 'garden', 'montaña', 'mountain',
                'landscape', 'paisaje', 'panoramic', 'panorámico', 'mirador', 'viewpoint',
                'eco', 'environment', 'ambiente', 'vista', 'view', 'natural', 'coast', 'costa'
            ],
            'gastronomía': [
                'restaurant', 'food', 'dining', 'culinary', 'comida', 'restaurante', 'gastronomía',
                'cuisine', 'gastronómico', 'sabor', 'taste', 'gourmet', 'comer', 'eat', 'dinner',
                'cena', 'chef', 'seafood', 'mariscos', 'local food', 'comida local', 'authentic',
                'tradicional', 'plato', 'dish', 'menú', 'meal', 'cena', 'almuerzo', 'lunch',
                'breakfast', 'desayuno', 'café', 'coffee', 'bar', 'bistro'
            ],
            'actividades_acuáticas': [
                'diving', 'snorkel', 'swim', 'beach', 'buceo', 'nadar', 'submarine', 'submarino',
                'acuático', 'aquatic', 'water', 'agua', 'sailing', 'navegación', 'surf', 'kayak',
                'paddleboard', 'fishing', 'pesca', 'boat', 'barco', 'lagoon', 'laguna', 'reef',
                'arrecife', 'coral', 'underwater', 'subacuático', 'mar', 'sea', 'ocean', 'océano',
                'playa', 'shore', 'costa', 'coast', 'bay', 'bahía', 'dock', 'muelle', 'port', 'puerto'
            ],
            'aventura': [
                'adventure', 'hiking', 'trekking', 'aventura', 'senderismo', 'climbing', 'escalada',
                'zip line', 'tirolesa', 'outdoor', 'aire libre', 'exploration', 'exploración',
                'safari', 'jeep', 'atv', 'quad', 'bike', 'bicicleta', 'cycling', 'ciclismo',
                'horse', 'caballo', 'horseback', 'tours', 'expedición', 'expedition', 'excursión',
                'tour', 'guide', 'guía', 'adrenaline', 'adrenalina', 'extreme', 'extremo'
            ],
            'vida_nocturna': [
                'club', 'bar', 'night', 'music', 'discoteca', 'nightlife', 'party', 'fiesta',
                'dance', 'baile', 'entertainment', 'entretenimiento', 'concert', 'concierto',
                'show', 'espectáculo', 'casino', 'cocktail', 'cóctel', 'drink', 'bebida',
                'pub', 'lounge', 'salsa', 'jazz', 'live music', 'música en vivo', 'disco',
                'dj', 'bailar', 'dancing', 'night', 'noche', 'late', 'tarde', 'evento', 'event'
            ],
            'romántico': [
                'romantic', 'romance', 'sunset', 'intimate', 'couple', 'romántico', 'pareja',
                'honeymoon', 'luna de miel', 'anniversary', 'aniversario', 'special', 'especial',
                'private', 'privado', 'exclusive', 'exclusivo', 'luxury', 'lujo', 'spa',
                'relaxation', 'relajación', 'massage', 'masaje', 'wellness', 'bienestar',
                'vista', 'view', 'cena', 'dinner', 'atardecer', 'sunset', 'atmosphere', 'ambiente'
            ]
        }
        
        # Añadir términos generales que aplican a todos los tipos de turismo
        general_terms = [
            'tourism', 'turismo', 'visitar', 'visit', 'place', 'lugar', 'destination',
            'destino', 'attraction', 'atracción', 'popular', 'famous', 'famoso', 
            'recommended', 'recomendado', 'essential', 'esencial', 'must see', 'imperdible'
        ]
        
        types = []
        # Añadir todos los términos generales
        types.extend(general_terms)
        
        # Añadir términos específicos por interés
        for interest in interests:
            if interest in interest_type_mapping:
                types.extend(interest_type_mapping[interest])
            else:
                # Si el interés no está mapeado, añadirlo directamente
                types.append(interest.lower())
                    
        # Asegurar que no haya duplicados
        return list(set(types)) if types else [i.lower() for i in interests]