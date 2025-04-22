# mcp_curacao/servers/excel_server.py
from mcp.server.fastmcp import FastMCP
import pandas as pd
import os
import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("excel-server")

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXCEL_DIR, EXCEL_SERVER_ID

# Inicializar servidor MCP
mcp = FastMCP(EXCEL_SERVER_ID)

# Variables globales para datos
datasets = {}

@mcp.tool()
def load_excel_data() -> dict:
    """
    Carga todos los archivos Excel en memoria.
    
    Returns:
        Información sobre datos cargados
    """
    try:
        # Cargar cada archivo Excel
        excel_files = {
            "activities": "activities.xlsx",
            "tourist_spots": "tourist_spots.xlsx",
            "restaurants": "restaurants.xlsx",
            "nightclubs": "nightclubs.xlsx",
            "tourism_packages": "tourism_packages.xlsx"
        }
        
        for dataset_name, filename in excel_files.items():
            file_path = os.path.join(EXCEL_DIR, filename)
            if os.path.exists(file_path):
                datasets[dataset_name] = pd.read_excel(file_path)
                logger.info(f"Cargado {dataset_name} con {len(datasets[dataset_name])} registros")
                
                # Normalizar columnas ID
                id_cols = [col for col in datasets[dataset_name].columns if col.startswith('id_')]
                if id_cols and 'id' not in datasets[dataset_name].columns:
                    datasets[dataset_name] = datasets[dataset_name].rename(columns={id_cols[0]: 'id'})
            else:
                logger.warning(f"Archivo no encontrado: {file_path}")
        
        return {
            "status": "success",
            "datasets_loaded": list(datasets.keys()),
            "total_records": {k: len(v) for k, v in datasets.items()}
        }
    except Exception as e:
        logger.error(f"Error cargando datos Excel: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
def get_recommendations(
    interests: list = None,
    budget: float = None,
    duration: float = None,
    locations: list = None,
    limit: int = 5
) -> list:
    """
    Obtiene recomendaciones personalizadas basadas en preferencias del usuario.
    
    Args:
        interests: Lista de intereses del usuario
        budget: Presupuesto total del viaje
        duration: Duración de la estancia en días
        locations: Ubicaciones de interés
        limit: Número máximo de recomendaciones
        
    Returns:
        Lista de recomendaciones personalizadas
    """
    recommendations = []
    
    # Calcular presupuesto diario si hay información
    daily_budget = None
    if budget and duration:
        daily_budget = budget / duration * 0.2  # 20% del presupuesto diario para actividades
    
    # Mapeo de intereses a términos de búsqueda
    interest_mapping = {
        "cultural": ["museum", "history", "culture", "art", "heritage"],
        "natural": ["beach", "nature", "park", "diving", "hiking"],
        "family": ["family", "kids", "children", "fun"],
        "gastronomy": ["food", "restaurant", "cuisine", "dining"],
        "nightlife": ["club", "bar", "party", "nightlife", "music"]
    }
    
    # Expandir intereses a términos de búsqueda
    search_terms = []
    if interests:
        for interest in interests:
            if interest in interest_mapping:
                search_terms.extend(interest_mapping[interest])
            else:
                search_terms.append(interest)
    
    # Buscar en datasets
    datasets_to_search = ["activities", "tourist_spots", "restaurants"]
    
    for dataset_name in datasets_to_search:
        if dataset_name not in datasets:
            continue
            
        df = datasets[dataset_name].copy()
        
        # Filtrar por ubicación
        if locations and 'location' in df.columns:
            location_mask = False
            for location in locations:
                location_mask = location_mask | df["location"].str.contains(location, case=False, na=False)
            df = df[location_mask]
        
        # Filtrar por presupuesto
        budget_column = None
        if "cost" in df.columns:
            budget_column = "cost"
        elif "entry_fee" in df.columns:
            budget_column = "entry_fee"
        elif "price" in df.columns:
            budget_column = "price"
        elif "average_person_expense" in df.columns:
            budget_column = "average_person_expense"
            
        if budget_column and daily_budget:
            df = df[df[budget_column] <= daily_budget]
        
        # Filtrar por intereses
        if search_terms:
            interest_columns = ["type", "description", "recommended_for", "ideal_for"]
            interest_columns = [col for col in interest_columns if col in df.columns]
            
            if interest_columns:
                interest_mask = False
                for col in interest_columns:
                    for term in search_terms:
                        col_mask = df[col].astype(str).str.contains(term, case=False, na=False)
                        interest_mask = interest_mask | col_mask
                
                df = df[interest_mask]
        
        # Ordenar por calificación
        if "rating" in df.columns:
            df = df.sort_values(by="rating", ascending=False)
        
        # Convertir a resultados
        for _, row in df.head(limit).iterrows():
            item_type = dataset_name
            if item_type.endswith('s'):
                item_type = item_type[:-1]
                
            recommendation = {
                "id": str(row["id"]),
                "name": row["name"],
                "type": item_type
            }
            
            # Añadir campos opcionales si existen
            for field in ["description", "location", "rating", "cost", "entry_fee"]:
                if field in row and not pd.isna(row[field]):
                    recommendation[field] = row[field]
            
            recommendations.append(recommendation)
    
    # Ordenar resultados por rating
    recommendations = sorted(
        recommendations, 
        key=lambda x: float(x.get("rating", 0)), 
        reverse=True
    )
    
    # Limitar resultados
    return recommendations[:limit]

@mcp.tool()
def get_item_details(item_id: str, item_type: str) -> dict:
    """
    Obtiene detalles completos de un elemento específico.
    
    Args:
        item_id: ID del elemento
        item_type: Tipo de elemento (activity, spot, restaurant, etc.)
        
    Returns:
        Detalles completos del elemento
    """
    dataset_mapping = {
        "activity": "activities",
        "spot": "tourist_spots",
        "restaurant": "restaurants",
        "nightclub": "nightclubs",
        "package": "tourism_packages"
    }
    
    dataset_name = dataset_mapping.get(item_type)
    if not dataset_name or dataset_name not in datasets:
        return {"error": f"Tipo de elemento no válido: {item_type}"}
    
    # Buscar elemento por ID
    df = datasets[dataset_name]
    
    # Convertir ID al tipo correcto si es necesario
    try:
        if df['id'].dtype == 'int64':
            item_id = int(item_id)
    except:
        pass
    
    item = df[df['id'] == item_id]
    if item.empty:
        return {"error": f"No se encontró elemento con ID {item_id}"}
    
    # Convertir a diccionario
    return item.iloc[0].to_dict()

# Cargar datos al iniciar
logger.info("Iniciando servidor MCP Excel y cargando datos...")
load_excel_data()

if __name__ == "__main__":
    logger.info(f"Servidor MCP Excel iniciado: {EXCEL_SERVER_ID}")
    mcp.serve()