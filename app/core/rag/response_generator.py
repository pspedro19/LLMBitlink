# core/rag/response_generator.py
def generate_paraphrased_response(results):
    """
    Genera una respuesta parafraseada a partir de los resultados de búsqueda.
    
    Args:
        results (List[Dict]): Lista de resultados recuperados.
    
    Returns:
        str: Respuesta parafraseada.
    """
    if not results:
        return "No se encontraron resultados relevantes para tu consulta."
    
    response = "He encontrado "
    response += f"{len(results)} resultado(s) relevante(s) para tu consulta. "
    titles = [res.get("title", "Sin título") for res in results]
    response += "Por ejemplo, se destacan: " + ", ".join(titles) + "."
    return response
