"""
Error handling middleware for the API.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from datetime import datetime
from utils.logger import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

ERROR_TYPES = {
    'validation_error': 'Data validation failed',
    'recommendation_error': 'Unable to generate recommendations',
    'query_error': 'Error processing query',
    'database_error': 'Database operation failed',
    'preference_error': 'Error processing preferences'
}

async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global error handler for the API
    
    Args:
        request (Request): FastAPI request object
        exc (Exception): Exception that was raised
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(
        error_type="server_error",
        message=str(exc),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    
    return JSONResponse(
        status_code=error_response["status_code"],
        content=error_response
    )

def create_error_response(
    error_type: str,
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response
    
    Args:
        error_type (str): Type of error
        message (str): Error message
        status_code (int): HTTP status code
        details (Dict[str, Any], optional): Additional error details
        
    Returns:
        Dict[str, Any]: Formatted error response
    """
    error_message = ERROR_TYPES.get(error_type, 'Unknown error')
    
    response = {
        "status": "error",
        "error_type": error_type,
        "error_message": f"{error_message}: {message}",
        "status_code": status_code,
        "timestamp": datetime.now().isoformat(),
        "path": None,
        "method": None
    }
    
    if details:
        response["details"] = details
        
    return response

def log_error(
    error_type: str,
    message: str,
    details: Dict[str, Any] = None
) -> None:
    """
    Log error with consistent formatting
    
    Args:
        error_type (str): Type of error
        message (str): Error message
        details (Dict[str, Any], optional): Additional error details
    """
    try:
        error_message = ERROR_TYPES.get(error_type, 'Unknown error')
        log_message = f"{error_message}: {message}"
        
        if details:
            log_message += f"\nDetails: {details}"
            
        logger.error(log_message)
    except Exception as e:
        logger.error(f"Error in error logging: {str(e)}")

class TourismError(Exception):
    """Custom error class for tourism system"""
    
    def __init__(self, error_type: str, message: str, details: Dict[str, Any] = None):
        self.error_type = error_type
        self.message = message
        self.details = details
        super().__init__(self.message)

def handle_tourism_error(error: TourismError) -> JSONResponse:
    """
    Handle custom tourism errors
    
    Args:
        error (TourismError): Tourism error to handle
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(
        error_type=error.error_type,
        message=error.message,
        details=error.details
    )
    
    log_error(
        error_type=error.error_type,
        message=error.message,
        details=error.details
    )
    
    return JSONResponse(
        status_code=error_response["status_code"],
        content=error_response
    )