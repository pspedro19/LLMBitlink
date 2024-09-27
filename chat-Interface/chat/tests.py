import requests

url = "http://localhost:8800/save_vectorization/"

# Datos de prueba
data = [
    {
        "document_id": 1,
        "content": "Este es un texto de prueba para vectorización.",
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    {
        "document_id": 2,
        "content": "Otro documento de ejemplo.",
        "embedding": [0.9, 0.8, 0.7, 0.6, 0.5]
    }
]

# Realizar la solicitud POST
response = requests.post(url, json=data)

# Mostrar la respuesta
print(f"Status code: {response.status_code}")
print(f"Response body: {response.json()}")