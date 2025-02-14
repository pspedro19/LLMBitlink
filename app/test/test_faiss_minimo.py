import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)
query = np.zeros((1, dimension), dtype=np.float32)
distances, ids = index.search(query, 1)
print("Distancias:", distances)
print("IDs:", ids)
