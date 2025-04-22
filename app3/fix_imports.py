import os
import re

def fix_imports_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Reemplazar 'from ' con 'from '
    updated_content = re.sub(r'from app\.', 'from ', content)
    # Reemplazar 'import ' con 'import '
    updated_content = re.sub(r'import app\.', 'import ', updated_content)
    
    if content != updated_content:
        print(f"Actualizando: {file_path}")
        with open(file_path, 'w') as file:
            file.write(updated_content)

def scan_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                fix_imports_in_file(os.path.join(root, file))

if __name__ == "__main__":
    scan_directory('.')
    print("Importaciones corregidas en todos los archivos.")
