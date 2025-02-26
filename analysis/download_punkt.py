import nltk

# Descargar el recurso punkt normal
nltk.download('punkt')

# También intentar descargar punto_tab
try:
    nltk.download('punkt_tab')
except:
    print("No se pudo descargar punkt_tab directamente")
    
    # Intentar una solución alternativa creando el directorio
    import os
    import shutil
    from nltk.data import find
    
    try:
        # Encontrar dónde está instalado punkt
        punkt_path = find('tokenizers/punkt')
        
        # Crear el directorio punkt_tab si no existe
        nltk_data_path = os.path.dirname(os.path.dirname(punkt_path))
        punkt_tab_dir = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
        os.makedirs(punkt_tab_dir, exist_ok=True)
        
        # Crear subdirectorio english
        english_dir = os.path.join(punkt_tab_dir, 'english')
        os.makedirs(english_dir, exist_ok=True)
        
        # Copiar archivos de punkt a punkt_tab/english
        punkt_files = os.listdir(punkt_path)
        for file in punkt_files:
            src = os.path.join(punkt_path, file)
            dst = os.path.join(english_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        
        print(f"Creado directorio punkt_tab en {punkt_tab_dir}")
    except Exception as e:
        print(f"Error al crear directorio punkt_tab: {e}")

print("Proceso completado")