import argparse
import os
import sys
import subprocess

# Añadir la ruta raíz al path para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_tiktok_extraction():
    # Paso 1: Extracción de URLs de TikTok
    print("PASO 1: Ejecutando extracción de URLs de TikTok...")
    import tiktokurl_extraction
    tiktokurl_extraction.main()

def run_etl_notebook():
    # Paso 2: Ejecutar el notebook de EDA y ETL
    print("PASO 2: Ejecutando procesamiento de EDA y ETL...")
    
    # Para ejecutar un notebook, necesitamos usar nbconvert
    notebook_path = "etl_and_eda/eda&etl_url_data.ipynb"
    if os.path.exists(notebook_path):
        subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--inplace",
            notebook_path
        ])
        print(f"Notebook {notebook_path} ejecutado correctamente")
    else:
        print(f"Error: No se encontró el notebook en {notebook_path}")
        print("Verifica la ruta correcta al notebook")

def run_nlp_analysis():
    # Paso 3: Análisis de NLP
    print("PASO 3: Ejecutando análisis NLP...")
    from analysis.nlp import nlp
    nlp.main()

def run_full_pipeline():
    print("Iniciando pipeline completo de procesamiento...")
    run_tiktok_extraction()
    run_etl_notebook()
    run_nlp_analysis()
    print("Pipeline completado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline de procesamiento de datos de TikTok')
    parser.add_argument('--all', action='store_true', help='Ejecutar todo el pipeline en orden')
    parser.add_argument('--extract', action='store_true', help='Ejecutar solo la extracción de TikTok')
    parser.add_argument('--etl', action='store_true', help='Ejecutar solo el notebook de EDA y ETL')
    parser.add_argument('--nlp', action='store_true', help='Ejecutar solo el análisis NLP')
    
    args = parser.parse_args()
    
    if args.all:
        run_full_pipeline()
    else:
        if args.extract:
            run_tiktok_extraction()
        if args.etl:
            run_etl_notebook()
        if args.nlp:
            run_nlp_analysis()

    # Si no se especificó ningún argumento, mostrar ayuda
    if not (args.all or args.extract or args.etl or args.nlp):
        parser.print_help()