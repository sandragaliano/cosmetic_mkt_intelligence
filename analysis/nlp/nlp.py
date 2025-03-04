# DEPENDENCIES
import os
import re
import json
import time
import pandas as pd
import numpy as np
import nltk
import spacy
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizer, BertForSequenceClassification
from collections import Counter

# Download NLTK resources
nltk.download('punkt')

# DIRECTORY SETUP
base_dir = 'C:/Users/sandr/Documents/scrp_tiktok_tfg'
data_dir = os.path.join(base_dir, 'data')
clean_data_dir = os.path.join(data_dir, 'clean_data')
analysis_dir = os.path.join(base_dir, 'analysis/nlp')

# Create directories if they don't exist
for directory in [data_dir, clean_data_dir, analysis_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directorio creado: {directory}")

# FILE PATHS
sephora_file = os.path.join(clean_data_dir, 'sephora_website_cleaned.csv')
tiktok_file = os.path.join(clean_data_dir, 'url_data_cleaned.xlsx')
output_file = os.path.join(analysis_dir, 'product_mentions_analysis.xlsx')

# Verify files exist
if not os.path.exists(sephora_file):
    raise FileNotFoundError(f"Archivo de datos de Sephora no encontrado: {sephora_file}")
if not os.path.exists(tiktok_file):
    raise FileNotFoundError(f"Archivo de datos de TikTok no encontrado: {tiktok_file}")

# LOAD DATA
sephora_df = pd.read_csv(sephora_file)
tiktok_df = pd.read_excel(tiktok_file)
url_column = 'id_urlvideo'

# NLP MODELS
print("Cargando modelos de NLP...")
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
)
nlp_en = spacy.load("en_core_web_md")

# HELPER FUNCTIONS
def extract_categories(categories_str):
    """
    Extrae categorías desde una string en formato 'Makeup', 'Face', 'Foundation'
    
    Args:
        categories_str (str): String con categorías separadas por comas
        
    Returns:
        list: Lista de categorías limpias
    """
    if not isinstance(categories_str, str):
        return []
    # Limpiamos las comillas y separamos las categorías
    categories = [cat.strip().strip("'").strip('"') for cat in categories_str.split(',')]
    return [cat for cat in categories if cat]  # Eliminar categorías vacías

def analyze_sentiment(text, analyzer):
    """
    Analiza el sentimiento de la transcripción.
    
    Args:
        text (str): transcription
        analyzer: Modelo de análisis de sentimiento
        
    Returns:
        dict: Diccionario con sentimiento y score
    """
    if pd.isna(text) or text == "":
        return {'sentiment': 'neutral', 'score': 0.5}
    
    # Si el texto es demasiado largo, dividirlo en fragmentos
    if len(text) > 512:
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        results = [analyzer(chunk)[0] for chunk in chunks]
        
        # Promediar los resultados
        labels = [r['label'] for r in results]
        scores = [r['score'] for r in results]
        
        # Contar la frecuencia de cada etiqueta
        label_counter = Counter(labels)
        most_common_label = label_counter.most_common(1)[0][0]
        
        # Score promedio para la etiqueta más común
        avg_score = sum([s for l, s in zip(labels, scores) if l == most_common_label]) / label_counter[most_common_label]
        
        return {
            'sentiment': most_common_label,
            'score': avg_score
        }
    else:
        result = analyzer(text)[0]
        return {
            'sentiment': result['label'],
            'score': result['score']
        }

def detect_sephora_products(transcription, sephora_products, model):
    """
    Detectamos en la transcripción los siguientes elementos de Sephora: 
    - Los productos de Sephora mencionados (title)
    - Las marcas detectadas (brands)
    - El tipo de productos mencionados (categories) en formato de lista: 'Makeup', 'Face', 'Foundation'
    
    Contempla tres escenarios:
    1. Se detecta el producto, la marca y la categoría del producto
    2. Se detecta solo la marca y la categoría
    3. Se detecta solo la marca o solo la categoría

    Args:
        transcription (str): url_data_cleaned('transcription')
        sephora_products (DataFrame): DataFrame de Sephora con columnas 'title', 'brand', 'product_with_brand' y 'categories'
        model: Modelo de similitud semántica
        
    Returns:
        list: Lista de diccionarios con los productos, marcas y categorías detectados.
    """
    if pd.isna(transcription) or transcription == "":
        return []
    
    # Dividir la transcripción en oraciones
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcription) if s.strip()]
    
    # Obtener listas de productos, marcas y categorías
    product_names = sephora_products['title'].dropna().tolist()
    brand_names = sephora_products['brand'].dropna().unique().tolist()
    
    # Extraer todas las categorías únicas
    all_categories = []
    for cats in sephora_products['categories'].dropna():
        all_categories.extend(extract_categories(cats))
    all_categories = list(set(all_categories))  # Eliminar duplicados
    
    stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 
    'just', 'don', 'should', 'now', 'like', 'yeah', 'uh', 'oh', 'hmm', 'hey', 'hi', 'hello', 
    'ok', 'okay', 'gonna', 'wanna', 'gotta', 'ya', 'nah', 'lol', 'omg', 'btw', 'idk', 'tbh', 
    'smh', 'imo', 'imho', 'brb', 'lmao', 'rofl', 'wtf', 'thx', 'pls', 'plz', 'dm', 'msg', 
    'thing', 'stuff', 'kinda', 'sorta', 'really', 'actually', 'basically', 'literally', 
    'probably', 'maybe', 'totally', 'definitely', 'honestly', 'seriously', 'whatever', 
    'anyway', 'k', 'bc', 'tho', 'didn', 'doesn', 'wasn', 'weren', 'ain', 'shouldn'
    }

    
    detected_items = []

    # ESCENARIO 1: Detectar productos completos (producto + marca + categoría)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Buscar menciones de marca primero
        for brand in brand_names:
            if not isinstance(brand, str) or len(brand) < 3:
                continue
                
            brand_lower = brand.lower()
            # Verificar si la marca está mencionada en la oración
            if re.search(r'\b' + re.escape(brand_lower) + r'\b', sentence_lower):
                # Buscar productos de esta marca
                brand_products = sephora_products[sephora_products['brand'] == brand]
                
                for _, product_row in brand_products.iterrows():
                    product_name = product_row['title']
                    if not isinstance(product_name, str):
                        continue
                        
                    product_name_lower = product_name.lower()
                    product_keywords = [
                        word for word in product_name_lower.split() 
                        if len(word) > 3 and word not in stopwords
                    ]
                    
                    # Contar cuántas palabras clave del producto aparecen en la oración
                    matches = sum(1 for keyword in product_keywords 
                                if re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower))
                    
                    # Requerir que al menos 30% de las palabras clave coincidan
                    min_matches = max(1, len(product_keywords) // 3)
                    
                    if matches >= min_matches:
                        # Obtener las categorías del producto
                        categories = extract_categories(product_row.get('categories', ''))
                        
                        product_entry = {
                            'product_id': product_row.name,
                            'brand': brand,
                            'product': product_name,
                            'categories': categories,
                            'sentence': sentence,
                            'detection_type': 'product_brand_category',
                            'method': 'exact_match',
                            'score': matches / max(1, len(product_keywords))
                        }
                        
                        # Evitar duplicados
                        if not any(entry.get('product') == product_name and entry.get('brand') == brand 
                                  for entry in detected_items):
                            detected_items.append(product_entry)
    
    # Si no se detectaron productos completos, intentar similitud semántica
    if not detected_items:
        product_embeddings = model.encode(product_names)
        sentence_embeddings = model.encode(sentences)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            similarities = np.inner(sentence_embeddings[i], product_embeddings)
            threshold = 0.6
            
            for j, score in enumerate(similarities):
                if score > threshold:
                    product_name = product_names[j]
                    product_info = sephora_products[sephora_products['product_with_brand'] == product_name]
                    
                    if not product_info.empty:
                        brand = product_info['brand'].values[0]
                        categories = extract_categories(product_info['categories'].values[0] 
                                                       if 'categories' in product_info else '')
                        
                        # Verificar que la marca esté mencionada
                        if brand.lower() in sentence_lower:
                            product_entry = {
                                'product_id': j,
                                'brand': brand,
                                'product': product_name,
                                'categories': categories,
                                'sentence': sentence,
                                'detection_type': 'product_brand_category',
                                'method': 'semantic_similarity',
                                'score': float(score)
                            }
                            
                            # Evitar duplicados
                            if not any(entry.get('product') == product_name and entry.get('brand') == brand 
                                      for entry in detected_items):
                                detected_items.append(product_entry)
    
    # ESCENARIO 2: Detectar solo marca y categoría
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Verificar si hay menciones de marcas
        detected_brands = []
        for brand in brand_names:
            if isinstance(brand, str) and len(brand) >= 3:
                brand_lower = brand.lower()
                if re.search(r'\b' + re.escape(brand_lower) + r'\b', sentence_lower):
                    detected_brands.append(brand)
        
        # Verificar si hay menciones de categorías
        detected_categories = []
        for category in all_categories:
            if isinstance(category, str) and len(category) >= 3:
                category_lower = category.lower()
                if re.search(r'\b' + re.escape(category_lower) + r'\b', sentence_lower):
                    detected_categories.append(category)
        
        # Si se detectó al menos una marca y una categoría
        if detected_brands and detected_categories:
            # Verificar si esta combinación no está ya en productos detectados
            new_entry = True
            for entry in detected_items:
                if (entry.get('detection_type') == 'product_brand_category' and 
                    any(brand == entry.get('brand', '') for brand in detected_brands) and
                    any(cat in entry.get('categories', []) for cat in detected_categories if entry.get('categories'))):
                    new_entry = False
                    break
            
            if new_entry:
                for brand in detected_brands:
                    brand_category_entry = {
                        'brand': brand,
                        'categories': detected_categories,
                        'sentence': sentence,
                        'detection_type': 'brand_category',
                        'method': 'exact_match',
                        'score': 1.0
                    }
                    detected_items.append(brand_category_entry)
    
    # ESCENARIO 3: Detectar solo marca o solo categoría
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Detectar solo marcas (si no están ya asociadas con categorías)
        for brand in brand_names:
            if not isinstance(brand, str) or len(brand) < 3:
                continue
                
            brand_lower = brand.lower()
            if re.search(r'\b' + re.escape(brand_lower) + r'\b', sentence_lower):
                # Verificar si esta marca no está ya en entradas previas
                if not any((entry.get('detection_type') in ['product_brand_category', 'brand_category'] and 
                            entry.get('brand') == brand and entry.get('sentence') == sentence) 
                          for entry in detected_items):
                    brand_entry = {
                        'brand': brand,
                        'sentence': sentence,
                        'detection_type': 'brand_only',
                        'method': 'exact_match',
                        'score': 1.0
                    }
                    detected_items.append(brand_entry)
        
        # Detectar solo categorías (si no están ya asociadas con marcas)
        for category in all_categories:
            if not isinstance(category, str) or len(category) < 3:
                continue
                
            category_lower = category.lower()
            if re.search(r'\b' + re.escape(category_lower) + r'\b', sentence_lower):
                # Verificar si esta categoría no está ya en entradas previas para esta oración
                if not any((entry.get('detection_type') in ['product_brand_category', 'brand_category'] and 
                           category in entry.get('categories', []) and entry.get('sentence') == sentence) 
                          for entry in detected_items):
                    category_entry = {
                        'categories': [category],
                        'sentence': sentence,
                        'detection_type': 'category_only',
                        'method': 'exact_match',
                        'score': 1.0
                    }
                    detected_items.append(category_entry)
    
    # Ordenar resultados por score (mayor primero)
    detected_items = sorted(detected_items, key=lambda x: x.get('score', 0), reverse=True)
    
    return detected_items

def load_processed_urls(output_path):
    """
    Carga las URLs ya procesadas de un archivo existente (Excel o CSV).
    
    Args:
        output_path (str): Ruta al archivo de resultados
        
    Returns:
        set: Conjunto de URLs ya procesadas
        pd.DataFrame: DataFrame con los resultados existentes o vacío
    """
    processed_urls = set()
    existing_results = pd.DataFrame()
    
    # Determinar si el archivo es Excel o CSV
    is_excel = output_path.endswith('.xlsx')
    
    # Verificar también el archivo alternativo
    alternative_path = output_path.replace('.xlsx', '.csv') if is_excel else output_path.replace('.csv', '.xlsx')
    
    # Verificar primero el archivo principal
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            if is_excel:
                existing_results = pd.read_excel(output_path, sheet_name='Resultados')
            else:
                existing_results = pd.read_csv(output_path)
                
            print(f"Archivo de resultados existente cargado: {output_path}")
            print(f"Registros existentes: {len(existing_results)}")
            
        except Exception as e:
            print(f"Error al cargar archivo existente {output_path}: {e}")
            existing_results = pd.DataFrame()
    
    # Si no se pudo cargar el archivo principal, intentar con el alternativo
    elif os.path.exists(alternative_path) and os.path.getsize(alternative_path) > 0:
        try:
            if alternative_path.endswith('.xlsx'):
                existing_results = pd.read_excel(alternative_path, sheet_name='Resultados')
            else:
                existing_results = pd.read_csv(alternative_path)
                
            print(f"Archivo alternativo de resultados cargado: {alternative_path}")
            print(f"Registros existentes: {len(existing_results)}")
            
        except Exception as e:
            print(f"Error al cargar archivo alternativo {alternative_path}: {e}")
            existing_results = pd.DataFrame()
    else:
        print(f"No se encontró archivo de resultados existente. Se creará uno nuevo.")
    
    # Extraer URLs procesadas si hay resultados
    if not existing_results.empty:
        # Identificar todas las posibles columnas que pueden contener la URL o ID
        id_columns = []
        for col in ['video_url', 'video_id', 'id_urlvideo']:
            if col in existing_results.columns:
                id_columns.append(col)
        
        if id_columns:
            # Procesar cada columna identificada
            for col in id_columns:
                urls = existing_results[col].dropna().astype(str).unique()
                processed_urls.update(urls)
            print(f"URLs ya procesadas: {len(processed_urls)}")
        else:
            print("ADVERTENCIA: No se encontró columna de identificación de video en el archivo existente.")
    
    return processed_urls, existing_results

def save_results(results_df, new_results, output_path):
    """
    Guarda los resultados en un archivo Excel, evitando columnas duplicadas.
    
    Args:
        results_df (pd.DataFrame): DataFrame con resultados existentes
        new_results (list): Lista de nuevos resultados
        output_path (str): Ruta para guardar resultados
        
    Returns:
        pd.DataFrame: DataFrame con todos los resultados combinados
    """
    # Cambiar la extensión del archivo a .xlsx
    if output_path.endswith('.csv'):
        output_path = output_path.replace('.csv', '.xlsx')
    
    # Convertir nuevos resultados a DataFrame
    if new_results:
        new_results_df = pd.DataFrame(new_results)
        
        # Combinar con resultados existentes si los hay
        if not results_df.empty:
            combined_results = pd.concat([results_df, new_results_df], ignore_index=True)
        else:
            combined_results = new_results_df
        
        # Crear copia para guardar
        combined_results_to_save = combined_results.copy()
        
        # Convertir columnas de tipo complejo a string para evitar errores al guardar
        complex_columns = ['categories']
        for col in complex_columns:
            if col in combined_results_to_save.columns:
                # Asegurarse de que se usa json.dumps para serializar correctamente
                combined_results_to_save[col] = combined_results_to_save[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if x is not None else "[]"
                )
        
        # Guardar a Excel con formato
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Hoja principal con todos los resultados
                combined_results_to_save.to_excel(writer, sheet_name='Resultados', index=False)
                
                # Crear hojas adicionales con análisis agregados
                if 'detection_type' in combined_results_to_save.columns:
                    # Resumen por tipo de detección
                    detection_summary = combined_results_to_save['detection_type'].value_counts().reset_index()
                    detection_summary.columns = ['Tipo de Detección', 'Cantidad']
                    detection_summary.to_excel(writer, sheet_name='Resumen_Detecciones', index=False)
                
                if 'brand' in combined_results_to_save.columns:
                    # Resumen por marca
                    brand_summary = combined_results_to_save['brand'].value_counts().reset_index()
                    brand_summary.columns = ['Marca', 'Menciones']
                    brand_summary.to_excel(writer, sheet_name='Resumen_Marcas', index=False)
                
                if 'sentiment' in combined_results_to_save.columns:
                    # Resumen por sentimiento
                    sentiment_summary = combined_results_to_save['sentiment'].value_counts().reset_index()
                    sentiment_summary.columns = ['Sentimiento', 'Cantidad']
                    sentiment_summary.to_excel(writer, sheet_name='Resumen_Sentimiento', index=False)
        
            print(f"Resultados guardados en formato Excel: {output_path}")
        except Exception as e:
            print(f"Error al guardar en Excel: {e}")
            # Fallback: guardar como CSV si hay error con Excel
            csv_path = output_path.replace('.xlsx', '.csv')
            combined_results_to_save.to_csv(csv_path, index=False)
            print(f"Resultados guardados en formato CSV como fallback: {csv_path}")
        
        # Devolver los resultados combinados con las columnas originales
        return combined_results
    
    return results_df

def process_tiktok_videos(tiktok_df, sephora_df, sentence_model, sentiment_analyzer, 
                         output_path, processed_urls=None, existing_results=None, save_interval=5):
    """
    Procesa los videos de TikTok para detectar productos y analizar sentimiento.
    Guarda los resultados de forma incremental.
    
    Args:
        tiktok_df (pd.DataFrame): DataFrame con datos de TikTok
        sephora_df (pd.DataFrame): DataFrame con productos de Sephora
        sentence_model: Modelo para similitud semántica
        sentiment_analyzer: Modelo para análisis de sentimiento
        output_path (str): Ruta para guardar resultados
        processed_urls (set, optional): Conjunto de URLs ya procesadas
        existing_results (pd.DataFrame, optional): DataFrame con resultados existentes
        save_interval (int, optional): Intervalo para guardar resultados parciales
        
    Returns:
        pd.DataFrame: DataFrame con todos los resultados
    """
    if processed_urls is None:
        processed_urls = set()
    
    if existing_results is None or existing_results.empty:
        results_df = pd.DataFrame()
    else:
        results_df = existing_results.copy()
    
    new_results = []
    processed_count = 0
    
    if 'transcription' not in tiktok_df.columns:
        print("Error: No se encontró columna 'transcription' en el DataFrame de TikTok")
        return results_df
    
    # Estadísticas iniciales
    total_videos = len(tiktok_df)
    skipped_count = 0
    print(f"Total de videos a procesar: {total_videos}")
    print(f"Videos ya procesados anteriormente: {len(processed_urls)}")
    
    # Recopilar IDs de videos disponibles
    all_video_ids = set()
    for idx, row in tiktok_df.iterrows():
        video_id = str(row['id_urlvideo'] if 'id_urlvideo' in row else str(idx))
        all_video_ids.add(video_id)
        if url_column in row:
            all_video_ids.add(str(row[url_column]))
    
    print(f"Total de URLs/IDs únicos en el dataset actual: {len(all_video_ids)}")
    print(f"URLs/IDs por procesar: {len(all_video_ids - processed_urls)}")
    
    # Crear lista de videos a procesar
    videos_to_process = []
    for idx, row in tiktok_df.iterrows():
        video_id = str(row['id_urlvideo'] if 'id_urlvideo' in row else str(idx))
        video_url = str(row[url_column]) if url_column in row else video_id
        
        # Si ni el ID ni la URL están en processed_urls, agregar a la lista
        if video_id not in processed_urls and video_url not in processed_urls:
            videos_to_process.append(idx)
        else:
            skipped_count += 1
    
    print(f"Se procesarán {len(videos_to_process)} videos, omitiendo {skipped_count} ya procesados.")
    
    # Procesar videos pendientes
    for i, idx in enumerate(videos_to_process):
        row = tiktok_df.loc[idx]
        video_id = str(row['id_urlvideo'] if 'id_urlvideo' in row else str(idx))
        video_url = str(row[url_column]) if url_column in row else video_id
        
        transcription = row['transcription']
        
        print(f"Procesando video {i+1}/{len(videos_to_process)} (ID: {video_id})")
        
        # Detectar productos
        detected_items = detect_sephora_products(transcription, sephora_df, sentence_model)
        
        if not detected_items:
            print(f"  No se detectaron menciones para el video {video_id}")
            # Guardar un registro vacío para indicar que el video fue procesado
            empty_result = {
                'video_id': video_id,
                'id_urlvideo': video_id,
                'detection_type': 'no_detection',
                'method': 'none',
                'score': 0,
                'sentiment': None,
                'sentiment_score': None
            }
            new_results.append(empty_result)
        else:
            # Análisis de sentimiento para toda la transcripción
            overall_sentiment = analyze_sentiment(transcription, sentiment_analyzer)
            
            print(f"  Se detectaron {len(detected_items)} menciones para el video {video_id}")
            
            # Procesar cada detección
            for item in detected_items:
                # Análisis de sentimiento para la oración donde se detectó la mención
                sentence_sentiment = analyze_sentiment(item.get('sentence', ''), sentiment_analyzer)
                
                # Preparar resultado
                result = {
                    'video_id': video_id,
                    'id_urlvideo': video_id,
                    'detection_type': item.get('detection_type', 'unknown'),
                    'method': item.get('method', 'unknown'),
                    'score': item.get('score', 0),
                    'sentence': item.get('sentence', ''),
                    'sentiment': sentence_sentiment['sentiment'],
                    'sentiment_score': sentence_sentiment['score'],
                    'overall_sentiment': overall_sentiment['sentiment'],
                    'overall_sentiment_score': overall_sentiment['score']
                }
                
                # Añadir campos específicos según el tipo de detección
                if 'product' in item:
                    result['product'] = item['product']
                    result['product_id'] = item.get('product_id', None)
                
                if 'brand' in item:
                    result['brand'] = item['brand']
                
                if 'categories' in item:
                    result['categories'] = item['categories']
                
                new_results.append(result)
            
        # Marcar como procesado
        processed_urls.add(video_url)
        processed_urls.add(video_id)
        processed_count += 1
        
        # Guardar resultados parciales
        if (i+1) % save_interval == 0 and new_results:
            print(f"Guardando resultados parciales después de procesar {i+1}/{len(videos_to_process)} videos...")
            results_df = save_results(results_df, new_results, output_path)
            new_results = []  # Limpiar lista después de guardar
    
    # Guardar resultados finales si quedan pendientes
    if new_results:
        print("Guardando resultados finales...")
        results_df = save_results(results_df, new_results, output_path)
    
    print(f"Procesamiento completado. Se procesaron {processed_count} videos nuevos.")
    return results_df

def main():
    """Función principal de ejecución del script"""
    start_time = time.time()
    # Cargar datos ya procesados
    processed_urls, existing_results = load_processed_urls(output_file)
    
    print("Iniciando procesamiento de videos...")
    save_interval = 5  # Guardar cada 5 videos procesados
    
    # Procesar videos
    results_df = process_tiktok_videos(
        tiktok_df, 
        sephora_df, 
        sentence_model, 
        sentiment_analyzer, 
        output_file,
        processed_urls,
        existing_results,
        save_interval
    )
    
    print(f"Procesamiento completo. Archivo guardado en: {output_file}")
    
    # ESTADÍSTICAS FINALES
    if len(results_df) > 0:
        # Verificar si existe la columna detection_type antes de usarla
        if 'detection_type' in results_df.columns:
            product_mentions = results_df[results_df['detection_type'] != 'no_detection']
        elif 'detection_method' in results_df.columns:
            # Compatibilidad con versiones anteriores que podrían usar detection_method
            product_mentions = results_df[results_df['detection_method'] != 'no_detection']
        else:
            # Si no existe ninguna de las columnas, usar todo el dataset
            print("Aviso: No se encontró columna 'detection_type' o 'detection_method'. Usando todos los registros.")
            product_mentions = results_df
        
        print(f"Menciones válidas: {len(product_mentions)}")
        
        if len(product_mentions) > 0:
            print("\nTipos de detecciones:")
            if 'detection_type' in product_mentions.columns:
                print(product_mentions['detection_type'].value_counts())
            elif 'detection_method' in product_mentions.columns:
                print(product_mentions['detection_method'].value_counts())
                
            print("\nMarcas mencionadas:")
            if 'brand' in product_mentions.columns:
                print(product_mentions['brand'].value_counts().head(10))
            
            print("\nSentimiento general:")
            if 'sentiment' in product_mentions.columns:
                print(product_mentions['sentiment'].value_counts())
            
            print("\nVideos procesados:")
            print(f"Total: {len(results_df['video_id'].unique())}")
            print(f"Con menciones: {len(product_mentions['video_id'].unique())}")
    
    # Mostrar tiempo total de ejecución
    end_time = time.time()
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("=" * 50)
    print("FIN DEL PROCESAMIENTO")
    print("=" * 50)

# Ejecutar el script si se llama directamente
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()