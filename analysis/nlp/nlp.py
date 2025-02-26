# DEPENDENCIES
# !pip install transformers sentence-transformers nltk spacy torch pandas
import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import nltk
import json
from cosmetic_patterns import cosmetic_patterns
nltk.download('punkt')

# python -m spacy download es_core_news_md

# Verificación de directorios
base_dir = 'C:/Users/sandr/Documents/scrp_tiktok_tfg'
data_dir = os.path.join(base_dir, 'data')
clean_data_dir = os.path.join(data_dir, 'clean_data')
analysis_dir = os.path.join(base_dir, 'analysis/nlp')

for directory in [data_dir, clean_data_dir, analysis_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directorio creado: {directory}")

# Paths archivos
sephora_file = os.path.join(clean_data_dir, 'sephora_website_cleaned.csv')
tiktok_file = os.path.join(clean_data_dir, 'url_data_cleaned.xlsx')
output_file = os.path.join(analysis_dir, 'product_mentions_analysis.csv')

# Verificar la existencia de los archivos
if not os.path.exists(sephora_file):
    raise FileNotFoundError(f"Archivo de datos de Sephora no encontrado: {sephora_file}")
if not os.path.exists(tiktok_file):
    raise FileNotFoundError(f"Archivo de datos de TikTok no encontrado: {tiktok_file}")

# Carga
sephora_df = pd.read_csv(sephora_file)

tiktok_df = pd.read_excel(tiktok_file)
url_column = 'id_urlvideo'

cosmetic_patterns = [
    # Hidratación y nutrición
    r'\b(hydrat(?:ing|ion|e)|moistur(?:e|izing)|quenching|deep hydration|plumping|dewy|refreshing)\b',
    r'\b(nourishing|repairing|strengthening|conditioning|restorative|replenishing|soothing|revitalizing)\b',

    # Acabado
    r'\b(matte|opaque|velvety|satin-matte|semi-matte|powdery|soft-matte|chalky|flat)\b',  # Acabado mate
    r'\b(shiny|glow(?:ing)?|radiant|dewy|illuminating|luminous|pearlescent|glossy|wet-look|shimmering)\b',  # Brillante
    r'\b(natural finish|skin-like|second-skin|subtle glow|soft-focus|blurred finish|velvety finish|airbrushed)\b',  # Natural
    r'\b(glowy|hydrated|glossy finish|glow-up|radiant finish|luminous finish)\b',  # Acabado luminoso

    # Tono y color
    r'\b(warm(?:-toned)?|cool(?:-toned)?|neutral(?:-toned)?|olive(?:-toned)?|rose-toned|yellow-toned|pink-toned)\b',  # Subtono
    r'\b(full coverage|medium coverage|sheer|buildable|tinted|translucent|color-adapting|light coverage)\b',  # Cobertura
    r'\b(color-correcting|tone-correcting|even skin tone|complexion-enhancing|brightening|neutralizing|complexion-perfecting)\b',  # Corrección de tono
    r'\b(high pigment|intense color|vivid|rich color|bold|color payoff|multi-dimensional|true-to-color)\b',  # Pigmentación

    # Duración y resistencia
    r'\b(long-lasting|24-hour wear|all-day wear|extended wear|fade-resistant|sweat-proof|heat-resistant|humidity-resistant)\b',
    r'\b(waterproof|smudge-proof|transfer-resistant|humidity-proof|oil-proof|weatherproof|mask-proof|teardrop-resistant)\b',

    # Textura y sensación
    r'\b(texture|smooth(?:ing)?|silky|lightweight|bouncy|creamy|buttery|airy|gel-based|whipped|mousse-like|featherlight|soft-touch)\b',
    r'\b(non-sticky|non-greasy|fast-absorbing|quick-dry|cooling|refreshing|weightless|velvety-smooth|hydrating)\b',

    # Protección y cuidado de la piel
    r'\b(SPF|sun protection|UV protection|broad spectrum|UVA/UVB protection|sunscreen-infused|sun-kissed|UV defense)\b',
    r'\b(anti-age|anti-aging|rejuvenating|firming|wrinkle reduction|youth-boosting|plumping|collagen-boosting|tightening)\b',
    r'\b(soothing|calming|redness-reducing|anti-inflammatory|gentle|hypoallergenic|sensitive-skin friendly|non-irritating)\b',
    r'\b(non-comedogenic|won’t clog pores|acne-safe|dermatologist-tested|skin barrier support|pore-refining|anti-breakout)\b',
    r'\b(exfoliating|resurfacing|cell turnover|AHA|BHA|glycolic acid|salicylic acid|retinol-infused|brightening acids)\b',
    r'\b(pore-minimizing|blurring|soft-focus|skin-smoothing|airbrushed finish|filter effect|flawless)\b',
    r'\b(firming|skin-tightening|elasticity-boosting|anti-sagging|lifting effect|contouring|sculpting)\b',

    # Tipo de piel
    r'\b(oily skin|dry skin|combination skin|normal skin|sensitive skin|acne-prone skin|mature skin|problem skin)\b',
    r'\b(oil-free|hydrating|non-drying|non-comedogenic|mattifying|moisturizing|balancing)\b',
    r'\b(pore-refining|pore-minimizing|pore-filling|oil-absorbing|sebum-controlling)\b',

    # Ingredientes y fórmula
    r'\b(formula|composition|blend|infused with|enriched with|custom formula|advanced formula|innovative formula|lightweight formula)\b',
    r'\b(ingredients|active ingredients|botanical extracts|clean formula|natural extracts|essential oils|peptides|anti-oxidants|vitamin C)\b',
    r'\b(vegan|cruelty-free|plant-based|no animal testing|eco-friendly|sustainable|biodegradable|paraben-free|silicone-free|gluten-free|alcohol-free)\b',

    # Aplicación y uso
    r'\b(easy to blend|streak-free|seamless application|finger-friendly|brush-friendly|sponge-friendly|mess-free)\b',
    r'\b(multitasking|2-in-1|3-in-1|multi-use|hybrid formula|primer-infused|self-setting|no powder needed|all-in-one)\b'
    ]

# Carga de BERT:
print("Cargando modelos de NLP...")
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Carga de NER
nlp_en = spacy.load("en_core_web_md")
# Intentamos cargar el modelo en español si está disponible
try:
    nlp_es = spacy.load("es_core_news_md")
    print("Modelo SpaCy en español cargado correctamente.")
except:
    print("Modelo SpaCy en español no encontrado. Solo se utilizará el modelo en inglés.")
    nlp_es = None


def extract_categories(categories_str):
    """Extrae categorías desde una string en formato 'Makeup', 'Face', 'Foundation'"""
    if not isinstance(categories_str, str):
        return []
    # Limpiamos las comillas y separamos las categorías
    categories = [cat.strip().strip("'").strip('"') for cat in categories_str.split(',')]
    return [cat for cat in categories if cat]  # Eliminar categorías vacías


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
              Cada diccionario puede contener 'product', 'brand' y/o 'categories' según lo que se haya detectado.
    """
    if pd.isna(transcription) or transcription == "":
        return []
    
    # Asegurarse de que la transcripción esté en inglés (aquí iría el código de traducción si es necesario)
    transcription_en = transcription
    
    # Dividir la transcripción en oraciones
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcription_en) if s.strip()]
    
    # Obtener listas de productos, marcas y categorías
    product_names = sephora_products['product_with_brand'].dropna().tolist()
    brand_names = sephora_products['brand'].dropna().unique().tolist()
    
    # Extraer todas las categorías únicas
    all_categories = []
    for cats in sephora_products['categories'].dropna():
        all_categories.extend(extract_categories(cats))
    all_categories = list(set(all_categories))  # Eliminar duplicados
    
    stopwords = {'the', 'el', 'la', 'los', 'las', 'de', 'para', 'con', 'and', 'y', 'a', 'o', 'u'}
    
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
                    
                    # Requerir que al menos 50% de las palabras clave coincidan
                    min_matches = max(1, len(product_keywords) // 2)
                    
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
            threshold = 0.7
            
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
        from collections import Counter
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


def extract_entities(text, nlp_es, nlp_en):
    """
    Extrae entidades nombradas de un texto.
    
    Args:
        text (str): transcripcion
        nlp_es: Modelo SpaCy en español
        nlp_en: Modelo SpaCy en inglés
        
    Returns:
        list: Lista de entidades extraídas con tipo y texto
    """
    if pd.isna(text) or text == "":
        return []
    
    # Detectar idioma (simple)
    spanish_words = ["el", "la", "los", "las", "es", "son", "para", "con", "y", "que", "de"]
    text_lower = text.lower()
    spanish_count = sum([1 for word in spanish_words if f" {word} " in f" {text_lower} "])
    
    # Seleccionar modelo según idioma detectado
    nlp = nlp_es if spanish_count > 2 else nlp_en
    
    # Procesar texto
    doc = nlp(text)
    
    # Extraer entidades
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'start': ent.start_char,
            'end': ent.end_char,
            'type': ent.label_
        })
    
    # Extraer atributos cosméticos usando patrones específicos: usamos cosmetic_patterns del cosmetic_patterns.py
    for pattern in cosmetic_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'COSMETIC_ATTRIBUTE'
            })
    
    return entities


def load_processed_urls(output_path):
    """
    Carga las URLs ya procesadas de un archivo existente.
    
    Args:
        output_path (str): Ruta al archivo de resultados
        
    Returns:
        set: Conjunto de URLs ya procesadas
        pd.DataFrame: DataFrame con los resultados existentes o vacío
    """
    processed_urls = set()
    existing_results = pd.DataFrame()
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            existing_results = pd.read_csv(output_path)
            print(f"Archivo de resultados existente cargado: {output_path}")
            print(f"Registros existentes: {len(existing_results)}")
            
            # Identificamos la columna que contiene las URLs o IDs de video
            id_column = None
            for col in ['video_url', 'video_id', 'id_urlvideo']:
                if col in existing_results.columns:
                    id_column = col
                    break
            
            if id_column:
                processed_urls = set(existing_results[id_column].dropna().unique())
                print(f"URLs ya procesadas: {len(processed_urls)}")
            else:
                print("No se encontró columna de identificación de video en el archivo existente.")
                
        except Exception as e:
            print(f"Error al cargar archivo existente: {e}")
            existing_results = pd.DataFrame()
    else:
        print(f"No se encontró archivo de resultados existente. Se creará uno nuevo.")
    
    return processed_urls, existing_results


def save_results(results_df, new_results, output_path):
    """
    Guarda los resultados en un archivo CSV.
    
    Args:
        results_df (pd.DataFrame): DataFrame con resultados existentes
        new_results (list): Lista de nuevos resultados
        output_path (str): Ruta para guardar resultados
        
    Returns:
        pd.DataFrame: DataFrame con todos los resultados combinados
    """
    # Convertir nuevos resultados a DataFrame
    if new_results:
        new_results_df = pd.DataFrame(new_results)
        
        # Combinar con resultados existentes si los hay
        if not results_df.empty:
            combined_results = pd.concat([results_df, new_results_df], ignore_index=True)
        else:
            combined_results = new_results_df
        
        # Crear copia para guardar (evitamos problemas con tipos complejos)
        combined_results_to_save = combined_results.copy()
        
        # Convertir columnas de tipo complejo a string para evitar errores al guardar
        complex_columns = ['extracted_entities', 'categories']
        for col in complex_columns:
            if col in combined_results_to_save.columns:
                combined_results_to_save[f'{col}_str'] = combined_results_to_save[col].apply(lambda x: str(x) if x is not None else "[]")
                combined_results_to_save = combined_results_to_save.drop(columns=[col])
        
        # Guardar a CSV
        combined_results_to_save.to_csv(output_path, index=False)
        print(f"Resultados guardados en: {output_path}")
        
        return combined_results
    
    return results_df


def process_tiktok_videos(tiktok_df, sephora_df, sentence_model, sentiment_analyzer, nlp_model, 
                         output_path, processed_urls=None, existing_results=None, save_interval=5):
    """
    Procesa los videos de TikTok para detectar productos, analizar sentimiento y extraer entidades.
    Guarda los resultados de forma incremental.
    
    Args:
        tiktok_df (pd.DataFrame): DataFrame con datos de TikTok
        sephora_df (pd.DataFrame): DataFrame con datos de Sephora
        sentence_model: Modelo de similitud semántica
        sentiment_analyzer: Modelo de análisis de sentimiento
        nlp_model: Modelo SpaCy para procesamiento de lenguaje
        output_path (str): Ruta para guardar resultados
        processed_urls (set): Conjunto de URLs ya procesadas
        existing_results (pd.DataFrame): DataFrame con resultados existentes
        save_interval (int): Intervalo para guardar resultados
        
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
    
    # Filtramos solo los videos no procesados
    total_videos = len(tiktok_df)
    skipped_count = 0
    
    for idx, row in tiktok_df.iterrows():
        video_id = row['id_urlvideo'] if 'id_urlvideo' in row else str(idx)
        video_url = row[url_column] if url_column in row else video_id
        
        # Verificar si ya fue procesado
        if video_url in processed_urls or video_id in processed_urls:
            skipped_count += 1
            if skipped_count % 10 == 0:
                print(f"Omitidos {skipped_count} videos ya procesados...")
            continue
        
        transcription = row['transcription']
        
        print(f"Procesando video {processed_count+1}/{total_videos-skipped_count} (ID: {video_id})")
        
        # DETECTAR PRODUCTOS
        detected_items = detect_sephora_products(transcription, sephora_df, sentence_model)
        
        if not detected_items:
            print(f"  No se detectaron menciones para el video {video_id}")
            # Aún así, marcar como procesado
            processed_urls.add(video_url)
            processed_urls.add(video_id)
            processed_count += 1
            # Guardar un registro vacío para indicar que el video fue procesado
            empty_result = {
                'video_id': video_id,
                'detection_type': 'no_detection',
                'method': 'none',
                'score': 0,
                'sentiment': None,
                'sentiment_score': None
            }
            new_results.append(empty_result)
        else:
            print(f"  Se detectaron {len(detected_items)} menciones para el video {video_id}")
            
            # Procesar cada detección
            for item in detected_items:
                # Análisis de sentimiento para la oración donde se detectó la mención
                sentiment = analyze_sentiment(item.get('sentence', ''), sentiment_analyzer)
                
                # Extracción de entidades
                entities = extract_entities(item.get('sentence', ''), nlp_es, nlp_en)
                
                # Preparar resultado según el tipo de detección
                result = {
                    'video_id': video_id,
                    'detection_type': item.get('detection_type', 'unknown'),
                    'method': item.get('method', 'unknown'),
                    'score': item.get('score', 0),
                    'sentence': item.get('sentence', ''),
                    'sentiment': sentiment['sentiment'],
                    'sentiment_score': sentiment['score'],
                    'extracted_entities': entities
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
        
        # Guardar resultados cada cierto número de videos procesados
        if processed_count % save_interval == 0 and new_results:
            print(f"Guardando resultados parciales después de procesar {processed_count} videos...")
            results_df = save_results(results_df, new_results, output_path)
            new_results = []  # Limpiar lista después de guardar
    
    # Guardar resultados finales si quedan pendientes
    if new_results:
        print("Guardando resultados finales...")
        results_df = save_results(results_df, new_results, output_path)
    
    return results_df


# FUNCIÓN PRINCIPAL DE EJECUCIÓN
def main():
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
        nlp_en,
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
    else:
        print("No se detectaron menciones.")
    
    print("Procesamiento completo.")


# Ejecutar el programa
if __name__ == "__main__":
    main()