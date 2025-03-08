import pandas as pd
import numpy as np
import os
import re
import json
import ast
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import random
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas corregidas
base_dir = r"C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\nlp\products and brands detection"
sentences_path = os.path.join(base_dir, "sentences_transcriptions.xlsx")
brands_products_path = os.path.join(base_dir, "all_brands_products.json")
brands_products_regex_path = os.path.join(base_dir, "all_brands_products_regex.json")
output_path = os.path.join(base_dir, "brand_product_mentions_bert_improved.xlsx")

print(f"Buscando archivo JSON en: {brands_products_path}")

# Descargar recursos necesarios para el análisis de sentimiento
try:
    nltk.download('vader_lexicon', quiet=True)
    print("Recursos NLTK descargados correctamente")
except Exception as e:
    print(f"Error descargando recursos NLTK: {e}, verificar conexión a internet")

# Función para verificar si GPU está disponible
def check_gpu():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("GPU no disponible. Usando CPU.")
        return device
    except Exception as e:
        print(f"Error al verificar GPU: {e}. Usando CPU.")
        return torch.device("cpu")

# Cargar datos
print("Cargando datos...")
try:
    sentences_df = pd.read_excel(sentences_path)
    print(f"Cargadas {len(sentences_df)} frases")
except Exception as e:
    print(f"Error al cargar el archivo de frases: {e}")
    # Crear un DataFrame vacío si hay error
    sentences_df = pd.DataFrame(columns=['video_url', 'sentence'])

# Procesamos todas las frases del archivo
print(f"Procesando todas las {len(sentences_df)} frases del archivo completo...")
# No se realiza ningún muestreo para procesar el dataset completo

# Cargar JSON con marcas y productos
try:
    with open(brands_products_path, 'r', encoding='utf-8') as f:
        brands_products = json.load(f)
    print(f"Cargadas {len(brands_products)} marcas y sus productos")
except Exception as e:
    print(f"Error al cargar el archivo JSON de marcas y productos: {e}")
    brands_products = {}  # Diccionario vacío si hay error

# Intentar cargar el JSON de expresiones regulares para marcas
try:
    with open(brands_products_regex_path, 'r', encoding='utf-8') as f:
        brands_regex = json.load(f)
    print(f"Cargados patrones regex para {len(brands_regex)} marcas")
    has_regex_patterns = True
except Exception as e:
    print(f"No se pudieron cargar los patrones regex: {e}")
    brands_regex = {}
    has_regex_patterns = False

# Lista de términos genéricos relacionados con cosmética que pueden generar falsos positivos
generic_makeup_terms = [
    "makeup", "make up", "make-up", "cosmetics", "foundation", "concealer", "powder", 
    "lipstick", "lip gloss", "eyeliner", "mascara", "eyeshadow", "blush", "bronzer", 
    "highlighter", "contour", "brow", "lashes", "face", "beauty", "cosmetic", "skin"
]

# Marcas adicionales que podrían no estar en el JSON original
additional_brands = [
    "Huda Beauty", "Rare Beauty", "Fenty Beauty", "Pat McGrath", "Natasha Denona",
    "Charlotte Tilbury", "NARS", "MAC Cosmetics", "Dior Beauty", "Giorgio Armani Beauty",
    "Gucci Beauty", "Lancôme", "YSL Beauty", "Estée Lauder", "Tom Ford Beauty",
    "Chanel Beauty", "Bobbi Brown", "Urban Decay", "Too Faced", "Tarte",
    "Benefit Cosmetics", "Hourglass", "Anastasia Beverly Hills", "Milk Makeup", "Glossier",
    "Kaja", "Kosas", "Ilia", "Saie", "Tower 28"
]

# Añadir marcas adicionales si no están ya
for brand in additional_brands:
    if brand not in brands_products:
        brands_products[brand] = []

# Marcas con nombres alternativos (versiones cortas o alias)
brand_aliases = {
    "Rare Beauty by Selena Gomez": ["Rare Beauty", "Selena Gomez Beauty", "Rare"],
    "Charlotte Tilbury": ["Charlotte", "Tilbury"],
    "Patrick Ta Beauty": ["Patrick Ta"],
    "Pat McGrath Labs": ["Pat McGrath", "Pat", "PMG"],
    "Tom Ford Beauty": ["Tom Ford"],
    "Anastasia Beverly Hills": ["ABH", "Anastasia"],
    "Kosas": ["Kosas Cosmetics"],
    "Bobbi Brown": ["Bobbi"],
    "Natasha Denona": ["Natasha", "ND"],
    "Sol de Janeiro": ["Sol", "SDJ"],
    "Drunk Elephant": ["Drunk E", "DE"],
    "Kiehl's": ["Kiehls"],
    "Hourglass": ["Hourglass Cosmetics"],
    "Olaplex": ["Olaplex Hair"],
    "Ouai": ["Ouai Hair"],
    "Make Up By Mario": ["Mario", "MUBM"],
    "Make Up For Ever": ["MUFE"],
    "Huda Beauty": ["Huda"],
    "Fenty Beauty": ["Fenty", "Rihanna Beauty"],
    "MAC Cosmetics": ["MAC"],
    "Urban Decay": ["UD"],
    "Too Faced": ["TF"],
    "Dior Beauty": ["Dior"],
    "Chanel Beauty": ["Chanel"],
    "Gucci Beauty": ["Gucci"],
    "Lancôme": ["Lancome"],
    "Estée Lauder": ["Estee Lauder"],
    "YSL Beauty": ["YSL", "Yves Saint Laurent"],
    "Benefit Cosmetics": ["Benefit"],
    "Giorgio Armani Beauty": ["Armani Beauty"],
    "NARS": ["NARS Cosmetics"],
    "Tarte": ["Tarte Cosmetics"]
}

# Crear mapeo inverso de alias a marca principal
alias_to_brand = {}
for brand, aliases in brand_aliases.items():
    for alias in aliases:
        alias_to_brand[alias.lower()] = brand

# Lista de marcas con palabras comunes que necesitan tratamiento especial
special_brands = {
    "Then I Met You": {
        "regex": r'\bthen\s+i\s+met\s+you\b',
        "similarity_threshold": 0.90  # Umbral muy alto para evitar falsos positivos
    },
    "The Ordinary": {
        "regex": r'\bthe\s+ordinary\b',
        "similarity_threshold": 0.88
    },
    "It Cosmetics": {
        "regex": r'\bit\s+cosmetics\b',
        "similarity_threshold": 0.88
    },
    "Make Up By Mario": {
        "regex": r'\bmake\s+up\s+by\s+mario\b',
        "similarity_threshold": 0.85
    },
    "Make Up For Ever": {
        "regex": r'\bmake\s+up\s+for\s+ever\b',
        "similarity_threshold": 0.85
    },
    "Kay Skin": {
        "regex": r'\bkay\s+skin\b',
        "similarity_threshold": 0.88,
        "avoid_single_word": "skin"  # Evitar detecciones cuando solo aparece esta palabra
    }
}

# Compilar los patrones regex
for brand in special_brands:
    special_brands[brand]["compiled_regex"] = re.compile(special_brands[brand]["regex"], re.IGNORECASE)

# Clasificar marcas para manejo especial
problematic_brands = []
normal_brands = []
very_common_words_brands = []

for brand in brands_products.keys():
    brand_lower = brand.lower()
    
    # Las marcas especiales se manejan de forma independiente
    if brand in special_brands:
        very_common_words_brands.append(brand)
    # Identificar marcas problemáticas que contienen términos genéricos
    elif any(term in brand_lower for term in generic_makeup_terms):
        problematic_brands.append(brand)
    else:
        normal_brands.append(brand)

print(f"Identificadas {len(problematic_brands)} marcas problemáticas y {len(very_common_words_brands)} marcas con palabras muy comunes de un total de {len(brands_products)} marcas")

# Usar todas las marcas disponibles en el archivo
print(f"Usando todas las {len(brands_products)} marcas disponibles")
filtered_brands_products = brands_products.copy()

# Definir un máximo razonable de productos por marca para evitar problemas de memoria
MAX_PRODUCTS_PER_BRAND = 15  # Incrementado para mejorar detección de productos
for brand in filtered_brands_products:
    if len(filtered_brands_products[brand]) > MAX_PRODUCTS_PER_BRAND:
        filtered_brands_products[brand] = filtered_brands_products[brand][:MAX_PRODUCTS_PER_BRAND]

print(f"Usando {len(filtered_brands_products)} marcas con máximo {MAX_PRODUCTS_PER_BRAND} productos cada una")

# Limpiar y normalizar datos
print("Limpiando y normalizando datos...")
cleaned_brands_products = {}
for brand, products in filtered_brands_products.items():
    if not isinstance(brand, str):
        brand = str(brand)
    
    # No convertimos Rare Beauty a "Rare Beauty by Selena Gomez" - lo dejamos como está
    # pero sí hacemos el resto de limpieza
    
    # Limpiar productos
    cleaned_products = []
    for product in products:
        if product is not None:
            if not isinstance(product, str):
                product = str(product)
            cleaned_products.append(product)
    
    cleaned_brands_products[brand] = cleaned_products

brands_products = cleaned_brands_products

# Cargar modelo BERT
print("Cargando modelo BERT optimizado...")
device = check_gpu()

try:
    # Usar un modelo más pequeño y eficiente
    model_name = "distilbert-base-uncased"  # Modelo más ligero que el multilingüe
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print(f"Modelo {model_name} cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo BERT: {e}")
    print("Terminando ejecución debido a error crítico")
    exit(1)

# Función de embedding por lotes mejorada con manejo de errores
def get_bert_embeddings_batch(texts, tokenizer, model, device, batch_size=4):  # Batch size moderado
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        try:
            batch_texts = texts[i:i+batch_size]
            
            # Tokenizar textos en lote
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                     max_length=128, return_tensors='pt').to(device)
            
            # Obtener embeddings
            with torch.no_grad():
                output = model(**encoded_input)
            
            # Extraer embedding del token [CLS] para cada texto del lote
            batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error en lote {i}-{i+batch_size}: {e}")
            # Procesar uno por uno si falla el lote
            for j in range(i, min(i+batch_size, len(texts))):
                try:
                    single_text = [texts[j]]
                    encoded_input = tokenizer(single_text, padding=True, truncation=True, 
                                             max_length=128, return_tensors='pt').to(device)
                    with torch.no_grad():
                        output = model(**encoded_input)
                    single_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(single_embedding[0])
                except Exception as e2:
                    # Si falla un texto individual, agregar un embedding con ceros
                    print(f"Error en texto {j}, usando embedding vacío: {e2}")
                    embeddings.append(np.zeros(768))  # Dimensión estándar para DistilBERT
    
    return embeddings

# Inicializar analizador de sentimiento
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Error al inicializar el analizador de sentimiento: {e}")
    # Crear una función de respaldo simple si falla
    def get_sentiment(text):
        return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
else:
    # Función para obtener puntuación de sentimiento
    def get_sentiment(text):
        if not isinstance(text, str):
            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
        try:
            return sia.polarity_scores(text)
        except:
            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}

# MEJORA: Filtro avanzado basado en expresiones regulares para marcas problemáticas
def create_brand_regex_patterns():
    """Crea patrones regex más estrictos para marcas problemáticas"""
    patterns = {}
    
    # Si tenemos el archivo JSON de regex, usamos esos patrones
    if has_regex_patterns and brands_regex:
        for brand, pattern in brands_regex.items():
            if brand in brands_products:
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    patterns[brand] = compiled_pattern
                except:
                    # Si el patrón no es válido, crear uno simple
                    pattern = r'\b' + re.escape(brand.lower()) + r'\b'
                    patterns[brand] = re.compile(pattern, re.IGNORECASE)
    
    # Crear patrones para las marcas que no tienen un patrón en el JSON
    for brand in problematic_brands:
        if brand not in patterns:
            brand_lower = brand.lower()
            # Patrón que requiere palabras completas y respeta límites de palabras
            pattern = r'\b' + re.escape(brand_lower) + r'\b'
            patterns[brand] = re.compile(pattern, re.IGNORECASE)
    
    return patterns

# Crear patrones regex para marcas problemáticas
brand_regex_patterns = create_brand_regex_patterns()

# Verificar si un alias de marca está presente en la frase
def check_brand_aliases(sentence, brand_aliases):
    """Verifica si algún alias de marca está presente en la frase"""
    sentence_lower = sentence.lower()
    found_aliases = {}
    
    # Buscar aliases completos (frases exactas)
    for alias, main_brand in alias_to_brand.items():
        alias_pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(alias_pattern, sentence_lower, re.IGNORECASE):
            found_aliases[main_brand] = alias
    
    return found_aliases

# MEJORA: Filtro avanzado que maneja mejor términos genéricos y marcas problemáticas
def advanced_rule_filter(sentence, normal_brands, problematic_brands, very_common_words_brands, patterns, special_brands, brand_aliases):
    """Filtro avanzado para preseleccionar marcas potenciales con menor tasa de falsos positivos"""
    potential_brands = []
    
    if not isinstance(sentence, str):
        return potential_brands
    
    sentence_lower = sentence.lower()
    
    # Buscar aliases de marcas
    found_aliases = check_brand_aliases(sentence, brand_aliases)
    for main_brand in found_aliases:
        if main_brand not in potential_brands:
            potential_brands.append(main_brand)
    
    # Primero procesar marcas con palabras muy comunes (requieren regex exacto)
    for brand in very_common_words_brands:
        if brand in special_brands:
            # Verificar si tiene una palabra a evitar y si esa palabra aparece sola
            avoid_word = special_brands[brand].get("avoid_single_word", None)
            if avoid_word and avoid_word in sentence_lower:
                # Verificar si la palabra está sola (sin el resto de la marca)
                # por ejemplo, si solo aparece "skin" sin "kay"
                if brand.lower() not in sentence_lower and re.search(r'\b' + re.escape(avoid_word) + r'\b', sentence_lower):
                    continue  # Saltar esta marca si solo aparece la palabra genérica
            
            # Buscar coincidencia exacta usando el patrón regex específico
            if special_brands[brand]["compiled_regex"].search(sentence_lower):
                potential_brands.append(brand)
    
    # Luego buscar marcas normales con método simple
    for brand in normal_brands:
        brand_lower = brand.lower()
        
        # Método 1: Buscar la marca completa
        if brand_lower in sentence_lower:
            # Verificación de límites de palabras para reducir falsos positivos
            if re.search(r'\b' + re.escape(brand_lower) + r'\b', sentence_lower):
                potential_brands.append(brand)
                continue
            
        # Método 2: Verificar palabras clave significativas
        key_words = [word for word in brand_lower.split() if len(word) > 3]
        
        # Si la marca no tiene palabras clave significativas, usar la marca completa
        if not key_words and len(brand_lower) > 2:
            key_words = [brand_lower]
        
        # Verificar si alguna palabra clave aparece en la frase
        match_count = 0
        for word in key_words:
            if re.search(r'\b' + re.escape(word) + r'\b', sentence_lower):
                match_count += 1
        
        # Si hay suficientes coincidencias de palabras clave, considerar la marca
        if match_count >= min(2, len(key_words)):
            potential_brands.append(brand)
    
    # Luego buscar marcas problemáticas usando patrones regex más estrictos
    for brand in problematic_brands:
        if brand in patterns and patterns[brand].search(sentence_lower):
            # Verificación adicional para marcas que contienen "make up" o similares
            brand_lower = brand.lower()
            
            # Para marcas con "makeup", verificar que no sea solo la palabra genérica
            if "makeup" in brand_lower or "make up" in brand_lower or "make-up" in brand_lower:
                # Buscar el nombre exacto de la marca con límites de palabras
                if patterns[brand].search(sentence_lower):
                    # Verificar que no es solo la palabra genérica
                    if len(brand_lower.split()) > 1:  # Si la marca tiene más de una palabra
                        potential_brands.append(brand)
            else:
                potential_brands.append(brand)
    
    return potential_brands

# Implementar guardado de progreso
def save_progress(results, filename="resultados_parciales.xlsx"):
    """Guarda los resultados parciales en caso de error"""
    full_path = os.path.join(base_dir, filename)
    try:
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.to_excel(full_path, index=False)
            print(f"Progreso guardado en: {full_path} ({len(results)} registros)")
        else:
            print("No hay resultados para guardar todavía")
    except Exception as e:
        print(f"Error al guardar progreso: {e}")
        # Intentar guardar como CSV si Excel falla
        try:
            csv_path = full_path.replace('.xlsx', '.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Guardado alternativo como CSV en: {csv_path}")
        except Exception as e2:
            print(f"También falló el guardado como CSV: {e2}")

# Generar embeddings de marcas y aliases
print("Generando embeddings de marcas...")
brand_names = list(brands_products.keys())

# Incluir aliases en el conjunto de embeddings
brand_alias_texts = []
brand_alias_mappings = []

for brand in brand_names:
    if brand in brand_aliases:
        for alias in brand_aliases[brand]:
            brand_alias_texts.append(alias)
            brand_alias_mappings.append(brand)

print(f"Añadiendo {len(brand_alias_texts)} aliases de marcas para embeddings")

try:
    # Generar embeddings para las marcas principales
    brand_embeddings_list = get_bert_embeddings_batch(brand_names, tokenizer, model, device)
    brand_embeddings = {brand: embedding for brand, embedding in zip(brand_names, brand_embeddings_list)}
    
    # Generar embeddings para los aliases
    if brand_alias_texts:
        alias_embeddings_list = get_bert_embeddings_batch(brand_alias_texts, tokenizer, model, device)
        
        # Mapear los embeddings de alias a las marcas principales
        for alias, embedding, main_brand in zip(brand_alias_texts, alias_embeddings_list, brand_alias_mappings):
            # Guardar el embedding del alias con un formato especial para identificarlos
            brand_embeddings[f"ALIAS:{alias}:{main_brand}"] = embedding
    
    print(f"Embeddings generados para {len(brand_embeddings)} marcas y aliases")
except Exception as e:
    print(f"Error al generar embeddings de marcas: {e}")
    brand_embeddings = {}
    # Terminamos la ejecución ya que esto es crítico
    print("Terminando ejecución debido a error crítico")
    exit(1)

# Generar embeddings de productos
print("Generando embeddings de productos...")
all_products = []
product_to_brand = {}

# Recopilar todos los productos y su marca asociada
for brand, products in brands_products.items():
    for product in products:
        if product and isinstance(product, str) and len(product) > 2:
            all_products.append(product)
            product_to_brand[product] = brand

print(f"Generando embeddings para {len(all_products)} productos...")

try:
    product_embeddings_list = get_bert_embeddings_batch(all_products, tokenizer, model, device)
    product_embeddings = {product: embedding for product, embedding in zip(all_products, product_embeddings_list)}
    print(f"Embeddings generados para {len(product_embeddings)} productos")
except Exception as e:
    print(f"Error al generar embeddings de productos: {e}")
    product_embeddings = {}
    print("Advertencia: La detección de productos puede ser limitada")

# AJUSTE: Umbrales de similitud por categoría de marca
NORMAL_BRAND_SIMILARITY_THRESHOLD = 0.70  # Reducido para capturar más marcas
PROBLEMATIC_BRAND_SIMILARITY_THRESHOLD = 0.80  # Umbral más alto para marcas problemáticas
ALIAS_SIMILARITY_THRESHOLD = 0.75  # Umbral para aliases de marcas
PRODUCT_SIMILARITY_THRESHOLD = 0.65  # Reducido para capturar más productos

# Lista para almacenar resultados
results = []

# Contador para guardado periódico
save_counter = 0
SAVE_FREQUENCY = 100  # Guardar cada 100 frases procesadas para reducir operaciones de disco

# Función para determinar si una detección de marca es válida
def is_valid_brand_mention(brand, sentence, similarity):
    sentence_lower = sentence.lower()
    brand_lower = brand.lower()
    
    # Si es un alias (formato especial), extraer la información y validar
    if brand.startswith("ALIAS:"):
        parts = brand.split(":")
        if len(parts) == 3:
            alias = parts[1]
            main_brand = parts[2]
            
            # Verificar si el alias aparece como palabra exacta
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', sentence_lower):
                # Buscar contexto de belleza o maquillaje
                beauty_context = ["makeup", "beauty", "cosmetic", "skincare", "product", 
                                 "foundation", "concealer", "lipstick", "eyeshadow"]
                if any(word in sentence_lower for word in beauty_context):
                    return True, main_brand  # Devolver la marca principal, no el alias
        
        return False, None
    
    # Para marcas con palabras muy comunes, aplicar verificación estricta
    if brand in special_brands:
        # 1. Verificar si supera el umbral de similitud específico para esta marca
        if similarity < special_brands[brand]["similarity_threshold"]:
            return False, None
            
        # 2. Verificar que el patrón exacto aparece en la frase
        if not special_brands[brand]["compiled_regex"].search(sentence_lower):
            return False, None
        
        # 3. Para "Kay Skin", verificación adicional para evitar falsas coincidencias con "skin"
        if brand == "Kay Skin" and "kay" not in sentence_lower and "skin" in sentence_lower:
            return False, None
            
        # Si es "Then I Met You", verificaciones adicionales
        if brand == "Then I Met You":
            # Verificar que no es solo parte de una frase común como "and then I..."
            parts = re.split(r'\s+', sentence_lower)
            for i, word in enumerate(parts):
                if word == "then" and i + 3 < len(parts):
                    if parts[i+1] == "i" and parts[i+2] == "met" and parts[i+3] == "you":
                        # Verificar que no es parte de una frase más larga
                        if i > 0 and parts[i-1] in ["and", "but", "so", "when"]:
                            return False, None
    
    # Si la marca es "It Cosmetics", verificación especial
    elif brand == "It Cosmetics":
        # Asegurarse que "it" es parte del nombre de la marca y no simplemente el pronombre
        if re.search(r'\bit\s+cosmetics\b', sentence_lower):
            # Buscar contexto de maquillaje
            context_words = ["makeup", "foundation", "brush", "concealer", "powder", "brand"]
            if any(word in sentence_lower for word in context_words):
                return True, brand
            else:
                return False, None
    
    # Para "The Ordinary", verificación especial    
    elif brand == "The Ordinary":
        # Verificar que no es solo "the ordinary" como frase común
        if re.search(r'\bthe\s+ordinary\b', sentence_lower):
            # Buscar contexto de skincare
            context_words = ["serum", "niacinamide", "acid", "skincare", "deciem", "product"]
            if any(word in sentence_lower for word in context_words):
                return True, brand
            else:
                return False, None
    
    # Para Rare Beauty, verificación especial si hay versión corta
    elif brand == "Rare Beauty by Selena Gomez" and "rare beauty" in sentence_lower:
        return True, brand
    
    # Para Huda Beauty, verificación especial
    elif brand == "Huda Beauty" and "huda" in sentence_lower:
        return True, brand
        
    # Para Charlotte Tilbury, verificación si solo mencionan "Charlotte"
    elif brand == "Charlotte Tilbury" and re.search(r'\bcharlotte\b', sentence_lower):
        return True, brand
    
    return True, brand

# Función para procesar productos más eficientemente
def detect_products(sentence, sentence_embedding, found_brands, product_embeddings, product_to_brand):
    """Detectar productos mencionados en la frase usando embeddings precomputados"""
    found_products = []
    
    # Limitar la búsqueda a productos de las marcas encontradas + algunos más
    candidate_products = []
    
    # Primero añadir productos de las marcas encontradas
    for brand in found_brands:
        if brand in brands_products:
            for product in brands_products[brand]:
                if product in product_embeddings:
                    candidate_products.append(product)
    
    # Añadir palabras comunes de productos que aparecen en la frase
    product_keywords = ["foundation", "concealer", "lipstick", "palette", "mascara", 
                         "serum", "cream", "moisturizer", "sunscreen", "cleanser"]
    
    sentence_lower = sentence.lower()
    for keyword in product_keywords:
        if keyword in sentence_lower:
            # Buscar productos que contengan esta palabra clave
            for product in product_embeddings:
                if keyword in product.lower() and product not in candidate_products:
                    candidate_products.append(product)
    
    # Limitar la cantidad de productos candidatos para mejorar rendimiento
    if len(candidate_products) > 25:
        candidate_products = random.sample(candidate_products, 25)
    
    # Calcular similitud para cada producto candidato
    for product in candidate_products:
        product_embedding = product_embeddings[product]
        try:
            similarity = cosine_similarity([sentence_embedding], [product_embedding])[0][0]
            
            # Verificar si el producto supera el umbral de similitud
            if similarity >= PRODUCT_SIMILARITY_THRESHOLD:
                # Verificación adicional: comprobar si alguna palabra del producto está en la frase
                product_words = [word for word in product.lower().split() if len(word) > 3]
                if product_words:
                    for word in product_words:
                        if word in sentence_lower:
                            brand = product_to_brand.get(product, "Unknown")
                            found_products.append({
                                'product': product,
                                'brand': brand,
                                'similarity': similarity
                            })
                            break
                else:
                    # Si el producto no tiene palabras significativas, confiar solo en la similitud
                    brand = product_to_brand.get(product, "Unknown")
                    found_products.append({
                        'product': product,
                        'brand': brand,
                        'similarity': similarity
                    })
        except Exception as e:
            # Ignorar errores en productos individuales
            continue
    
    return found_products

# Detectar marcas y productos usando BERT con optimizaciones
print("Detectando marcas y productos con BERT optimizado...")
try:
    for i, row in tqdm(sentences_df.iterrows(), total=len(sentences_df)):
        try:
            save_counter += 1
            if save_counter >= SAVE_FREQUENCY:
                save_progress(results, "resultados_parciales.xlsx")
                save_counter = 0
                
            video_url = row['video_url'] if 'video_url' in row else "unknown"
            sentence = row['sentence'] if 'sentence' in row else ""
            
            if not isinstance(sentence, str) or pd.isna(sentence) or sentence == "":
                continue
            
            # MEJORA: Usar filtro avanzado para reducir falsos positivos
            potential_brands = advanced_rule_filter(
                sentence, normal_brands, problematic_brands, 
                very_common_words_brands, brand_regex_patterns, 
                special_brands, brand_aliases
            )
            
            # Para "Huda Beauty" y otras marcas que podrían faltar,
            # verificar directamente si aparecen términos específicos
            sentence_lower = sentence.lower()
            if "huda" in sentence_lower and "Huda Beauty" not in potential_brands:
                potential_brands.append("Huda Beauty")
                
            if "rare beauty" in sentence_lower and "Rare Beauty by Selena Gomez" not in potential_brands:
                potential_brands.append("Rare Beauty by Selena Gomez")
            
            # Si no hay marcas potenciales, continuar con la siguiente frase
            if not potential_brands:
                continue
            
            # Obtener embedding de la frase
            try:
                sentence_embedding = get_bert_embeddings_batch([sentence], tokenizer, model, device)[0]
            except Exception as e:
                print(f"Error al generar embedding para frase: {e}")
                continue
            
            # Detectar marcas con umbrales adaptados
            found_brands = []
            
            # Primero comprobar con las marcas potenciales del filtro previo
            for brand in potential_brands:
                if brand not in brand_embeddings:
                    continue
                    
                brand_embedding = brand_embeddings[brand]
                try:
                    similarity = cosine_similarity([sentence_embedding], [brand_embedding])[0][0]
                except:
                    continue
                
                # Elegir umbral según categoría de marca
                if brand in very_common_words_brands:
                    threshold = special_brands[brand]["similarity_threshold"]
                elif brand in problematic_brands:
                    threshold = PROBLEMATIC_BRAND_SIMILARITY_THRESHOLD
                else:
                    threshold = NORMAL_BRAND_SIMILARITY_THRESHOLD
                
                # Verificar si supera el umbral
                if similarity >= threshold:
                    # Validación adicional para reducir falsos positivos
                    is_valid, actual_brand = is_valid_brand_mention(brand, sentence, similarity)
                    if is_valid and actual_brand and actual_brand not in found_brands:
                        found_brands.append(actual_brand)
            
            # Luego comprobar con los aliases
            for key, embedding in brand_embeddings.items():
                if key.startswith("ALIAS:"):
                    try:
                        similarity = cosine_similarity([sentence_embedding], [embedding])[0][0]
                        if similarity >= ALIAS_SIMILARITY_THRESHOLD:
                            is_valid, actual_brand = is_valid_brand_mention(key, sentence, similarity)
                            if is_valid and actual_brand and actual_brand not in found_brands:
                                found_brands.append(actual_brand)
                    except:
                        continue
            
            # Si no se encontraron marcas, continuar con la siguiente frase
            if not found_brands:
                continue
            
            # Procesar productos de manera más eficiente
            # Usar los embeddings precomputados en lugar de generarlos por cada frase
            found_products = detect_products(
                sentence, 
                sentence_embedding, 
                found_brands,
                product_embeddings,
                product_to_brand
            )
            
            # Análisis de sentimiento
            sentiment = get_sentiment(sentence)
            
            # Añadir a resultados si se encontraron marcas o productos
            if found_brands or found_products:
                results.append({
                    'video_url': video_url,
                    'sentence': sentence,
                    'mentioned_brands': found_brands,
                    'mentioned_products': [p['product'] for p in found_products],
                    'product_brands': [p['brand'] for p in found_products],
                    'product_similarities': [p['similarity'] for p in found_products],
                    'sentiment_negative': sentiment['neg'],
                    'sentiment_neutral': sentiment['neu'],
                    'sentiment_positive': sentiment['pos'],
                    'sentiment_compound': sentiment['compound'],
                    'word_count': len(sentence.split())
                })
        except Exception as e:
            print(f"Error procesando frase: {e}")
            continue
except Exception as e:
    print(f"Error general en el procesamiento: {e}")
finally:
    # Guardar los resultados obtenidos hasta el momento
    save_progress(results, "resultados_finales.xlsx")

# Crear DataFrame con resultados
results_df = pd.DataFrame(results)

# Añadir categoría de sentimiento
if not results_df.empty:
    results_df['sentiment_category'] = results_df['sentiment_compound'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )

# Generar estadísticas y métricas
print("\n===== Estadísticas de detección con BERT =====")
if not results_df.empty and len(results_df) > 0:
    # Estadísticas básicas
    num_sentences_with_detections = len(results_df)
    percent_with_detections = (num_sentences_with_detections / len(sentences_df)) * 100
    num_with_brands = sum(results_df['mentioned_brands'].apply(len) > 0)
    num_with_products = sum(results_df['mentioned_products'].apply(len) > 0)
    
    print(f"Total de frases analizadas: {len(sentences_df)}")
    print(f"Frases con detecciones: {num_sentences_with_detections} ({percent_with_detections:.2f}%)")
    print(f"Frases con marcas: {num_with_brands}")
    print(f"Frases con productos: {num_with_products}")
    
    # Generar resúmenes de marcas y productos
    # Contar menciones de marcas
    brand_counts = defaultdict(int)
    for brands in results_df['mentioned_brands']:
        for brand in brands:
            brand_counts[brand] += 1
    
    brand_counts_df = pd.DataFrame({
        'Brand': list(brand_counts.keys()),
        'Mentions': list(brand_counts.values())
    }).sort_values('Mentions', ascending=False)
    
    # Contar menciones de productos
    product_counts = defaultdict(int)
    for i, row in results_df.iterrows():
        for j, product in enumerate(row['mentioned_products']):
            if j < len(row['product_brands']):
                product_brand = f"{product} ({row['product_brands'][j]})"
                product_counts[product_brand] += 1
    
    product_counts_df = pd.DataFrame({
        'Product (Brand)': list(product_counts.keys()),
        'Mentions': list(product_counts.values())
    }).sort_values('Mentions', ascending=False)
    
    # Mostrar top marcas y productos
    print("\nTop 10 marcas más mencionadas:")
    print(brand_counts_df.head(10).to_string(index=False))
    
    print("\nTop 10 productos más mencionados:")
    print(product_counts_df.head(10).to_string(index=False))
    
    # Calcular sentimiento promedio por marca
    brand_sentiment = defaultdict(list)
    for _, row in results_df.iterrows():
        for brand in row['mentioned_brands']:
            brand_sentiment[brand].append(row['sentiment_compound'])
    
    brand_sentiment_df = pd.DataFrame({
        'Brand': list(brand_sentiment.keys()),
        'Avg_Sentiment': [np.mean(scores) for scores in brand_sentiment.values()],
        'Mentions': [len(scores) for scores in brand_sentiment.values()]
    }).sort_values('Avg_Sentiment', ascending=False)
    
    # Guardar resultados en Excel
    print(f"\nGuardando resultados en: {output_path}")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Detecciones', index=False)
            brand_counts_df.to_excel(writer, sheet_name='Resumen_Marcas', index=False)
            product_counts_df.to_excel(writer, sheet_name='Resumen_Productos', index=False)
            brand_sentiment_df.to_excel(writer, sheet_name='Sentimiento_por_Marca', index=False)
            
            # Añadir hoja de configuración y metadatos
            config_df = pd.DataFrame({
                'Parameter': ['Total Brands Analyzed', 'Normal Brands', 'Problematic Brands', 
                             'Special Treatment Brands', 'Aliases de Marcas', 'Total Products',
                             'Max Products Per Brand', 'Normal Brand Threshold', 
                             'Problematic Brand Threshold', 'Alias Threshold', 
                             'Product Threshold', 'BERT Model Used'],
                'Value': [len(filtered_brands_products), len(normal_brands), len(problematic_brands),
                         len(very_common_words_brands), len(brand_alias_texts), len(product_embeddings),
                         MAX_PRODUCTS_PER_BRAND, NORMAL_BRAND_SIMILARITY_THRESHOLD, 
                         PROBLEMATIC_BRAND_SIMILARITY_THRESHOLD, ALIAS_SIMILARITY_THRESHOLD, 
                         PRODUCT_SIMILARITY_THRESHOLD, model_name]
            })
            config_df.to_excel(writer, sheet_name='Configuración', index=False)
        print("Archivo Excel guardado correctamente.")
    except Exception as e:
        print(f"Error al guardar el archivo Excel final: {e}")
        # Intentar guardar en formato CSV como alternativa
        try:
            results_df.to_csv(output_path.replace('.xlsx', '.csv'), index=False)
            print(f"Se guardó una versión alternativa en CSV: {output_path.replace('.xlsx', '.csv')}")
        except:
            print("No se pudo guardar ni en Excel ni en CSV.")
else:
    print("No se encontraron detecciones.")

print("\nProceso completado.")