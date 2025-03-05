# cosmetic_patterns.py
import os
import re
import pandas as pd
import numpy as np
import json
import time
import spacy
from spacy.tokens import DocBin
from spacy.lang.es import Spanish
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# DIRECTORY SETUP
base_dir = 'C:/Users/sandr/Documents/scrp_tiktok_tfg'
data_dir = os.path.join(base_dir, 'data')
clean_data_dir = os.path.join(data_dir, 'clean_data')
analysis_dir = os.path.join(base_dir, 'analysis/nlp')

# FILE PATHS
products_file = os.path.join(analysis_dir, 'product_mentions_analysis.xlsx')
output_file = os.path.join(analysis_dir, 'cosmetic_attributes_analysis.xlsx')

# Verificar que el archivo de menciones de productos existe
if not os.path.exists(products_file):
    raise FileNotFoundError(f"Archivo de menciones de productos no encontrado: {products_file}")

# DEFINIR ATRIBUTOS COSMÉTICOS A DETECTAR
COSMETIC_ATTRIBUTES = {
    'TEXTURE': ['matte', 'satin', 'shiny', 'creamy', 'lightweight', 'silky', 'soft',
                'velvety', 'watery', 'gel', 'liquid', 'foamy', 'powder', 'compact', 
                'mousse', 'balm', 'emulsion', 'fluid', 'jelly', 'serum'],
    
    'COVERAGE': ['full coverage', 'medium coverage', 'light coverage', 'high coverage', 
                 'low coverage', 'translucent', 'opaque', 'buildable', 'sheer', 'modular'],
    
    'LONGEVITY': ['long-lasting', 'waterproof', 'water-resistant', 'smudge-proof', 
                  'transfer-proof', 'no transfer', 'fade-resistant', '24h', '12h', '8h', 
                  'permanent', 'all-day wear'],
    
    'HYDRATION': ['hydrating', 'moisturizing', 'nourishing', 'moisture boost', 'refreshing', 
                  'soothing', 'calming', 'rehydrating', 'replenishing', 'repairing', 'dewy'],
    
    'INGREDIENTS': ['hyaluronic acid', 'retinol', 'vitamin C', 'vitamin E', 'aloe vera', 
                    'argan oil', 'niacinamide', 'collagen', 'elastin', 'ceramides', 'peptides', 
                    'AHA', 'BHA', 'glycolic acid', 'salicylic acid', 'SPF', 'sun protection', 
                    'antioxidants', 'oil-free', 'paraben-free', 'vegan', 'cruelty-free', 
                    'fragrance-free', 'hypoallergenic', 'dermatologist-tested'],
    
    'EFFECT': ['illuminating', 'glow', 'radiant', 'dewy', 'natural', 'lifting effect', 
               'anti-aging', 'anti-wrinkle', 'volumizing', 'lengthening', 'curling', 'plumping', 
               'definition', 'contouring', 'bronzing', 'bronzer', 'highlighting', 'mattifying', 
               'anti-shine', 'correcting', 'color-correcting', 'neutralizing', 'tone-evening', 
               'soothing', 'refreshing', 'antioxidant', 'firming', 'smoothing', 'anti-blemish', 
               'brightening', 'blurring', 'poreless', 'skin-smoothing'],
    
    'FORMAT': ['stick', 'bar', 'tube', 'brush', 'applicator', 'spatula', 'roll-on', 'spray',
               'serum', 'mask', 'ampoules', 'cream', 'gel-cream', 'mousse', 'drops',
               'capsules', 'powder', 'liquid', 'oil', 'cushion', 'compact', 'pad', 'pen']
}


# MODELOS NLP
def load_nlp_models():
    """
    Carga modelos NLP para análisis de texto
    """
    print("Cargando modelos NLP...")
    
    # Cargar modelo spaCy para inglés
    try:
        nlp_en = spacy.load("en_core_web_md")
    except:
        print("Downloading spaCy English model...")
        spacy.cli.download("en_core_web_md")
        nlp_en = spacy.load("en_core_web_md")
    
    # Cargar modelo NER optimizado para inglés de Hugging Face
    try:
        # Intentar cargar modelo NER específico para inglés
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    except:
        print("Fallback to alternative NER model...")
        # En caso de error, usar un modelo alternativo
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased") 
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    return nlp_en, ner_pipeline

def load_product_mentions():
    """
    Carga el archivo de menciones de productos previamente generado
    
    Returns:
        pd.DataFrame: DataFrame con las menciones de productos
    """
    print(f"Cargando menciones de productos desde {products_file}...")
    
    try:
        # Intentar cargar desde Excel
        mentions_df = pd.read_excel(products_file, sheet_name='Resultados')
        print(f"Datos cargados exitosamente. {len(mentions_df)} registros encontrados.")
        return mentions_df
    except Exception as e:
        print(f"Error al cargar desde Excel: {e}")
        # Intentar cargar como CSV como fallback
        csv_file = products_file.replace('.xlsx', '.csv')
        if os.path.exists(csv_file):
            try:
                mentions_df = pd.read_csv(csv_file)
                print(f"Datos cargados desde CSV. {len(mentions_df)} registros encontrados.")
                return mentions_df
            except Exception as csv_e:
                print(f"Error al cargar desde CSV: {csv_e}")
        
        raise FileNotFoundError(f"No se pudo cargar el archivo de menciones de productos en ningún formato")

def detect_cosmetic_patterns(text, attribute_dict=COSMETIC_ATTRIBUTES, nlp_model=None):
    """
    Detecta patrones de atributos cosméticos en el texto
    
    Args:
        text (str): Texto a analizar
        attribute_dict (dict): Diccionario de atributos cosméticos y sus valores
        nlp_model: Modelo spaCy para procesamiento de texto
        
    Returns:
        list: Lista de diccionarios con los atributos detectados
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    detected_attributes = []
    text_lower = text.lower()
    
    # 1. Detección basada en patrones definidos
    for category, attributes in attribute_dict.items():
        for attribute in attributes:
            attribute_lower = attribute.lower()
            # Buscar coincidencia exacta o con límites de palabra
            pattern = r'\b' + re.escape(attribute_lower) + r'\b'
            matches = re.finditer(pattern, text_lower)
            
            for match in matches:
                # Extraer contexto cercano (10 palabras antes y después)
                start_idx = max(0, match.start() - 50)
                end_idx = min(len(text_lower), match.end() + 50)
                context = text_lower[start_idx:end_idx]
                
                detected_attributes.append({
                    'category': category,
                    'attribute': attribute,
                    'position': match.start(),
                    'match_text': match.group(),
                    'context': context.strip(),
                    'method': 'pattern_matching'
                })
    
    # 2. Detección basada en NLP si se proporciona modelo
    if nlp_model is not None:
        doc = nlp_model(text)
        
        # Extraer frases nominales que pueden ser atributos cosméticos
        for chunk in doc.noun_chunks:
            # Filtrar frases nominales relevantes al dominio cosmético (en inglés)
            cosmetic_indicators = ['color', 'effect', 'finish', 'texture', 'feel', 'look',
                                  'result', 'formula', 'technology', 'action', 'benefit']
            
            chunk_text = chunk.text.lower()
            is_relevant = any(indicator in chunk_text for indicator in cosmetic_indicators)
            
            if is_relevant and len(chunk_text) > 3:  # Evitar chunks muy cortos
                # Identificar a qué categoría puede pertenecer
                assigned_category = "OTRO"
                for category, attributes in attribute_dict.items():
                    # Verificar si algún término de la categoría está en el chunk
                    if any(attr.lower() in chunk_text for attr in attributes):
                        assigned_category = category
                        break
                
                # Extraer contexto cercano
                context_start = max(0, chunk.start_char - 50)
                context_end = min(len(text), chunk.end_char + 50)
                context = text[context_start:context_end]
                
                detected_attributes.append({
                    'category': assigned_category,
                    'attribute': chunk.text,
                    'position': chunk.start_char,
                    'match_text': chunk.text,
                    'context': context.strip(),
                    'method': 'nlp_extraction'
                })
    
    return detected_attributes

def apply_ner_model(text, ner_pipeline):
    """
    Aplica modelo NER para detectar entidades relevantes
    
    Args:
        text (str): Texto a analizar
        ner_pipeline: Modelo NER de Hugging Face
        
    Returns:
        list: Lista de entidades detectadas
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Limitar longitud para evitar problemas con el modelo
    if len(text) > 512:
        # Dividir en fragmentos
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        all_entities = []
        
        for chunk in chunks:
            try:
                entities = ner_pipeline(chunk)
                all_entities.extend(entities)
            except Exception as e:
                print(f"Error al procesar chunk con NER: {e}")
                continue
        
        return all_entities
    else:
        try:
            return ner_pipeline(text)
        except Exception as e:
            print(f"Error al procesar texto con NER: {e}")
            return []

def classify_ner_entities(entities):
    """
    Clasifica entidades NER en categorías cosméticas
    
    Args:
        entities (list): Lista de entidades detectadas por el modelo NER
        
    Returns:
        list: Lista de entidades clasificadas
    """
    if not entities:
        return []
    
    classified_entities = []
    
    for entity in entities:
        if isinstance(entity, dict) and 'entity' in entity and 'word' in entity:
            entity_text = entity['word'].lower()
            
            # Clasificar entidad en categoría cosmética
            assigned_category = "OTRO"
            for category, attributes in COSMETIC_ATTRIBUTES.items():
                if any(attr.lower() in entity_text for attr in attributes):
                    assigned_category = category
                    break
            
            # Filtrar solo entidades que parezcan atributos cosméticos
            if assigned_category != "OTRO" or entity['entity'] in ['B-MISC', 'I-MISC', 'B-ORG', 'I-ORG']:
                classified_entities.append({
                    'category': assigned_category,
                    'attribute': entity['word'],
                    'entity_type': entity['entity'],
                    'score': entity.get('score', 0),
                    'method': 'ner_model'
                })
    
    return classified_entities

def analyze_cosmetic_attributes(mentions_df, nlp_model, ner_pipeline):
    """
    Analiza atributos cosméticos en las transcripciones
    
    Args:
        mentions_df (pd.DataFrame): DataFrame con menciones de productos
        nlp_model: Modelo spaCy para procesamiento de texto
        ner_pipeline: Modelo NER para detección de entidades
        
    Returns:
        tuple: (DataFrame con atributos detectados, métricas de análisis)
    """
    print("Analyzing cosmetic attributes in sentences...")
    
    all_attributes = []
    metrics = {
        'total_sentences': 0,
        'sentences_with_attributes': 0,
        'total_attributes': 0,
        'attributes_by_category': defaultdict(int),
        'attributes_by_method': defaultdict(int),
        'attributes_by_brand': defaultdict(lambda: defaultdict(int)),
        'top_attributes': defaultdict(int)
    }
    
    # Verificar columnas necesarias
    if 'sentence' not in mentions_df.columns:
        print("Error: Column 'sentence' not found in DataFrame")
        return pd.DataFrame(), metrics
    
    # Procesar a nivel de frase para mantener asociaciones correctas
    # Utilizamos una combinación de video_id y sentence para procesamiento único
    print("Processing by individual sentences to maintain accurate brand-product associations...")
    
    # Crear un identificador único por frase (combinación de video_id y sentence)
    if 'video_id' in mentions_df.columns and 'sentence' in mentions_df.columns:
        # Eliminar filas con frases vacías
        clean_df = mentions_df.dropna(subset=['sentence'])
        
        # Crear ID único para cada combinación de video+frase
        clean_df['sentence_id'] = clean_df['video_id'].astype(str) + '_' + clean_df['sentence'].astype(str).apply(lambda x: str(hash(x))[-8:])
        
        # Obtener frases únicas
        unique_sentences = clean_df.drop_duplicates(subset=['sentence_id'])[['sentence_id', 'video_id', 'sentence']]
        metrics['total_sentences'] = len(unique_sentences)
        
        for idx, row in tqdm(unique_sentences.iterrows(), total=len(unique_sentences), desc="Processing sentences"):
            sentence_id = row['sentence_id']
            video_id = row['video_id']
            sentence = row['sentence']
            
            # Verificar que la frase sea válida
            if not isinstance(sentence, str) or len(sentence.strip()) < 5:
                continue
                
            # Obtener marcas y productos específicamente mencionados en esta frase
            sentence_df = clean_df[clean_df['sentence_id'] == sentence_id]
            
            # Extraer marcas y productos específicos de esta frase
            sentence_brands = []
            sentence_products = []
            
            if 'brand' in sentence_df.columns and 'product' in sentence_df.columns:
                # Crear pares marca-producto para evitar asociaciones incorrectas
                for _, s_row in sentence_df.iterrows():
                    if pd.notna(s_row.get('brand')) and pd.notna(s_row.get('product')):
                        sentence_brands.append(s_row['brand'])
                        sentence_products.append(s_row['product'])
            
            # 1. Detectar patrones de atributos cosméticos
            pattern_attributes = detect_cosmetic_patterns(sentence, COSMETIC_ATTRIBUTES, nlp_model)
            
            # 2. Aplicar modelo NER para detectar entidades
            ner_entities = apply_ner_model(sentence, ner_pipeline)
            classified_entities = classify_ner_entities(ner_entities)
            
            # 3. Combinar resultados
            sentence_attributes = pattern_attributes + classified_entities
            
            # Si se detectaron atributos, procesar resultados
            if sentence_attributes:
                metrics['sentences_with_attributes'] += 1
                metrics['total_attributes'] += len(sentence_attributes)
                
                # Añadir información de la frase a cada atributo detectado
                for attr in sentence_attributes:
                    attr['video_id'] = video_id
                    attr['sentence_id'] = sentence_id
                    attr['sentence'] = sentence
                    
                    # Añadir marcas y productos asociados específicamente en esta frase
                    if sentence_brands and sentence_products:
                        # Si hay múltiples pares marca-producto en la misma frase, 
                        # los guardamos todos pero indicamos que la asociación puede ser ambigua
                        attr['associated_brands'] = sentence_brands
                        attr['associated_products'] = sentence_products
                        attr['association_confidence'] = 'high' if len(sentence_brands) == 1 else 'medium'
                    else:
                        attr['associated_brands'] = None
                        attr['associated_products'] = None
                        attr['association_confidence'] = 'none'
                    
                    # Actualizar métricas
                    metrics['attributes_by_category'][attr['category']] += 1
                    metrics['attributes_by_method'][attr.get('method', 'unknown')] += 1
                    metrics['top_attributes'][attr.get('attribute', '').lower()] += 1
                    
                    # Registrar atributos por marca (solo cuando hay una asociación clara)
                    if sentence_brands and len(sentence_brands) == 1:
                        brand = sentence_brands[0]
                        metrics['attributes_by_brand'][brand][attr['category']] += 1
                
                all_attributes.extend(sentence_attributes)
    
    # Convertir a DataFrame
    if all_attributes:
        attributes_df = pd.DataFrame(all_attributes)
        print(f"Analysis completed. Detected {len(attributes_df)} cosmetic attributes across {metrics['sentences_with_attributes']} sentences.")
        return attributes_df, metrics
    else:
        print("No cosmetic attributes detected.")
        return pd.DataFrame(), metrics

def generate_report(attributes_df, metrics, output_file):
    """
    Genera informe detallado de atributos cosméticos
    
    Args:
        attributes_df (pd.DataFrame): DataFrame con atributos detectados
        metrics (dict): Métricas del análisis
        output_file (str): Ruta del archivo de salida
    """
    print(f"Generating cosmetic attributes report in {output_file}...")
    
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. Hoja con datos completos
            if not attributes_df.empty:
                # Ordenar por video_id y categoría
                sorted_df = attributes_df.sort_values(by=['video_id', 'category'])
                sorted_df.to_excel(writer, sheet_name='Detected_Attributes', index=False)
            
            # 2. Hoja con resumen general
            summary_data = []
            
            # Métricas generales
            summary_data.append(["GENERAL SUMMARY", ""])
            summary_data.append(["Total sentences analyzed", metrics['total_sentences']])
            summary_data.append(["Sentences with attributes", metrics['sentences_with_attributes']])
            summary_data.append(["Detection rate (%)", f"{metrics['sentences_with_attributes']/max(1, metrics['total_sentences'])*100:.1f}%"])
            summary_data.append(["Total attributes detected", metrics['total_attributes']])
            summary_data.append(["Average attributes per sentence", f"{metrics['total_attributes']/max(1, metrics['sentences_with_attributes']):.2f}"])
            
            # Distribución por categoría
            if metrics['attributes_by_category']:
                summary_data.append(["", ""])
                summary_data.append(["DISTRIBUTION BY CATEGORY", "Count"])
                for category, count in sorted(metrics['attributes_by_category'].items(), key=lambda x: x[1], reverse=True):
                    summary_data.append([category, count])
            
            # Distribución por método
            if metrics['attributes_by_method']:
                summary_data.append(["", ""])
                summary_data.append(["DISTRIBUTION BY METHOD", "Count"])
                for method, count in sorted(metrics['attributes_by_method'].items(), key=lambda x: x[1], reverse=True):
                    summary_data.append([method, count])
            
            # Top atributos
            if metrics['top_attributes']:
                summary_data.append(["", ""])
                summary_data.append(["TOP 20 ATTRIBUTES", "Mentions"])
                top_20 = sorted(metrics['top_attributes'].items(), key=lambda x: x[1], reverse=True)[:20]
                for attr, count in top_20:
                    summary_data.append([attr, count])
            
            # Guardar resumen
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', header=False, index=False)
            
            # 3. Hoja con atributos por marca
            if metrics['attributes_by_brand']:
                brand_data = []
                brand_data.append(["BRAND"] + list(COSMETIC_ATTRIBUTES.keys()) + ["TOTAL"])
                
                for brand, categories in sorted(metrics['attributes_by_brand'].items(), 
                                             key=lambda x: sum(x[1].values()), reverse=True):
                    row = [brand]
                    brand_total = 0
                    
                    for category in COSMETIC_ATTRIBUTES.keys():
                        count = categories.get(category, 0)
                        row.append(count)
                        brand_total += count
                    
                    row.append(brand_total)
                    brand_data.append(row)
                
                brand_df = pd.DataFrame(brand_data)
                brand_df.to_excel(writer, sheet_name='Attributes_By_Brand', header=False, index=False)
            
            # 4. Hoja con co-ocurrencias de categorías
            if not attributes_df.empty and 'category' in attributes_df.columns:
                # Crear matriz de co-ocurrencia de categorías
                categories = list(COSMETIC_ATTRIBUTES.keys())
                cooccurrence_matrix = np.zeros((len(categories), len(categories)))
                
                # Agrupar por sentence_id para evitar asociaciones incorrectas
                if 'sentence_id' in attributes_df.columns:
                    # Usar sentence_id como unidad de análisis para co-ocurrencias
                    for sentence_id in attributes_df['sentence_id'].unique():
                        sentence_categories = attributes_df[attributes_df['sentence_id'] == sentence_id]['category'].unique()
                        
                        # Registrar co-ocurrencias
                        for i, cat1 in enumerate(categories):
                            for j, cat2 in enumerate(categories):
                                if cat1 in sentence_categories and cat2 in sentence_categories:
                                    cooccurrence_matrix[i, j] += 1
                
                # Convertir a DataFrame
                cooc_df = pd.DataFrame(cooccurrence_matrix, index=categories, columns=categories)
                cooc_df.to_excel(writer, sheet_name='Cooccurrences', index=True)
                
            # 5. Hoja con atributos por producto (nueva)
            if not attributes_df.empty and 'associated_products' in attributes_df.columns:
                # Extraer todos los productos mencionados
                all_products = []
                product_attributes = defaultdict(lambda: defaultdict(int))
                
                for _, row in attributes_df.iterrows():
                    if not pd.isna(row['associated_products']) and row['associated_products'] is not None:
                        products = row['associated_products']
                        if isinstance(products, list):
                            for product in products:
                                all_products.append(product)
                                product_attributes[product][row['category']] += 1
                
                # Si hay productos con atributos
                if product_attributes:
                    product_data = []
                    product_data.append(["PRODUCT"] + list(COSMETIC_ATTRIBUTES.keys()) + ["TOTAL"])
                    
                    for product, categories in sorted(product_attributes.items(), 
                                                 key=lambda x: sum(x[1].values()), reverse=True):
                        row = [product]
                        product_total = 0
                        
                        for category in COSMETIC_ATTRIBUTES.keys():
                            count = categories.get(category, 0)
                            row.append(count)
                            product_total += count
                        
                        row.append(product_total)
                        product_data.append(row)
                    
                    product_df = pd.DataFrame(product_data)
                    product_df.to_excel(writer, sheet_name='Attributes_By_Product', header=False, index=False)
        
        print(f"Report successfully generated in {output_file}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        print("Failed to generate Excel report. Please check the error message above.")

def main():
    """Main function for cosmetic attribute analysis"""
    start_time = time.time()
    
    try:
        # 1. Cargar modelos NLP
        nlp_en, ner_pipeline = load_nlp_models()
        
        # 2. Cargar datos de menciones de productos
        mentions_df = load_product_mentions()
        
        if mentions_df.empty:
            print("No data to analyze. Exiting.")
            return
        
        # 3. Analizar atributos cosméticos
        attributes_df, metrics = analyze_cosmetic_attributes(mentions_df, nlp_en, ner_pipeline)
        
        # 4. Generar informe
        generate_report(attributes_df, metrics, output_file)
        
        # Mostrar tiempo de ejecución
        end_time = time.time()
        execution_time = end_time - start_time
        minutes, seconds = divmod(execution_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print("===== ANALYSIS COMPLETED =====")
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()