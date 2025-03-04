# evaluation_metrics.py
# Código para evaluar el rendimiento de modelos NLP
# Este archivo debe estar en la misma carpeta que nlp.py

import pandas as pd
import numpy as np
import json
import os
import time
import random
import re
import traceback
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Importar funciones desde nlp.py (asegúrate de que nlp.py esté en la misma carpeta)
import sys
try:
    # Intentar importar funciones específicas
    from nlp import (
        detect_sephora_products, 
        analyze_sentiment, 
        extract_entities,
        cosmetic_patterns,
        get_cosmetic_category,
        sentence_model,
        sentiment_analyzer,
        nlp_en,
        sephora_df
    )
    print("✓ Funciones importadas correctamente desde nlp.py")
except ImportError:
    print("⚠ Error al importar funciones desde nlp.py")
    print("  Asegúrate de que nlp.py está en la misma carpeta y contiene las funciones necesarias.")
    print("  Alternativamente, puedes definir la ruta a nlp.py a continuación:")
    
    # Si nlp.py está en otra ubicación, descomentar y modificar la siguiente línea:
    # sys.path.append('ruta/a/la/carpeta/que/contiene/nlp.py')
    # from nlp import detect_sephora_products, analyze_sentiment, extract_entities
    
    sys.exit(1)

# Definir constantes y rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_evaluacion')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Archivo de resultados
DEFAULT_RESULTS_FILE = os.path.join(BASE_DIR, 'product_mentions_analysis.xlsx')

# --------------------------------------
# FUNCIONES DE CARGA DE DATOS
# --------------------------------------

def load_results_data(file_path=None):
    """
    Carga los datos de resultados desde un archivo Excel o CSV.
    
    Args:
        file_path (str, optional): Ruta al archivo de resultados. Si es None, 
                                   usa la ruta predeterminada.
        
    Returns:
        pd.DataFrame: DataFrame con los resultados, o None si hay un error.
    """
    if file_path is None:
        file_path = DEFAULT_RESULTS_FILE
    
    if not os.path.exists(file_path):
        print(f"⚠ Archivo no encontrado: {file_path}")
        return None
    
    try:
        if file_path.endswith('.xlsx'):
            results_df = pd.read_excel(file_path, sheet_name='Resultados')
        else:
            results_df = pd.read_csv(file_path)
        
        print(f"✓ Datos cargados correctamente: {len(results_df)} registros")
        return results_df
    except Exception as e:
        print(f"⚠ Error al cargar los datos: {e}")
        return None

def load_sample_videos(tiktok_df=None, sample_size=10):
    """
    Carga una muestra aleatoria de videos para evaluación.
    
    Args:
        tiktok_df (pd.DataFrame, optional): DataFrame con datos de TikTok.
        sample_size (int): Tamaño de la muestra.
        
    Returns:
        dict: Diccionario de {video_id: transcripción}.
    """
    sample_videos = {}
    
    if tiktok_df is None:
        # Si no se proporciona un DataFrame, intentar cargar desde un archivo de TikTok
        tiktok_file = os.path.join(BASE_DIR, '..', 'data', 'clean_data', 'url_data_cleaned.xlsx')
        if os.path.exists(tiktok_file):
            try:
                tiktok_df = pd.read_excel(tiktok_file)
                print(f"✓ Datos de TikTok cargados: {len(tiktok_df)} videos")
            except Exception as e:
                print(f"⚠ Error al cargar datos de TikTok: {e}")
                return sample_videos
        else:
            # Si no se encuentra el archivo, cargar desde los resultados
            results_df = load_results_data()
            if results_df is not None and 'sentence' in results_df.columns:
                # Usar las oraciones de los resultados como muestra
                sample_indices = random.sample(range(len(results_df)), min(sample_size, len(results_df)))
                for i in sample_indices:
                    video_id = results_df.iloc[i].get('video_id', f'sample_{i}')
                    text = results_df.iloc[i]['sentence']
                    if not pd.isna(text) and text:
                        sample_videos[str(video_id)] = text
                
                print(f"✓ Muestra creada a partir de resultados: {len(sample_videos)} textos")
                return sample_videos
            else:
                print("⚠ No se pudo cargar muestra de videos")
                return sample_videos
    
    # Si se proporciona un DataFrame, seleccionar muestra aleatoria
    if len(tiktok_df) > 0:
        sample_indices = random.sample(range(len(tiktok_df)), min(sample_size, len(tiktok_df)))
        for i in sample_indices:
            video_id = tiktok_df.iloc[i].get('id_urlvideo', f'sample_{i}')
            if 'transcription' in tiktok_df.columns and not pd.isna(tiktok_df.iloc[i]['transcription']):
                sample_videos[str(video_id)] = tiktok_df.iloc[i]['transcription']
    
    print(f"✓ Muestra creada: {len(sample_videos)} videos")
    return sample_videos

# --------------------------------------
# FUNCIONES DE EVALUACIÓN DE MODELOS
# --------------------------------------

def evaluate_bert_embedding_model(sample_texts, sephora_df, sentence_model):
    """
    Evalúa el rendimiento del modelo BERT de embeddings (SentenceTransformer)
    
    Args:
        sample_texts (list): Lista de textos para evaluar
        sephora_df (DataFrame): DataFrame con productos de Sephora
        sentence_model: Modelo SentenceTransformer
        
    Returns:
        dict: Métricas de evaluación del modelo
    """
    # Obtener los nombres de productos para comparación (limitado para eficiencia)
    product_names = sephora_df['product_with_brand'].dropna().tolist()[:100]
    brand_names = sephora_df['brand'].dropna().unique().tolist()
    
    # Inicializar métricas
    metrics = {
        'processing_time_ms': 0,
        'avg_similarity': 0,
        'similarity_distribution': {
            'high_similarity': 0,  # >0.8
            'medium_similarity': 0,  # 0.6-0.8
            'low_similarity': 0,  # <0.6
        },
        'brand_matches': 0,
        'embedding_dimensions': 0,
        'tokens_processed': 0,
        'processing_speed': 0,  # tokens por segundo
    }
    
    # Procesar cada texto
    start_time = time.time()
    total_tokens = 0
    similarity_scores = []
    brand_matches = 0
    
    try:
        # Primero, calcula embeddings para productos (una sola vez)
        products_embedding_start = time.time()
        product_embeddings = sentence_model.encode(product_names)
        products_embedding_time = time.time() - products_embedding_start
        
        # Guarda dimensión de embeddings
        metrics['embedding_dimensions'] = product_embeddings.shape[1]
        
        # Aproximación de tokens (basada en palabras)
        total_product_tokens = sum(len(p.split()) for p in product_names)
        
        # Para cada texto de muestra
        for text in sample_texts:
            if pd.isna(text) or text == "":
                continue
                
            # Dividir en oraciones
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if not sentences:
                continue
                
            # Aproximación de tokens
            text_tokens = sum(len(s.split()) for s in sentences)
            total_tokens += text_tokens
            
            # Calcular embeddings para las oraciones
            sentence_embeddings = sentence_model.encode(sentences)
            
            # Para cada oración, calcular similitud con productos
            for i, sentence in enumerate(sentences):
                # Verificar si hay alguna marca en la oración
                has_brand = any(brand.lower() in sentence.lower() for brand in brand_names if isinstance(brand, str))
                
                # Calcular similitud con productos
                similarities = np.inner(sentence_embeddings[i], product_embeddings)
                
                # Almacenar puntuaciones de similitud
                max_similarity = np.max(similarities)
                similarity_scores.append(max_similarity)
                
                # Verificar si la mayor similitud coincide con la presencia de marca
                if max_similarity > 0.7 and has_brand:
                    brand_matches += 1
        
        # Tiempo total de procesamiento
        total_time = time.time() - start_time
        metrics['processing_time_ms'] = total_time * 1000
        
        # Calcular métricas
        if similarity_scores:
            metrics['avg_similarity'] = np.mean(similarity_scores)
            metrics['similarity_distribution']['high_similarity'] = np.mean(np.array(similarity_scores) > 0.8)
            metrics['similarity_distribution']['medium_similarity'] = np.mean((np.array(similarity_scores) > 0.6) & (np.array(similarity_scores) <= 0.8))
            metrics['similarity_distribution']['low_similarity'] = np.mean(np.array(similarity_scores) <= 0.6)
        
        metrics['brand_matches'] = brand_matches
        metrics['tokens_processed'] = total_tokens + total_product_tokens
        
        if total_time > 0:
            metrics['processing_speed'] = metrics['tokens_processed'] / total_time
        
        # Detalles del modelo
        metrics['model_info'] = {
            'name': str(sentence_model.__class__.__name__),
            'dimensions': sentence_model.get_sentence_embedding_dimension(),
            'model_path': getattr(sentence_model, 'model_name_or_path', 'unknown'),
            'product_embedding_time_ms': products_embedding_time * 1000
        }
    
    except Exception as e:
        print(f"⚠ Error en evaluación BERT embedding: {e}")
        traceback.print_exc()
    
    return metrics


def evaluate_bert_sentiment_model(sample_texts, sentiment_analyzer):
    """
    Evalúa el rendimiento del modelo BERT de análisis de sentimiento
    
    Args:
        sample_texts (list): Lista de textos para evaluar
        sentiment_analyzer: Pipeline de análisis de sentimiento
        
    Returns:
        dict: Métricas de evaluación del modelo
    """
    metrics = {
        'processing_time_ms': 0,
        'label_distribution': {},
        'avg_confidence': 0,
        'confidence_by_label': {},
        'processing_speed': 0,  # tokens por segundo
        'tokens_processed': 0,
        'sentence_length_impact': {
            'short': {'avg_time': 0, 'avg_confidence': 0, 'count': 0},  # <100 chars
            'medium': {'avg_time': 0, 'avg_confidence': 0, 'count': 0},  # 100-300 chars
            'long': {'avg_time': 0, 'avg_confidence': 0, 'count': 0},  # >300 chars
        }
    }
    
    try:
        # Procesar textos
        confidence_scores = []
        label_counts = Counter()
        confidence_by_label = {}
        
        short_times = []
        medium_times = []
        long_times = []
        
        short_confs = []
        medium_confs = []
        long_confs = []
        
        total_time = 0
        total_tokens = 0
        
        for text in sample_texts:
            if pd.isna(text) or text == "":
                continue
                
            # Aproximación de tokens
            tokens = len(text.split())
            total_tokens += tokens
            
            # Medir tiempo por longitud
            if len(text) > 512:
                chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                chunk_times = []
                chunk_scores = []
                chunk_labels = []
                
                for chunk in chunks:
                    chunk_start = time.time()
                    result = sentiment_analyzer(chunk)[0]
                    chunk_time = time.time() - chunk_start
                    
                    chunk_times.append(chunk_time)
                    chunk_scores.append(result['score'])
                    chunk_labels.append(result['label'])
                    
                    # Categorizar por longitud
                    if len(chunk) < 100:
                        short_times.append(chunk_time)
                        short_confs.append(result['score'])
                    elif len(chunk) < 300:
                        medium_times.append(chunk_time)
                        medium_confs.append(result['score'])
                    else:
                        long_times.append(chunk_time)
                        long_confs.append(result['score'])
                
                # Procesar resultados agregados
                total_time += sum(chunk_times)
                most_common_label = Counter(chunk_labels).most_common(1)[0][0]
                avg_score = sum([s for l, s in zip(chunk_labels, chunk_scores) if l == most_common_label]) / chunk_labels.count(most_common_label)
                
                label_counts[most_common_label] += 1
                
                if most_common_label not in confidence_by_label:
                    confidence_by_label[most_common_label] = []
                confidence_by_label[most_common_label].append(avg_score)
                confidence_scores.append(avg_score)
                
            else:
                # Textos cortos, procesamiento directo
                start_time = time.time()
                result = sentiment_analyzer(text)[0]
                process_time = time.time() - start_time
                total_time += process_time
                
                label = result['label']
                score = result['score']
                
                label_counts[label] += 1
                
                if label not in confidence_by_label:
                    confidence_by_label[label] = []
                confidence_by_label[label].append(score)
                confidence_scores.append(score)
                
                # Categorizar por longitud
                if len(text) < 100:
                    short_times.append(process_time)
                    short_confs.append(score)
                elif len(text) < 300:
                    medium_times.append(process_time)
                    medium_confs.append(score)
                else:
                    long_times.append(process_time)
                    long_confs.append(score)
        
        # Calcular métricas
        metrics['processing_time_ms'] = total_time * 1000
        
        if confidence_scores:
            metrics['avg_confidence'] = np.mean(confidence_scores)
        
        # Distribución de etiquetas
        total_samples = sum(label_counts.values())
        if total_samples > 0:
            metrics['label_distribution'] = {label: count/total_samples for label, count in label_counts.items()}
        
        # Confianza por etiqueta
        for label, scores in confidence_by_label.items():
            metrics['confidence_by_label'][label] = np.mean(scores)
        
        # Velocidad de procesamiento
        if total_time > 0:
            metrics['processing_speed'] = total_tokens / total_time
        metrics['tokens_processed'] = total_tokens
        
        # Impacto de longitud de oración
        if short_times:
            metrics['sentence_length_impact']['short']['avg_time'] = np.mean(short_times) * 1000
            metrics['sentence_length_impact']['short']['avg_confidence'] = np.mean(short_confs)
            metrics['sentence_length_impact']['short']['count'] = len(short_times)
        
        if medium_times:
            metrics['sentence_length_impact']['medium']['avg_time'] = np.mean(medium_times) * 1000
            metrics['sentence_length_impact']['medium']['avg_confidence'] = np.mean(medium_confs)
            metrics['sentence_length_impact']['medium']['count'] = len(medium_times)
        
        if long_times:
            metrics['sentence_length_impact']['long']['avg_time'] = np.mean(long_times) * 1000
            metrics['sentence_length_impact']['long']['avg_confidence'] = np.mean(long_confs)
            metrics['sentence_length_impact']['long']['count'] = len(long_times)
    
    except Exception as e:
        print(f"⚠ Error en evaluación BERT sentiment: {e}")
        traceback.print_exc()
    
    return metrics


def evaluate_spacy_ner_model(sample_texts, nlp_model, cosmetic_patterns=None):
    """
    Evalúa el rendimiento del modelo SpaCy NER, incluyendo la detección de atributos cosméticos
    
    Args:
        sample_texts (list): Lista de textos para evaluar
        nlp_model: Modelo SpaCy
        cosmetic_patterns (list, optional): Lista de patrones regex para atributos cosméticos
        
    Returns:
        dict: Métricas de evaluación del modelo
    """
    metrics = {
        'processing_time_ms': 0,
        'entity_counts': {},
        'entity_tokens_ratio': 0,  # Proporción de tokens que son entidades
        'cosmetic_attribute_counts': {},
        'pattern_match_counts': {},
        'ngram_counts': 0,
        'tokens_processed': 0,
        'processing_speed': 0,  # tokens por segundo
        'avg_entity_length': 0,
        'entity_type_distribution': {},
        'context_overlap': 0  # Grado de superposición de contextos
    }
    
    try:
        # Contadores
        entity_counts = Counter()
        cosmetic_counts = Counter()
        pattern_counts = Counter()
        
        # Procesamiento
        total_entities = 0
        total_tokens = 0
        total_time = 0
        entity_lengths = []
        
        context_ranges = []  # Para medir superposición
        
        for text_idx, text in enumerate(sample_texts):
            if pd.isna(text) or text == "":
                continue
                
            # Procesar con SpaCy
            start_time = time.time()
            doc = nlp_model(text)
            spacy_time = time.time() - start_time
            
            # Contar tokens
            total_tokens += len(doc)
            
            # Contar entidades NER de SpaCy
            spacy_entities = list(doc.ents)
            total_entities += len(spacy_entities)
            
            for ent in spacy_entities:
                entity_counts[ent.label_] += 1
                entity_lengths.append(len(ent.text.split()))
                
                # Guardar rango de contexto
                start_context = max(0, ent.start_char - 30)
                end_context = min(len(text), ent.end_char + 30)
                context_ranges.append((start_context, end_context))
            
            # Si hay patrones cosméticos, procesar
            if cosmetic_patterns:
                pattern_start_time = time.time()
                
                # Procesar con extract_entities completo (que usa el modelo NLP y los patrones)
                entities = extract_entities(text, nlp_model)
                
                pattern_time = time.time() - pattern_start_time
                
                # Contar atributos cosméticos
                cosmetic_entities = [e for e in entities if e.get('type') == 'COSMETIC_ATTRIBUTE']
                
                for entity in cosmetic_entities:
                    subtype = entity.get('subtype', 'UNKNOWN')
                    cosmetic_counts[subtype] += 1
                    
                    if entity.get('text'):
                        entity_lengths.append(len(entity.get('text').split()))
                    
                    # Contar por patrón
                    if 'NGRAM' in str(subtype):
                        pattern_counts['ngram'] += 1
                    elif 'INDUSTRY_TERM' in str(subtype):
                        pattern_counts['industry_term'] += 1
                    else:
                        pattern_counts['regex_pattern'] += 1
                    
                    # Guardar rango de contexto
                    if 'start' in entity and 'end' in entity:
                        start_context = max(0, entity['start'] - 30)
                        end_context = min(len(text), entity['end'] + 30)
                        context_ranges.append((start_context, end_context))
                
                total_time += spacy_time + pattern_time
            else:
                total_time += spacy_time
        
        # Calcular métricas
        metrics['processing_time_ms'] = total_time * 1000
        metrics['entity_counts'] = dict(entity_counts)
        
        if total_tokens > 0:
            metrics['entity_tokens_ratio'] = total_entities / total_tokens
            metrics['processing_speed'] = total_tokens / total_time if total_time > 0 else 0
        
        metrics['tokens_processed'] = total_tokens
        
        if entity_lengths:
            metrics['avg_entity_length'] = np.mean(entity_lengths)
        
        # Calcular distribución de tipos de entidades
        total_entity_count = sum(entity_counts.values())
        if total_entity_count > 0:
            metrics['entity_type_distribution'] = {ent_type: count/total_entity_count 
                                                for ent_type, count in entity_counts.items()}
        
        # Para entidades cosméticas
        if cosmetic_patterns:
            metrics['cosmetic_attribute_counts'] = dict(cosmetic_counts)
            metrics['pattern_match_counts'] = dict(pattern_counts)
            metrics['ngram_counts'] = pattern_counts.get('ngram', 0)
        
        # Calcular superposición de contextos
        if len(context_ranges) > 1:
            overlap_count = 0
            total_pairs = 0
            
            for i, (start1, end1) in enumerate(context_ranges):
                for j, (start2, end2) in enumerate(context_ranges[i+1:], i+1):
                    total_pairs += 1
                    
                    # Hay superposición si el inicio de uno está dentro del rango del otro
                    if (start1 <= start2 <= end1) or (start2 <= start1 <= end2):
                        overlap_count += 1
            
            if total_pairs > 0:
                metrics['context_overlap'] = overlap_count / total_pairs
    
    except Exception as e:
        print(f"⚠ Error en evaluación SpaCy NER: {e}")
        traceback.print_exc()
    
    return metrics


def evaluate_nlp_models_on_existing_results(results_df, sample_size=50):
    """
    Evalúa el desempeño de los modelos NLP a partir de resultados ya existentes
    con protección contra errores de tipo de datos
    
    Args:
        results_df (pd.DataFrame): DataFrame con resultados del análisis
        sample_size (int): Tamaño de la muestra a evaluar
        
    Returns:
        dict: Métricas de evaluación de los modelos
    """
    print("=" * 60)
    print("EVALUACIÓN DE MODELOS NLP A PARTIR DE RESULTADOS EXISTENTES")
    print("=" * 60)
    
    try:
        # Seleccionar muestra aleatoria (o todo si es menor)
        sample = results_df.sample(min(sample_size, len(results_df)))
        
        # Métricas para modelo BERT (Detección de productos)
        bert_detection_metrics = {
            'detection_types': {},
            'detection_methods': {},
            'avg_confidence': 0,
            'confidence_distribution': {
                'high': 0,  # >0.8
                'medium': 0,  # 0.5-0.8
                'low': 0  # <0.5
            }
        }
        
        # Métricas para modelo BERT (Sentimiento)
        bert_sentiment_metrics = {
            'sentiment_distribution': {},
            'avg_confidence': 0,
            'confidence_by_sentiment': {},
            'sentiment_by_detection_type': {}
        }
        
        # Métricas para SpaCy NER
        spacy_ner_metrics = {
            'entity_type_distribution': {},
            'entity_subtype_distribution': {},
            'avg_entities_per_sample': 0,
            'entity_context_quality': {
                'with_context': 0,
                'without_context': 0
            }
        }
        
        # 1. Analizar métricas del modelo BERT (Detección)
        detection_types = {}
        detection_methods = {}
        confidence_scores = []
        
        # 2. Analizar métricas del modelo BERT (Sentimiento)
        sentiment_counts = Counter()
        sentiment_scores = []
        sentiment_by_detection = {}
        sentiment_confidence_by_type = {}
        
        # 3. Analizar métricas del modelo SpaCy NER
        entity_types = Counter()
        entity_subtypes = Counter()
        entities_per_sample = []
        context_quality = {'with_context': 0, 'without_context': 0}
        
        # Procesar cada muestra
        for idx, row in sample.iterrows():
            # BERT Detección
            detection_type = row.get('detection_type', 'unknown')
            detection_method = row.get('method', 'unknown')
            
            detection_types[detection_type] = detection_types.get(detection_type, 0) + 1
            detection_methods[detection_method] = detection_methods.get(detection_method, 0) + 1
            
            # Asegurarse de que score sea numérico
            if 'score' in row:
                try:
                    score_value = float(row['score'])
                    confidence_scores.append(score_value)
                except (ValueError, TypeError):
                    pass  # Ignorar valores no numéricos
            
            # BERT Sentimiento
            if 'sentiment' in row and row['sentiment']:
                sentiment = str(row['sentiment'])  # Convertir a string para evitar problemas
                sentiment_counts[sentiment] += 1
                
                # Relación sentimiento-detección
                if detection_type not in sentiment_by_detection:
                    sentiment_by_detection[detection_type] = Counter()
                sentiment_by_detection[detection_type][sentiment] += 1
                
                # Confianza por tipo
                if 'sentiment_score' in row:
                    try:
                        sentiment_score = float(row['sentiment_score'])
                        if sentiment not in sentiment_confidence_by_type:
                            sentiment_confidence_by_type[sentiment] = []
                        sentiment_confidence_by_type[sentiment].append(sentiment_score)
                        sentiment_scores.append(sentiment_score)
                    except (ValueError, TypeError):
                        pass  # Ignorar valores no numéricos
            
            # SpaCy NER
            if 'extracted_entities_str' in row:
                try:
                    entities = json.loads(row['extracted_entities_str'])
                    if isinstance(entities, list):
                        entities_per_sample.append(len(entities))
                        
                        for entity in entities:
                            if isinstance(entity, dict):
                                entity_type = entity.get('type', 'unknown')
                                entity_types[entity_type] += 1
                                
                                if entity_type == 'COSMETIC_ATTRIBUTE':
                                    subtype = entity.get('subtype', 'UNKNOWN')
                                    entity_subtypes[subtype] += 1
                                
                                # Calidad del contexto
                                if 'context' in entity and entity['context']:
                                    context_quality['with_context'] += 1
                                else:
                                    context_quality['without_context'] += 1
                except:
                    # Error al procesar JSON o acceder a atributos
                    pass
        
        # Calcular métricas finales
        
        # 1. BERT Detección
        bert_detection_metrics['detection_types'] = detection_types
        bert_detection_metrics['detection_methods'] = detection_methods
        
        if confidence_scores:
            bert_detection_metrics['avg_confidence'] = np.mean(confidence_scores)
            bert_detection_metrics['confidence_distribution']['high'] = np.mean(np.array(confidence_scores) > 0.8)
            bert_detection_metrics['confidence_distribution']['medium'] = np.mean((np.array(confidence_scores) > 0.5) & (np.array(confidence_scores) <= 0.8))
            bert_detection_metrics['confidence_distribution']['low'] = np.mean(np.array(confidence_scores) <= 0.5)
        
        # 2. BERT Sentimiento
        total_sentiments = sum(sentiment_counts.values())
        if total_sentiments > 0:
            bert_sentiment_metrics['sentiment_distribution'] = {
                sentiment: count/total_sentiments for sentiment, count in sentiment_counts.items()
            }
        
        if sentiment_scores:
            bert_sentiment_metrics['avg_confidence'] = np.mean(sentiment_scores)
        
        for sentiment, scores in sentiment_confidence_by_type.items():
            bert_sentiment_metrics['confidence_by_sentiment'][sentiment] = np.mean(scores)
        
        for detection_type, sentiments in sentiment_by_detection.items():
            total = sum(sentiments.values())
            if total > 0:
                bert_sentiment_metrics['sentiment_by_detection_type'][detection_type] = {
                    sentiment: count/total for sentiment, count in sentiments.items()
                }
        
        # 3. SpaCy NER
        total_entities = sum(entity_types.values())
        if total_entities > 0:
            spacy_ner_metrics['entity_type_distribution'] = {
                entity_type: count/total_entities for entity_type, count in entity_types.items()
            }
        
        total_cosmetic_attrs = sum(entity_subtypes.values())
        if total_cosmetic_attrs > 0:
            spacy_ner_metrics['entity_subtype_distribution'] = {
                subtype: count/total_cosmetic_attrs for subtype, count in entity_subtypes.items()
            }
        
        if entities_per_sample:
            spacy_ner_metrics['avg_entities_per_sample'] = np.mean(entities_per_sample)
        
        total_context_checks = sum(context_quality.values())
        if total_context_checks > 0:
            spacy_ner_metrics['entity_context_quality']['with_context'] = context_quality['with_context'] / total_context_checks
            spacy_ner_metrics['entity_context_quality']['without_context'] = context_quality['without_context'] / total_context_checks
        
        # Imprimir resultados
        
        # 1. BERT Detección
        print("\n1. MODELO BERT (DETECCIÓN DE PRODUCTOS/MARCAS)")
        print("-" * 40)
        
        print("  Distribución de tipos de detección:")
        for d_type, count in sorted(detection_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(sample)) * 100
            print(f"    {d_type}: {count} ({percentage:.1f}%)")
        
        print("\n  Métodos de detección:")
        # Asegurar que todos sean numéricos y luego filtrar 'none'
        numeric_methods = {k: count for k, count in detection_methods.items() if k != 'none'}
        total_methods = sum(numeric_methods.values())
        
        for method, count in sorted(numeric_methods.items(), key=lambda x: x[1], reverse=True):
            if total_methods > 0:
                percentage = (count / total_methods) * 100
                print(f"    {method}: {count} ({percentage:.1f}%)")
        
        print(f"\n  Confianza promedio: {bert_detection_metrics['avg_confidence']:.4f}")
        print("  Distribución de confianza:")
        for level, pct in bert_detection_metrics['confidence_distribution'].items():
            print(f"    {level}: {pct:.1%}")
        
        # 2. BERT Sentimiento
        print("\n2. MODELO BERT (ANÁLISIS DE SENTIMIENTO)")
        print("-" * 40)
        
        print("  Distribución de sentimiento:")
        for sentiment, pct in sorted(bert_sentiment_metrics['sentiment_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {sentiment}: {pct:.1%}")
        
        print(f"\n  Confianza promedio: {bert_sentiment_metrics['avg_confidence']:.4f}")
        print("  Confianza por sentimiento:")
        for sentiment, conf in sorted(bert_sentiment_metrics['confidence_by_sentiment'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {sentiment}: {conf:.4f}")
        
        print("\n  Sentimiento por tipo de detección:")
        for detection, sentiments in bert_sentiment_metrics['sentiment_by_detection_type'].items():
            print(f"    {detection}:")
            for sentiment, pct in sorted(sentiments.items(), key=lambda x: x[1], reverse=True)[:3]:  # Top 3
                print(f"      {sentiment}: {pct:.1%}")
        
        # 3. SpaCy NER
        print("\n3. MODELO SPACY NER (ENTIDADES Y ATRIBUTOS)")
        print("-" * 40)
        
        print("  Distribución de tipos de entidades:")
        for entity_type, pct in sorted(spacy_ner_metrics['entity_type_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {entity_type}: {pct:.1%}")
        
        print("\n  Top subtypes de atributos cosméticos:")
        sorted_subtypes = sorted(spacy_ner_metrics['entity_subtype_distribution'].items(), key=lambda x: x[1], reverse=True)
        for subtype, pct in sorted_subtypes[:10]:  # Top 10
            print(f"    {subtype}: {pct:.1%}")
        
        print(f"\n  Entidades promedio por muestra: {spacy_ner_metrics['avg_entities_per_sample']:.2f}")
        print("  Calidad del contexto de entidades:")
        print(f"    Con contexto: {spacy_ner_metrics['entity_context_quality']['with_context']:.1%}")
        print(f"    Sin contexto: {spacy_ner_metrics['entity_context_quality']['without_context']:.1%}")
        
        # Retornar métricas completas
        results = {
            'bert_detection': bert_detection_metrics,
            'bert_sentiment': bert_sentiment_metrics,
            'spacy_ner': spacy_ner_metrics,
            'sample_size': len(sample)
        }
        
    except Exception as e:
        import traceback
        print(f"Error durante la evaluación: {e}")
        print(traceback.format_exc())
        results = {}
    
    print("\n" + "=" * 60)
    print("FIN DE LA EVALUACIÓN DE MODELOS NLP (RESULTADOS EXISTENTES)")
    print("=" * 60)
    
    return results


def combined_model_evaluation(sample_videos=None, sample_size=10):
    """
    Realiza una evaluación combinada de todos los modelos NLP utilizados
    
    Args:
        sample_videos (dict, optional): Diccionario de {video_id: transcripción}
        sample_size (int): Tamaño de la muestra a evaluar
        
    Returns:
        dict: Métricas de evaluación conjunta
    """
    print("=" * 60)
    print("EVALUACIÓN DE MODELOS NLP (BERT & NER)")
    print("=" * 60)
    
    try:
        # Si no se proporcionan videos, seleccionar muestra aleatoria
        if not sample_videos:
            sample_videos = load_sample_videos(sample_size=sample_size)
            
        if not sample_videos:
            print("⚠ No se pudieron cargar videos para evaluar.")
            return {}
        
        # Convertir a listas para evaluar
        sample_texts = list(sample_videos.values())
        
        print(f"\nEvaluando {len(sample_texts)} textos de muestra...")
        
        # 1. Evaluación del modelo de embeddings (BERT SentenceTransformer)
        print("\n1. EVALUACIÓN DEL MODELO DE EMBEDDINGS (BERT)")
        print("-" * 40)
        
        start_time = time.time()
        bert_embed_metrics = evaluate_bert_embedding_model(sample_texts, sephora_df, sentence_model)
        print(f"Tiempo de evaluación: {time.time() - start_time:.2f} segundos")
        
        print(f"  Dimensiones de embeddings: {bert_embed_metrics['embedding_dimensions']}")
        print(f"  Similitud promedio: {bert_embed_metrics['avg_similarity']:.4f}")
        print("  Distribución de similitud:")
        for level, value in bert_embed_metrics['similarity_distribution'].items():
            print(f"    {level}: {value:.2%}")
        print(f"  Velocidad: {bert_embed_metrics['processing_speed']:.2f} tokens/seg")
        
        # 2. Evaluación del modelo de sentimiento (BERT)
        print("\n2. EVALUACIÓN DEL MODELO DE SENTIMIENTO (BERT)")
        print("-" * 40)
        
        start_time = time.time()
        bert_sentiment_metrics = evaluate_bert_sentiment_model(sample_texts, sentiment_analyzer)
        print(f"Tiempo de evaluación: {time.time() - start_time:.2f} segundos")
        
        print(f"  Confianza promedio: {bert_sentiment_metrics['avg_confidence']:.4f}")
        print("  Distribución de etiquetas:")
        for label, value in bert_sentiment_metrics['label_distribution'].items():
            print(f"    {label}: {value:.2%}")
        print("  Confianza por etiqueta:")
        for label, conf in bert_sentiment_metrics['confidence_by_label'].items():
            print(f"    {label}: {conf:.4f}")
        print(f"  Velocidad: {bert_sentiment_metrics['processing_speed']:.2f} tokens/seg")
        
        # 3. Evaluación del modelo NER (SpaCy + patrones)
        print("\n3. EVALUACIÓN DEL MODELO NER (SpaCy + Patrones)")
        print("-" * 40)
        
        start_time = time.time()
        ner_metrics = evaluate_spacy_ner_model(sample_texts, nlp_en, cosmetic_patterns)
        print(f"Tiempo de evaluación: {time.time() - start_time:.2f} segundos")
        
        print(f"  Entidades detectadas: {sum(ner_metrics['entity_counts'].values())}")
        print(f"  Atributos cosméticos: {sum(ner_metrics['cosmetic_attribute_counts'].values()) if 'cosmetic_attribute_counts' in ner_metrics else 0}")
        print(f"  Longitud promedio de entidad: {ner_metrics['avg_entity_length']:.2f} tokens")
        print(f"  Proporción tokens/entidades: {ner_metrics['entity_tokens_ratio']:.4f}")
        
        print("\n  Top tipos de entidades SpaCy:")
        top_entities = sorted(ner_metrics['entity_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
        for entity_type, count in top_entities:
            print(f"    {entity_type}: {count}")
        
        if 'cosmetic_attribute_counts' in ner_metrics and ner_metrics['cosmetic_attribute_counts']:
            print("\n  Top atributos cosméticos:")
            top_cosmetics = sorted(ner_metrics['cosmetic_attribute_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
            for attr_type, count in top_cosmetics:
                print(f"    {attr_type}: {count}")
        
        print("\n  Método de detección:")
        if 'pattern_match_counts' in ner_metrics:
            for method, count in ner_metrics['pattern_match_counts'].items():
                print(f"    {method}: {count}")
        
        print(f"  Velocidad: {ner_metrics['processing_speed']:.2f} tokens/seg")
        
        # 4. Resumen comparativo
        print("\n4. RESUMEN COMPARATIVO DE MODELOS")
        print("-" * 40)
        
        models_summary = {
            "BERT Embeddings": {
                "throughput": bert_embed_metrics['processing_speed'],
                "confidence": bert_embed_metrics['avg_similarity'],
                "tokens": bert_embed_metrics['tokens_processed']
            },
            "BERT Sentiment": {
                "throughput": bert_sentiment_metrics['processing_speed'],
                "confidence": bert_sentiment_metrics['avg_confidence'],
                "tokens": bert_sentiment_metrics['tokens_processed']
            },
            "SpaCy NER": {
                "throughput": ner_metrics['processing_speed'],
                "entity_ratio": ner_metrics['entity_tokens_ratio'],
                "tokens": ner_metrics['tokens_processed']
            }
        }
        
        print("  Velocidad de procesamiento (tokens/seg):")
        for model, metrics in models_summary.items():
            print(f"    {model}: {metrics.get('throughput', 0):.2f}")
        
        print("\n  Confianza/Precisión:")
        print(f"    BERT Embeddings (similitud): {models_summary['BERT Embeddings']['confidence']:.4f}")
        print(f"    BERT Sentiment: {models_summary['BERT Sentiment']['confidence']:.4f}")
        print(f"    SpaCy NER (entidades/token): {models_summary['SpaCy NER']['entity_ratio']:.4f}")
        
        # Resultados completos
        results = {
            "bert_embeddings": bert_embed_metrics,
            "bert_sentiment": bert_sentiment_metrics,
            "spacy_ner": ner_metrics,
            "summary": models_summary
        }
        
        # Guardar resultados
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_file = os.path.join(OUTPUT_DIR, f"evaluation_results_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResultados guardados en: {result_file}")
        
    except Exception as e:
        print(f"\n⚠ Error durante la evaluación: {e}")
        traceback.print_exc()
        results = {}
    
    print("\n" + "=" * 60)
    print("FIN DE LA EVALUACIÓN DE MODELOS NLP")
    print("=" * 60)
    
    return results


# --------------------------------------
# FUNCIONES DE VISUALIZACIÓN
# --------------------------------------

def create_model_performance_charts(evaluation_results, output_dir=None):
    """
    Crea gráficos para visualizar el rendimiento de los modelos
    
    Args:
        evaluation_results (dict): Resultados de evaluación
        output_dir (str, optional): Directorio para guardar gráficos
        
    Returns:
        list: Rutas a los gráficos creados
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    chart_files = []
    
    try:
        # Configurar estilo
        plt.style.use('ggplot')
        
        # 1. Gráfico de velocidad de procesamiento
        fig, ax = plt.subplots(figsize=(10, 6))
        models = []
        speeds = []
        
        for model, data in evaluation_results.get('summary', {}).items():
            if 'throughput' in data:
                models.append(model)
                speeds.append(data['throughput'])
        
        if models and speeds:
            bars = ax.bar(models, speeds, color=['#3498db', '#e74c3c', '#2ecc71'])
            ax.set_title('Velocidad de Procesamiento por Modelo', fontsize=16)
            ax.set_ylabel('Tokens por segundo', fontsize=14)
            ax.set_ylim(bottom=0)
            
            # Añadir etiquetas
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12)
            
            # Guardar gráfico
            chart_path = os.path.join(output_dir, 'model_speed_comparison.png')
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()
            chart_files.append(chart_path)
            
        # 2. Gráfico de distribución de sentimiento
        if 'bert_sentiment' in evaluation_results:
            sentiment_dist = evaluation_results['bert_sentiment'].get('label_distribution', {})
            if sentiment_dist:
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiments = []
                percentages = []
                
                for sentiment, pct in sorted(sentiment_dist.items(), key=lambda x: x[1], reverse=True):
                    sentiments.append(sentiment)
                    percentages.append(pct * 100)  # Convertir a porcentaje
                
                bars = ax.bar(sentiments, percentages, color=plt.cm.viridis(np.linspace(0, 0.8, len(sentiments))))
                ax.set_title('Distribución de Sentimiento', fontsize=16)
                ax.set_ylabel('Porcentaje (%)', fontsize=14)
                ax.set_ylim(bottom=0, top=max(percentages) * 1.1)
                
                # Añadir etiquetas
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=12)
                
                # Guardar gráfico
                chart_path = os.path.join(output_dir, 'sentiment_distribution.png')
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
                chart_files.append(chart_path)
        
        # 3. Gráfico de tipos de entidades NER
        if 'spacy_ner' in evaluation_results:
            entity_dist = evaluation_results['spacy_ner'].get('entity_type_distribution', {})
            if entity_dist:
                # Ordenar y limitar a top 10
                top_entities = sorted(entity_dist.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                entity_types = [e[0] for e in top_entities]
                percentages = [e[1] * 100 for e in top_entities]  # Convertir a porcentaje
                
                bars = ax.barh(entity_types, percentages, color=plt.cm.plasma(np.linspace(0.1, 0.9, len(entity_types))))
                ax.set_title('Top 10 Tipos de Entidades Detectadas', fontsize=16)
                ax.set_xlabel('Porcentaje (%)', fontsize=14)
                ax.set_xlim(right=max(percentages) * 1.1)
                
                # Añadir etiquetas
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.annotate(f'{width:.1f}%',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(3, 0),
                                textcoords="offset points",
                                ha='left', va='center', fontsize=10)
                
                # Guardar gráfico
                chart_path = os.path.join(output_dir, 'entity_types_distribution.png')
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
                chart_files.append(chart_path)
        
        # 4. Gráfico de distribución de atributos cosméticos
        if 'spacy_ner' in evaluation_results:
            cosmetic_dist = evaluation_results['spacy_ner'].get('cosmetic_attribute_counts', {})
            if cosmetic_dist:
                # Ordenar y limitar a top 10
                top_attrs = sorted(cosmetic_dist.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig, ax = plt.subplots(figsize=(12, 8))
                attr_types = [a[0] for a in top_attrs]
                counts = [a[1] for a in top_attrs]
                
                # Colores
                colors = plt.cm.tab20(np.linspace(0, 1, len(attr_types)))
                
                # Crear gráfico de pastel
                wedges, texts, autotexts = ax.pie(
                    counts, 
                    labels=attr_types, 
                    autopct='%1.1f%%',
                    startangle=90, 
                    colors=colors,
                    wedgeprops={'edgecolor': 'w', 'linewidth': 1}
                )
                
                # Propiedades del texto
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    autotext.set_color('white')
                
                ax.set_title('Distribución de Atributos Cosméticos', fontsize=16)
                ax.axis('equal')  # Círculo en lugar de elipse
                
                # Guardar gráfico
                chart_path = os.path.join(output_dir, 'cosmetic_attributes_distribution.png')
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
                chart_files.append(chart_path)
        
        print(f"✓ {len(chart_files)} gráficos creados en: {output_dir}")
    
    except Exception as e:
        print(f"⚠ Error al crear gráficos: {e}")
        traceback.print_exc()
    
    return chart_files


# --------------------------------------
# FUNCIONES PRINCIPALES
# --------------------------------------

def run_full_evaluation(results_file=None, sample_size=20, create_charts=True):
    """
    Ejecuta la evaluación completa de los modelos NLP
    
    Args:
        results_file (str, optional): Ruta al archivo de resultados
        sample_size (int): Tamaño de la muestra para evaluación
        create_charts (bool): Si se deben crear gráficos
        
    Returns:
        dict: Resultados de la evaluación
    """
    print("\n" + "=" * 70)
    print(" EVALUACIÓN COMPLETA DE MODELOS NLP (BERT + NER)")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Evaluación directa de los modelos con muestra aleatoria
    print("\n[1/3] Evaluando modelos con muestra aleatoria...")
    direct_results = combined_model_evaluation(sample_size=sample_size)
    all_results['direct_evaluation'] = direct_results
    
    # 2. Evaluación basada en resultados existentes
    print("\n[2/3] Evaluando a partir de resultados existentes...")
    results_df = load_results_data(results_file)
    
    if results_df is not None and len(results_df) > 0:
        existing_results = evaluate_nlp_models_on_existing_results(results_df, sample_size=sample_size)
        all_results['existing_results_evaluation'] = existing_results
    else:
        print("⚠ No se pudieron cargar datos de resultados para evaluación.")
    
    # 3. Crear visualizaciones si se solicita
    if create_charts and direct_results:
        print("\n[3/3] Creando visualizaciones...")
        chart_files = create_model_performance_charts(direct_results)
        all_results['charts'] = chart_files
    else:
        print("\n[3/3] Omitiendo creación de visualizaciones.")
    
    # Guardar resultados completos
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f"full_evaluation_{timestamp}.json")
    
    try:
        # Filtrar tipos no serializables
        all_results_filtered = {}
        for key, value in all_results.items():
            if key != 'charts':  # No serializar rutas de archivos
                all_results_filtered[key] = value
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results_filtered, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Resultados completos guardados en: {result_file}")
    except Exception as e:
        print(f"\n⚠ Error al guardar resultados completos: {e}")
    
    print("\n" + "=" * 70)
    print(" EVALUACIÓN COMPLETA FINALIZADA ")
    print("=" * 70)
    
    return all_results


def main():
    """Función principal del script"""
    print("\n📊 HERRAMIENTA DE EVALUACIÓN DE MODELOS NLP 📊\n")
    print("Esta herramienta evalúa el rendimiento de los modelos NLP (BERT y NER) utilizados")
    print("en el análisis de transcripciones de TikTok y la detección de productos cosméticos.\n")
    
    # Opciones
    print("Seleccione una opción:")
    print("1. Ejecutar evaluación completa")
    print("2. Evaluar solo modelos (muestra aleatoria)")
    print("3. Evaluar solo resultados existentes")
    print("4. Crear solo visualizaciones")
    print("5. Salir")
    
    try:
        option = int(input("\nOpción (1-5): "))
        
        if option == 1:
            # Personalización
            sample_size = int(input("Tamaño de muestra (10-50): ") or "20")
            sample_size = max(10, min(50, sample_size))  # Limitar entre 10 y 50
            
            create_charts = input("¿Crear gráficos? (s/n): ").lower() in ('s', 'si', 'yes', 'y', '')
            
            # Ejecutar
            run_full_evaluation(sample_size=sample_size, create_charts=create_charts)
        
        elif option == 2:
            sample_size = int(input("Tamaño de muestra (5-30): ") or "10")
            sample_size = max(5, min(30, sample_size))  # Limitar entre 5 y 30
            
            combined_model_evaluation(sample_size=sample_size)
        
        elif option == 3:
            results_file = input("Ruta al archivo de resultados (Enter para predeterminado): ") or None
            sample_size = int(input("Tamaño de muestra (20-100): ") or "50")
            sample_size = max(20, min(100, sample_size))  # Limitar entre 20 y 100
            
            results_df = load_results_data(results_file)
            if results_df is not None:
                evaluate_nlp_models_on_existing_results(results_df, sample_size=sample_size)
            else:
                print("⚠ No se pudieron cargar datos de resultados.")
        
        elif option == 4:
            # Primero, evaluar modelos para obtener datos
            print("Evaluando modelos para obtener datos para gráficos...")
            eval_results = combined_model_evaluation(sample_size=10)
            
            if eval_results:
                create_model_performance_charts(eval_results)
            else:
                print("⚠ No se pudieron obtener datos para crear gráficos.")
        
        elif option == 5:
            print("\nSaliendo...")
            return
        
        else:
            print("\n⚠ Opción no válida.")
    
    except ValueError:
        print("\n⚠ Por favor, ingrese un número válido.")
    except KeyboardInterrupt:
        print("\n\nEvaluación cancelada por el usuario.")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        traceback.print_exc()
    
    print("\n¡Evaluación completada!")


if __name__ == "__main__":
    main()