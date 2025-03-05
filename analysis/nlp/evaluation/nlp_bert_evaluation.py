# evaluar_resultados_nlp.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from collections import Counter

# DIRECTORY SETUP
base_dir = 'C:/Users/sandr/Documents/scrp_tiktok_tfg'
analysis_dir = os.path.join(base_dir, 'analysis/nlp')

# Create evaluation output directory if needed
evaluation_dir = os.path.join(analysis_dir, 'evaluation')
if not os.path.exists(evaluation_dir):
    os.makedirs(evaluation_dir)
    print(f"Directorio creado: {evaluation_dir}")

# FILE PATHS
results_file = os.path.join(analysis_dir, 'product_mentions_analysis.xlsx')
evaluation_output = os.path.join(evaluation_dir, 'nlp_bert_evresults.xlsx')

# Verify results file exists
if not os.path.exists(results_file):
    raise FileNotFoundError(f"Archivo de resultados NLP no encontrado: {results_file}")

def load_nlp_results():
    """
    Carga los resultados del análisis NLP previamente ejecutado
    
    Returns:
        pd.DataFrame: DataFrame con los resultados del análisis
    """
    print(f"Cargando resultados de análisis NLP desde {results_file}...")
    
    try:
        # Intentar cargar desde Excel
        results_df = pd.read_excel(results_file, sheet_name='Resultados')
        print(f"Datos cargados exitosamente. {len(results_df)} registros encontrados.")
        return results_df
    except Exception as e:
        print(f"Error al cargar desde Excel: {e}")
        # Intentar cargar como CSV como fallback
        csv_file = results_file.replace('.xlsx', '.csv')
        if os.path.exists(csv_file):
            try:
                results_df = pd.read_csv(csv_file)
                print(f"Datos cargados desde CSV. {len(results_df)} registros encontrados.")
                return results_df
            except Exception as csv_e:
                print(f"Error al cargar desde CSV: {csv_e}")
        
        raise FileNotFoundError(f"No se pudo cargar el archivo de resultados NLP en ningún formato")

def parse_categories(categories_str):
    """
    Convierte las categorías desde formato string JSON a lista
    
    Args:
        categories_str: String con formato JSON o lista
        
    Returns:
        list: Lista de categorías
    """
    if pd.isna(categories_str) or categories_str == "":
        return []
    
    if isinstance(categories_str, list):
        return categories_str
    
    try:
        # Intentar parsear como JSON
        return json.loads(categories_str)
    except:
        # Si falla, intentar parsear como string literal
        if categories_str.startswith('[') and categories_str.endswith(']'):
            # Eliminar corchetes y dividir por comas
            categories_str = categories_str[1:-1]
            return [cat.strip().strip("'").strip('"') for cat in categories_str.split(',')]
        
        # Último intento: dividir por comas
        return [cat.strip().strip("'").strip('"') for cat in categories_str.split(',')]

def evaluate_sentiment_analysis(results_df):
    """
    Evalúa los resultados del análisis de sentimiento
    
    Args:
        results_df (pd.DataFrame): DataFrame con los resultados
        
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    print("Evaluando resultados del análisis de sentimiento...")
    
    sentiment_metrics = {}
    
    # Verificar que existan columnas de sentimiento
    sentiment_cols = [col for col in results_df.columns if 'sentiment' in col.lower()]
    if not sentiment_cols:
        print("No se encontraron columnas relacionadas con análisis de sentimiento")
        return sentiment_metrics
    
    # Evaluar sentimiento a nivel de oración
    if 'sentiment' in results_df.columns:
        # Distribución de sentimientos
        sentiment_dist = results_df['sentiment'].value_counts().to_dict()
        sentiment_metrics['sentence_sentiment_distribution'] = sentiment_dist
        
        # Calcular score promedio por sentimiento
        if 'sentiment_score' in results_df.columns:
            sentiment_scores = {}
            for sentiment in sentiment_dist.keys():
                avg_score = results_df[results_df['sentiment'] == sentiment]['sentiment_score'].mean()
                sentiment_scores[sentiment] = avg_score
            sentiment_metrics['avg_sentiment_scores'] = sentiment_scores
            
            # Calcular estadísticas adicionales sobre los scores de confianza del modelo
            confidence_stats = {
                'mean': results_df['sentiment_score'].mean(),
                'median': results_df['sentiment_score'].median(),
                'std': results_df['sentiment_score'].std(),
                'min': results_df['sentiment_score'].min(),
                'max': results_df['sentiment_score'].max(),
                'q1': results_df['sentiment_score'].quantile(0.25),
                'q3': results_df['sentiment_score'].quantile(0.75)
            }
            sentiment_metrics['confidence_stats'] = confidence_stats
            
            # Calcular distribución de confianza por intervalos
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            confidence_bins = pd.cut(results_df['sentiment_score'], bins=bins, labels=bin_labels)
            confidence_dist = confidence_bins.value_counts().to_dict()
            sentiment_metrics['confidence_distribution'] = confidence_dist
            
            # Calcular tasa de alta confianza (scores > 0.8)
            high_confidence = (results_df['sentiment_score'] > 0.8).mean() * 100
            sentiment_metrics['high_confidence_rate'] = high_confidence
            
            # Calcular tasa de baja confianza (scores < 0.4)
            low_confidence = (results_df['sentiment_score'] < 0.4).mean() * 100
            sentiment_metrics['low_confidence_rate'] = low_confidence
    
    # Evaluar sentimiento general
    if 'overall_sentiment' in results_df.columns:
        # Distribución de sentimientos generales
        overall_dist = results_df['overall_sentiment'].value_counts().to_dict()
        sentiment_metrics['overall_sentiment_distribution'] = overall_dist
        
        # Calcular score promedio por sentimiento general
        if 'overall_sentiment_score' in results_df.columns:
            overall_scores = {}
            for sentiment in overall_dist.keys():
                avg_score = results_df[results_df['overall_sentiment'] == sentiment]['overall_sentiment_score'].mean()
                overall_scores[sentiment] = avg_score
            sentiment_metrics['avg_overall_sentiment_scores'] = overall_scores
            
            # Estadísticas sobre estabilidad del sentimiento en el mismo video
            if 'video_id' in results_df.columns:
                video_sentiment_variance = []
                for video_id in results_df['video_id'].unique():
                    video_df = results_df[results_df['video_id'] == video_id]
                    if len(video_df) > 1 and 'sentiment' in video_df.columns:
                        # Verificar si hay múltiples sentimientos en el mismo video
                        unique_sentiments = video_df['sentiment'].nunique()
                        if unique_sentiments > 1:
                            video_sentiment_variance.append(video_id)
                
                sentiment_metrics['videos_with_mixed_sentiment'] = len(video_sentiment_variance)
                sentiment_metrics['mixed_sentiment_rate'] = len(video_sentiment_variance) / len(results_df['video_id'].unique()) * 100
    
    # Evaluar concordancia entre sentimiento de oración y sentimiento general
    if 'sentiment' in results_df.columns and 'overall_sentiment' in results_df.columns:
        # Crear matriz de concordancia
        concordance_matrix = pd.crosstab(
            results_df['sentiment'],
            results_df['overall_sentiment'],
            normalize='index'
        )
        sentiment_metrics['sentiment_concordance'] = concordance_matrix
        
        # Calcular tasa de concordancia general
        concordance = (results_df['sentiment'] == results_df['overall_sentiment']).mean() * 100
        sentiment_metrics['sentiment_concordance_rate'] = concordance
    
    # Si hay sentimiento por marcas, evaluar la variabilidad de sentimiento por marca
    if 'brand' in results_df.columns and 'sentiment' in results_df.columns:
        brand_sentiment_consistency = {}
        top_brands = results_df['brand'].value_counts().head(10).index.tolist()
        
        for brand in top_brands:
            brand_df = results_df[results_df['brand'] == brand]
            if len(brand_df) > 5:  # Solo evaluar marcas con suficientes menciones
                # Calcular entropy de la distribución de sentimiento para esta marca
                sentiment_counts = brand_df['sentiment'].value_counts(normalize=True)
                entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in sentiment_counts)
                brand_sentiment_consistency[brand] = {
                    'entropy': entropy,
                    'unique_sentiments': brand_df['sentiment'].nunique(),
                    'dominant_sentiment': brand_df['sentiment'].mode()[0],
                    'dominant_sentiment_rate': brand_df['sentiment'].value_counts(normalize=True).max() * 100
                }
        
        sentiment_metrics['brand_sentiment_consistency'] = brand_sentiment_consistency
    
    return sentiment_metrics

def evaluate_product_detection(results_df):
    """
    Evalúa los resultados de la detección de productos
    
    Args:
        results_df (pd.DataFrame): DataFrame con los resultados
        
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    print("Evaluando resultados de detección de productos...")
    
    detection_metrics = {}
    
    # Verificar si existe columna detection_type
    if 'detection_type' not in results_df.columns:
        print("No se encontró columna 'detection_type' para evaluar detección de productos")
        return detection_metrics
    
    # Distribución de tipos de detección
    detection_dist = results_df['detection_type'].value_counts().to_dict()
    detection_metrics['detection_type_distribution'] = detection_dist
    
    # Porcentaje de videos con detecciones
    video_ids = results_df['video_id'].unique()
    videos_with_detection = results_df[results_df['detection_type'] != 'no_detection']['video_id'].unique()
    detection_rate = len(videos_with_detection) / len(video_ids) if len(video_ids) > 0 else 0
    detection_metrics['video_detection_rate'] = detection_rate
    
    # Evaluar métodos de detección
    if 'method' in results_df.columns:
        method_dist = results_df['method'].value_counts().to_dict()
        detection_metrics['detection_method_distribution'] = method_dist
    
    # Evaluar scores de detección
    if 'score' in results_df.columns:
        score_by_method = {}
        all_methods = results_df['method'].unique()
        for method in all_methods:
            if method != 'none':
                avg_score = results_df[results_df['method'] == method]['score'].mean()
                score_by_method[method] = avg_score
        detection_metrics['avg_detection_score_by_method'] = score_by_method
    
    # Análisis de detecciones "product_only" (si existe esta categoría)
    if 'product_only' in detection_dist:
        product_only_df = results_df[results_df['detection_type'] == 'product_only']
        
        # Contar productos detectados sin marca
        if 'product' in results_df.columns:
            product_counts = product_only_df['product'].value_counts().head(20).to_dict()
            detection_metrics['top_products_without_brand'] = product_counts
        
        # Analizar sentimiento para productos sin marca
        if 'sentiment' in results_df.columns and 'product' in results_df.columns:
            sentiment_by_product = {}
            top_products = list(product_counts.keys())[:10] if 'product' in results_df.columns else []
            
            for product in top_products:
                product_sentiments = product_only_df[product_only_df['product'] == product]['sentiment'].value_counts(normalize=True).to_dict()
                sentiment_by_product[product] = product_sentiments
            
            detection_metrics['sentiment_by_product'] = sentiment_by_product
    
    return detection_metrics

def evaluate_brand_category_detection(results_df):
    """
    Evalúa los resultados de la detección de marcas y categorías
    
    Args:
        results_df (pd.DataFrame): DataFrame con los resultados
        
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    print("Evaluando resultados de detección de marcas y categorías...")
    
    brand_cat_metrics = {}
    
    # Verificar columnas necesarias
    has_brand = 'brand' in results_df.columns
    has_categories = 'categories' in results_df.columns
    
    if not has_brand and not has_categories:
        print("No se encontraron columnas de marcas ni categorías")
        return brand_cat_metrics
    
    # Análisis de marcas
    if has_brand:
        # Top marcas mencionadas
        brand_mentions = results_df['brand'].dropna().value_counts().head(20).to_dict()
        brand_cat_metrics['top_brands'] = brand_mentions
        
        # Sentimiento por marca (si hay datos de sentimiento)
        if 'sentiment' in results_df.columns:
            brand_sentiment = {}
            top_brands = list(brand_mentions.keys())[:10]  # Analizar solo las 10 principales marcas
            
            for brand in top_brands:
                brand_df = results_df[results_df['brand'] == brand]
                if not brand_df.empty:
                    sentiment_dist = brand_df['sentiment'].value_counts(normalize=True).to_dict()
                    brand_sentiment[brand] = sentiment_dist
            
            brand_cat_metrics['brand_sentiment'] = brand_sentiment
    
    # Análisis de categorías
    if has_categories:
        # Procesar categorías JSON a listas
        if results_df['categories'].dtype == 'object':
            # Convertir columna categories a listas Python
            results_df['categories_list'] = results_df['categories'].apply(parse_categories)
        else:
            results_df['categories_list'] = results_df['categories']
        
        # Obtener conteo de categorías
        all_categories = []
        for cat_list in results_df['categories_list'].dropna():
            if isinstance(cat_list, list):
                all_categories.extend(cat_list)
        
        category_counts = Counter(all_categories)
        top_categories = dict(category_counts.most_common(20))
        brand_cat_metrics['top_categories'] = top_categories
        
        # Análisis de co-ocurrencia de categorías
        if len(category_counts) > 1:
            category_pairs = []
            for cat_list in results_df['categories_list'].dropna():
                if isinstance(cat_list, list) and len(cat_list) > 1:
                    # Generar todos los pares posibles
                    pairs = [(a, b) for idx, a in enumerate(cat_list) for b in cat_list[idx+1:]]
                    category_pairs.extend(pairs)
            
            pair_counts = Counter(category_pairs)
            top_pairs = dict(pair_counts.most_common(15))
            brand_cat_metrics['top_category_pairs'] = top_pairs
    
    # Análisis de combinaciones marca-categoría
    if has_brand and has_categories:
        brand_category_combos = []
        
        for _, row in results_df.dropna(subset=['brand', 'categories_list']).iterrows():
            brand = row['brand']
            for category in row['categories_list']:
                brand_category_combos.append((brand, category))
        
        combo_counts = Counter(brand_category_combos)
        top_combos = dict(combo_counts.most_common(20))
        
        # Convertir las tuplas a strings para facilitar exportación
        top_combos_str = {f"{brand} - {cat}": count for (brand, cat), count in top_combos.items()}
        brand_cat_metrics['top_brand_category_combos'] = top_combos_str
    
    return brand_cat_metrics

def generate_videos_summary(results_df):
    """
    Genera un resumen de los videos con productos y marcas detectados
    
    Args:
        results_df (pd.DataFrame): DataFrame con los resultados del análisis
        
    Returns:
        pd.DataFrame: DataFrame con el resumen de videos
    """
    print("Generando resumen de videos con productos y marcas detectados...")
    
    # Verificar columnas necesarias
    if 'video_id' not in results_df.columns:
        print("No se encontró columna 'video_id' para generar resumen de videos")
        return pd.DataFrame()
    
    # Obtener IDs de videos únicos
    video_ids = results_df['video_id'].unique()
    
    # Preparar DataFrame de resumen
    videos_summary = []
    
    for video_id in video_ids:
        video_data = results_df[results_df['video_id'] == video_id]
        
        # Usar id_urlvideo directamente como identificador único
        if 'id_urlvideo' in video_data.columns:
            video_url = video_data['id_urlvideo'].iloc[0]
            
            # Preparar datos básicos
            video_summary = {
                'video_url': video_url,
            }
            
            # Recopilar productos detectados
            if 'product' in video_data.columns:
                products = video_data['product'].dropna().unique()
                video_summary['detected_products'] = ', '.join(products) if len(products) > 0 else ''
                video_summary['product_count'] = len(products)
            else:
                video_summary['detected_products'] = ''
                video_summary['product_count'] = 0
            
            # Recopilar marcas detectadas
            if 'brand' in video_data.columns:
                brands = video_data['brand'].dropna().unique()
                video_summary['detected_brands'] = ', '.join(brands) if len(brands) > 0 else ''
                video_summary['brand_count'] = len(brands)
            else:
                video_summary['detected_brands'] = ''
                video_summary['brand_count'] = 0
            
            # Recopilar categorías detectadas
            if 'categories_list' in video_data.columns:
                all_categories = []
                for cat_list in video_data['categories_list'].dropna():
                    if isinstance(cat_list, list):
                        all_categories.extend(cat_list)
                # Eliminar duplicados y ordenar
                unique_categories = sorted(list(set(all_categories)))
                video_summary['detected_categories'] = ', '.join(unique_categories) if len(unique_categories) > 0 else ''
                video_summary['category_count'] = len(unique_categories)
            elif 'categories' in video_data.columns:
                # Intentar obtener categorías de columna 'categories' si no existe 'categories_list'
                all_categories = []
                for cats in video_data['categories'].dropna():
                    parsed_cats = parse_categories(cats)
                    all_categories.extend(parsed_cats)
                unique_categories = sorted(list(set(all_categories)))
                video_summary['detected_categories'] = ', '.join(unique_categories) if len(unique_categories) > 0 else ''
                video_summary['category_count'] = len(unique_categories)
            else:
                video_summary['detected_categories'] = ''
                video_summary['category_count'] = 0
                
            # Obtener sentimiento general si está disponible
            if 'overall_sentiment' in video_data.columns:
                # Tomar el sentimiento más frecuente
                video_summary['sentiment'] = video_data['overall_sentiment'].mode().iloc[0] if not video_data['overall_sentiment'].isna().all() else 'Unknown'
            elif 'sentiment' in video_data.columns:
                video_summary['sentiment'] = video_data['sentiment'].mode().iloc[0] if not video_data['sentiment'].isna().all() else 'Unknown'
            
            videos_summary.append(video_summary)
    
    # Convertir a DataFrame
    videos_df = pd.DataFrame(videos_summary)
    
    # Ordenar por número de productos y marcas detectados (descendente)
    if 'product_count' in videos_df.columns and 'brand_count' in videos_df.columns:
        videos_df = videos_df.sort_values(by=['product_count', 'brand_count'], ascending=False)
    
    return videos_df

def generate_evaluation_report(results_df, sentiment_metrics, detection_metrics, brand_cat_metrics, output_file=None):
    """
    Genera reporte de evaluación en Excel
    
    Args:
        results_df (pd.DataFrame): DataFrame con los resultados
        sentiment_metrics (dict): Métricas de evaluación de sentimiento
        detection_metrics (dict): Métricas de evaluación de detección de productos
        brand_cat_metrics (dict): Métricas de evaluación de marcas y categorías
        output_file (str, optional): Ruta de archivo de salida. Si es None, usa evaluation_output
    """
    if output_file is None:
        output_file = evaluation_output
        
    print(f"Generando reporte de evaluación en {output_file}...")
    
    try:
        with pd.ExcelWriter(evaluation_output, engine='openpyxl') as writer:
            # 1. Hoja con resumen ejecutivo
            summary_data = []
            
            # Añadir métricas generales
            summary_data.append(["RESUMEN GENERAL", ""])
            summary_data.append(["Total de registros", len(results_df)])
            summary_data.append(["Videos únicos", len(results_df['video_id'].unique())])
            
            if 'detection_type' in results_df.columns:
                detection_counts = results_df['detection_type'].value_counts()
                valid_detections = sum(detection_counts[dt] for dt in detection_counts.index if dt != 'no_detection')
                summary_data.append(["Detecciones válidas", valid_detections])
                summary_data.append(["Tasa de detección (%)", f"{detection_metrics.get('video_detection_rate', 0)*100:.1f}%"])
            
            # Añadir top brands
            if 'top_brands' in brand_cat_metrics:
                summary_data.append(["", ""])
                summary_data.append(["TOP 5 MARCAS", "Menciones"])
                for i, (brand, count) in enumerate(list(brand_cat_metrics['top_brands'].items())[:5]):
                    summary_data.append([brand, count])
            
            # Añadir distribución de sentimiento
            if 'overall_sentiment_distribution' in sentiment_metrics:
                summary_data.append(["", ""])
                summary_data.append(["DISTRIBUCIÓN SENTIMIENTO", "Cantidad"])
                for sentiment, count in sentiment_metrics['overall_sentiment_distribution'].items():
                    summary_data.append([sentiment, count])
            
            # Crear DataFrame y guardar
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen_Ejecutivo', header=False, index=False)
            
            # 2. Hoja con análisis de sentimiento
            if sentiment_metrics:
                sentiment_data = []
                
                # Distribución de sentimiento a nivel oración
                if 'sentence_sentiment_distribution' in sentiment_metrics:
                    sentiment_data.append(["SENTIMIENTO (NIVEL ORACIÓN)", "Cantidad"])
                    for sentiment, count in sentiment_metrics['sentence_sentiment_distribution'].items():
                        sentiment_data.append([sentiment, count])
                
                # Scores promedio por sentimiento
                if 'avg_sentiment_scores' in sentiment_metrics:
                    sentiment_data.append(["", ""])
                    sentiment_data.append(["SCORE PROMEDIO POR SENTIMIENTO", "Score"])
                    for sentiment, score in sentiment_metrics['avg_sentiment_scores'].items():
                        sentiment_data.append([sentiment, f"{score:.4f}"])
                
                # Distribución de sentimiento general
                if 'overall_sentiment_distribution' in sentiment_metrics:
                    sentiment_data.append(["", ""])
                    sentiment_data.append(["SENTIMIENTO GENERAL", "Cantidad"])
                    for sentiment, count in sentiment_metrics['overall_sentiment_distribution'].items():
                        sentiment_data.append([sentiment, count])
                
                # Scores promedio por sentimiento general
                if 'avg_overall_sentiment_scores' in sentiment_metrics:
                    sentiment_data.append(["", ""])
                    sentiment_data.append(["SCORE PROMEDIO SENTIMIENTO GENERAL", "Score"])
                    for sentiment, score in sentiment_metrics['avg_overall_sentiment_scores'].items():
                        sentiment_data.append([sentiment, f"{score:.4f}"])
                
                # Crear DataFrame y guardar
                if sentiment_data:
                    sentiment_df = pd.DataFrame(sentiment_data)
                    sentiment_df.to_excel(writer, sheet_name='Análisis_Sentimiento', header=False, index=False)
                
                # Matriz de concordancia si existe
                if 'sentiment_concordance' in sentiment_metrics:
                    sentiment_metrics['sentiment_concordance'].to_excel(writer, sheet_name='Concordancia_Sentimiento')
            
            # 3. Hoja con análisis de detección
            if detection_metrics:
                detection_data = []
                
                # Distribución de tipos de detección
                if 'detection_type_distribution' in detection_metrics:
                    detection_data.append(["TIPO DE DETECCIÓN", "Cantidad"])
                    for det_type, count in detection_metrics['detection_type_distribution'].items():
                        detection_data.append([det_type, count])
                
                # Distribución de métodos de detección
                if 'detection_method_distribution' in detection_metrics:
                    detection_data.append(["", ""])
                    detection_data.append(["MÉTODO DE DETECCIÓN", "Cantidad"])
                    for method, count in detection_metrics['detection_method_distribution'].items():
                        detection_data.append([method, count])
                
                # Scores promedio por método
                if 'avg_detection_score_by_method' in detection_metrics:
                    detection_data.append(["", ""])
                    detection_data.append(["SCORE PROMEDIO POR MÉTODO", "Score"])
                    for method, score in detection_metrics['avg_detection_score_by_method'].items():
                        detection_data.append([method, f"{score:.4f}"])
                
                # Crear DataFrame y guardar
                if detection_data:
                    detection_df = pd.DataFrame(detection_data)
                    detection_df.to_excel(writer, sheet_name='Análisis_Detección', header=False, index=False)
                
                # Análisis de productos "product_only" si existen
                if 'top_products_without_brand' in detection_metrics:
                    product_only_df = pd.DataFrame(
                        list(detection_metrics['top_products_without_brand'].items()),
                        columns=['Producto (sin marca)', 'Menciones']
                    )
                    product_only_df.to_excel(writer, sheet_name='Productos_Sin_Marca', index=False)
            
            # 4. Hoja con análisis de marcas y categorías
            if brand_cat_metrics:
                # TOP marcas
                if 'top_brands' in brand_cat_metrics:
                    brands_df = pd.DataFrame(
                        list(brand_cat_metrics['top_brands'].items()),
                        columns=['Marca', 'Menciones']
                    )
                    brands_df.to_excel(writer, sheet_name='Top_Marcas', index=False)
                
                # TOP categorías
                if 'top_categories' in brand_cat_metrics:
                    categories_df = pd.DataFrame(
                        list(brand_cat_metrics['top_categories'].items()),
                        columns=['Categoría', 'Menciones']
                    )
                    categories_df.to_excel(writer, sheet_name='Top_Categorías', index=False)
                
                # TOP combinaciones marca-categoría
                if 'top_brand_category_combos' in brand_cat_metrics:
                    combos_df = pd.DataFrame(
                        list(brand_cat_metrics['top_brand_category_combos'].items()),
                        columns=['Combinación', 'Menciones']
                    )
                    combos_df.to_excel(writer, sheet_name='Top_Combinaciones', index=False)
        
        print(f"Reporte de evaluación generado exitosamente en {evaluation_output}")
        
    except Exception as e:
        print(f"Error al generar reporte: {e}")
        import traceback
        traceback.print_exc()
        
        # Intentar guardar en CSV como fallback
        csv_output = evaluation_output.replace('.xlsx', '.csv')
        try:
            # Guardar al menos el resumen ejecutivo
            pd.DataFrame(summary_data).to_csv(csv_output, index=False, header=False)
            print(f"Se guardó un resumen básico en CSV: {csv_output}")
        except:
            print("No se pudo guardar ni siquiera el resumen básico en CSV")

def main():
    """Función principal de evaluación"""
    try:
        # 1. Cargar resultados del análisis NLP
        results_df = load_nlp_results()
        
        if results_df.empty:
            print("No hay datos para evaluar. Finalizando.")
            return
        
        # 2. Realizar evaluaciones
        sentiment_metrics = evaluate_sentiment_analysis(results_df)
        detection_metrics = evaluate_product_detection(results_df)
        brand_cat_metrics = evaluate_brand_category_detection(results_df)
        
        # 3. Generar resumen de videos
        # Verificar si necesitamos procesar categorías primero
        if 'categories' in results_df.columns and 'categories_list' not in results_df.columns:
            results_df['categories_list'] = results_df['categories'].apply(parse_categories)
        
        videos_summary_df = generate_videos_summary(results_df)
        
        # 4. Comprobar si el archivo de evaluación ya existe
        file_exists = os.path.exists(evaluation_output) and os.path.getsize(evaluation_output) > 0
        
        if file_exists:
            print(f"El archivo de evaluación {evaluation_output} ya existe. Creando versión actualizada...")
            # Crear un nombre para el nuevo archivo con timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            new_output_file = evaluation_output.replace('.xlsx', f'_{timestamp}.xlsx')
            
            # Generar el reporte en el nuevo archivo
            generate_evaluation_report(results_df, sentiment_metrics, detection_metrics, brand_cat_metrics, new_output_file)
            
            # Añadir resumen de videos
            if not videos_summary_df.empty:
                try:
                    with pd.ExcelWriter(new_output_file, mode='a', engine='openpyxl') as writer:
                        videos_summary_df.to_excel(writer, sheet_name='Resumen_Videos', index=False)
                    print(f"Resumen de videos añadido al archivo {new_output_file}")
                except Exception as e:
                    print(f"Error al añadir resumen de videos: {e}")
                    # Guardar en archivo separado como fallback
                    videos_summary_file = os.path.join(evaluation_dir, f'videos_summary_{timestamp}.xlsx')
                    videos_summary_df.to_excel(videos_summary_file, index=False)
                    print(f"Resumen de videos guardado en archivo separado: {videos_summary_file}")
            
            print(f"Evaluación actualizada guardada en: {new_output_file}")
        else:
            # Generar reporte en el archivo original
            generate_evaluation_report(results_df, sentiment_metrics, detection_metrics, brand_cat_metrics, evaluation_output)
            
            # Añadir resumen de videos
            if not videos_summary_df.empty:
                try:
                    with pd.ExcelWriter(evaluation_output, mode='a', engine='openpyxl') as writer:
                        videos_summary_df.to_excel(writer, sheet_name='Resumen_Videos', index=False)
                    print(f"Resumen de videos añadido al archivo {evaluation_output}")
                except Exception as e:
                    print(f"Error al añadir resumen de videos: {e}")
                    # Guardar en archivo separado como fallback
                    videos_summary_file = os.path.join(evaluation_dir, 'videos_summary.xlsx')
                    videos_summary_df.to_excel(videos_summary_file, index=False)
                    print(f"Resumen de videos guardado en archivo separado: {videos_summary_file}")
        
        print("===== EVALUACIÓN COMPLETADA =====")
        
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()