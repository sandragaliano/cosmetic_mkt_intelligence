import pandas as pd
import numpy as np
import os
import re
import ast
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# Define file paths
SEPHORA_PATH = r"C:\Users\sandr\Documents\scrp_tiktok_tfg\data\clean_data\sephora_website_cleaned.csv"
TIKTOK_PATH = r"C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\nlp\products and brands detection\resultados_finales.xlsx"
OUTPUT_DIR = r"C:\Users\sandr\Documents\scrp_tiktok_tfg\analysis\nlp\Brand_characteristics"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "brand_summary.xlsx")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Define brand mappings for normalization
BRAND_MAPPING = {
    'FENTY': 'FENTY BEAUTY',
    'FENTY BEAUTY BY RIHANNA': 'FENTY BEAUTY',
    'RARE BEAUTY': 'RARE BEAUTY BY SELENA GOMEZ',
    'RARE BEAUTY BY SELENA GOMEZ': 'RARE BEAUTY',
    'CHARLOTTE': 'CHARLOTTE TILBURY',
    'DRUNK': 'DRUNK ELEPHANT',
    'BENEFIT': 'BENEFIT COSMETICS',
    'ISLE OF PARADISE': 'ISLE',
    'HOURGLASS': 'HOURGLASS COSMETICS',
    'PAT MCGRATH': 'PAT MCGRATH LABS',
    'MILK': 'MILK MAKEUP',
    'MAKEUP BY MARIO': 'MARIO',
    'SUMMER FRIDAYS': 'SUMMER',
    'HUDA': 'HUDA BEAUTY',
    'TOWER 28': 'TOWER28',
    'TOWER 28 BEAUTY': 'TOWER28'
}

def normalize_brand_name(brand):
    """Normalize brand names for better matching"""
    if not brand or pd.isna(brand):
        return ''
    
    brand_upper = str(brand).strip().upper()
    
    # Apply direct mappings
    if brand_upper in BRAND_MAPPING:
        return BRAND_MAPPING[brand_upper]
    
    # Check for partial matches in keys
    for key in BRAND_MAPPING:
        if key in brand_upper:
            return BRAND_MAPPING[key]
        elif brand_upper in key:
            return BRAND_MAPPING[key]
    
    return brand_upper

def load_data():
    """Load Sephora and TikTok data"""
    print("Loading data files...")
    
    # Load Sephora data
    sephora_df = pd.read_csv(SEPHORA_PATH)
    print(f"Loaded Sephora data: {sephora_df.shape[0]} rows, {sephora_df.shape[1]} columns")
    
    # Print column names for debugging
    print(f"Sephora columns: {', '.join(sephora_df.columns)}")
    
    # Load TikTok data
    tiktok_df = pd.read_excel(TIKTOK_PATH)
    print(f"Loaded TikTok data: {tiktok_df.shape[0]} rows, {tiktok_df.shape[1]} columns")
    
    # Print column names for debugging
    print(f"TikTok columns: {', '.join(tiktok_df.columns)}")
    
    return sephora_df, tiktok_df

def preprocess_data(sephora_df, tiktok_df):
    """Preprocess both datasets"""
    print("Preprocessing data...")
    
    # Preprocess Sephora data
    # Normalize brand names
    sephora_df['brand_normalized'] = sephora_df['brand'].apply(normalize_brand_name)
    
    # Combine "What it is" and "What else you need to know" columns
    what_it_is_col = None
    what_else_col = None
    
    # Identify column names - try different variations
    for col in sephora_df.columns:
        if col == 'What it is' or col == 'whatItIs' or col == 'what_it_is' or 'what it is' in col.lower():
            what_it_is_col = col
            print(f"Found 'What it is' column: {col}")
        elif 'what else' in col.lower() or 'need to know' in col.lower():
            what_else_col = col
            print(f"Found 'What else' column: {col}")
    
    # Create combined description column
    sephora_df['full_description'] = ''
    
    if what_it_is_col:
        sephora_df['full_description'] += sephora_df[what_it_is_col].fillna('').astype(str)
    
    if what_else_col:
        sephora_df['full_description'] += ' ' + sephora_df[what_else_col].fillna('').astype(str)
    
    # Find price column - prioritize current price
    price_col = None
    
    if 'currency' in sephora_df.columns:
        price_cols = [col for col in sephora_df.columns if 'price' in col.lower()]
        if price_cols:
            price_col = price_cols[0]
            print(f"Found price column: {price_col}")
    
    if not price_col and 'id' in sephora_df.columns:
        price_col = 'id'  # Fallback to id
        print("Using 'id' as price column")
    
    # Convert price to numeric
    if price_col:
        sephora_df[f'{price_col}_numeric'] = pd.to_numeric(sephora_df[price_col], errors='coerce')
    else:
        # If no price column found, add a placeholder
        print("No price column found, using placeholder")
        sephora_df['price_numeric'] = np.nan
        price_col = 'price'
    
    # Check for lovesCount column
    if 'lovesCount' not in sephora_df.columns and 'loves' in ' '.join(sephora_df.columns).lower():
        # Try to find a column that might be loves count
        for col in sephora_df.columns:
            if 'love' in col.lower():
                print(f"Found potential Loves Count column: {col}")
                sephora_df['lovesCount'] = sephora_df[col]
                break
    
    if 'lovesCount' in sephora_df.columns:
        print(f"LovesCount column found with data types: {sephora_df['lovesCount'].dtype}")
        # Ensure lovesCount is numeric
        sephora_df['lovesCount'] = pd.to_numeric(sephora_df['lovesCount'], errors='coerce')
    
    # Preprocess TikTok data
    # Identify column with brand mentions
    brand_col = None
    sentence_col = None
    
    # Check common names
    if 'mentioned_brands' in tiktok_df.columns:
        brand_col = 'mentioned_brands'
    elif 'C' in tiktok_df.columns:
        brand_col = 'C'
    
    if 'sentence' in tiktok_df.columns:
        sentence_col = 'sentence'
    elif 'B' in tiktok_df.columns:
        sentence_col = 'B'
    
    print(f"Found brand column: {brand_col}, sentence column: {sentence_col}")
    
    # Ensure brand_col and sentence_col are not None
    if not brand_col or not sentence_col:
        # Try to identify by looking at content
        for col in tiktok_df.columns:
            if tiktok_df[col].astype(str).str.contains(r'\[|\]').any():
                brand_col = col
                print(f"Identified brand column by content: {col}")
            elif tiktok_df[col].astype(str).str.len().mean() > 30:
                sentence_col = col
                print(f"Identified sentence column by content: {col}")
    
    # Parse brand mentions and normalize names
    if brand_col:
        # Function to safely parse brand mentions
        def parse_brands(x):
            if isinstance(x, list):
                return [normalize_brand_name(b) for b in x]
            elif isinstance(x, str):
                try:
                    brands = ast.literal_eval(x)
                    return [normalize_brand_name(b) for b in brands]
                except:
                    # Try regex if literal_eval fails
                    brands = re.findall(r'[\'"]([^\'"]*)[\'"]', x)
                    return [normalize_brand_name(b) for b in brands]
            return []
        
        tiktok_df['brands_parsed'] = tiktok_df[brand_col].apply(parse_brands)
    else:
        tiktok_df['brands_parsed'] = [[]]
    
    # Print unique brands for debugging
    all_brands = []
    for brands in tiktok_df['brands_parsed']:
        all_brands.extend(brands)
    
    print(f"Found {len(set(all_brands))} unique normalized brands in TikTok data")
    print(f"Top 10 brands: {', '.join(sorted(list(set(all_brands))))[:10]}")
    
    return sephora_df, tiktok_df, price_col, sentence_col

def calculate_brand_metrics(sephora_df, price_col):
    """Calculate metrics by brand from Sephora data"""
    print("Calculating brand metrics from Sephora...")
    
    brand_metrics = []
    
    for brand, group in sephora_df.groupby('brand_normalized'):
        # Skip empty brand names
        if not brand or pd.isna(brand) or brand == '':
            continue
        
        # Count products
        product_count = len(group)
        
        # Calculate average price
        price_numeric = group[f'{price_col}_numeric']
        avg_price = price_numeric.mean()
        
        # If average price is very high, divide by 100 (convert cents to dollars)
        if avg_price and avg_price > 1000:
            avg_price = avg_price / 100
        
        # Calculate average rating
        avg_rating = group['rating'].mean() if 'rating' in group.columns else None
        
        # Calculate total and average loves
        if 'lovesCount' in group.columns:
            # Ensure values are numeric
            loves_values = pd.to_numeric(group['lovesCount'], errors='coerce')
            total_loves = loves_values.sum()
            avg_loves = loves_values.mean()
        else:
            total_loves = avg_loves = None
        
        # Extract key characteristics
        characteristics = []
        word_counts = Counter()
        if 'full_description' in group.columns:
            all_desc = ' '.join(group['full_description'].fillna('').astype(str))
            
            # Remove common words and short words
            words = re.findall(r'\b[a-zA-Z]{4,}\b', all_desc.lower())
            stop_words = {'what', 'with', 'this', 'that', 'from', 'have', 'they', 'will', 
                         'your', 'skin', 'formula', 'product', 'using', 'used', 'like', 
                         'make', 'more', 'than', 'help', 'also', 'into', 'when', 'just',
                         'which', 'gives', 'gives', 'after', 'before', 'about', 'other',
                         'features', 'contains', 'contains', 'include', 'includes', 'including',
                         'apply', 'every', 'daily', 'week', 'month', 'year', 'time', 'times',
                         'back', 'next', 'last', 'first', 'second', 'third'}
            
            # Add brand name to stop words
            brand_words = re.findall(r'\b[a-zA-Z]{2,}\b', brand.lower())
            stop_words.update(brand_words)
            
            # Get the original brand name (with proper capitalization)
            original_brand = group['brand'].iloc[0]
            brand_parts = re.findall(r'\b[a-zA-Z]{2,}\b', str(original_brand).lower())
            stop_words.update(brand_parts)
            
            # Filter and count words
            filtered_words = [w for w in words if w not in stop_words]
            word_counts = Counter(filtered_words)
            characteristics = [word for word, count in word_counts.most_common(15)]
        
        # Store brand metrics
        brand_metrics.append({
            'brand': brand,
            'original_name': group['brand'].iloc[0],  # Keep original capitalization
            'product_count': product_count,
            'avg_price': avg_price,
            'avg_rating': avg_rating,
            'total_loves': total_loves,
            'avg_loves': avg_loves,
            'key_characteristics': characteristics,
            'characteristic_counts': {word: count for word, count in word_counts.most_common(50)} if word_counts else {}
        })
    
    # Convert to DataFrame
    brand_df = pd.DataFrame(brand_metrics)
    print(f"Calculated metrics for {len(brand_df)} brands from Sephora")
    
    return brand_df

def analyze_tiktok_mentions(tiktok_df, sentence_col):
    """Analyze brand mentions in TikTok data"""
    print("Analyzing TikTok mentions...")
    
    # Initialize sentiment analyzer
    try:
        sia = SentimentIntensityAnalyzer()
    except:
        print("Downloading NLTK resources...")
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
    
    # Extract and analyze brand mentions
    brand_mentions = {}
    
    for _, row in tiktok_df.iterrows():
        if not isinstance(row['brands_parsed'], list) or not row['brands_parsed']:
            continue
        
        sentence = str(row[sentence_col]) if not pd.isna(row[sentence_col]) else ""
        if not sentence:
            continue
        
        # Calculate sentiment for this sentence
        sentiment = sia.polarity_scores(sentence)
        
        # Process each brand mentioned
        for brand in row['brands_parsed']:
            if not brand:
                continue
                
            # Initialize brand data if needed
            if brand not in brand_mentions:
                brand_mentions[brand] = {
                    'mention_count': 0,
                    'compound_scores': [],
                    'positive_scores': [],
                    'neutral_scores': [],
                    'negative_scores': [],
                    'sentences': [],
                    'terms': Counter(),
                    'original_name': brand
                }
            
            # Update brand data
            brand_mentions[brand]['mention_count'] += 1
            brand_mentions[brand]['compound_scores'].append(sentiment['compound'])
            brand_mentions[brand]['positive_scores'].append(sentiment['pos'])
            brand_mentions[brand]['neutral_scores'].append(sentiment['neu'])
            brand_mentions[brand]['negative_scores'].append(sentiment['neg'])
            brand_mentions[brand]['sentences'].append(sentence)
            
            # Extract terms for this sentence, removing brand name
            brand_parts = re.findall(r'\b[a-zA-Z]{2,}\b', brand.lower())
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
            stop_words = {'what', 'with', 'this', 'that', 'from', 'have', 'they', 'will', 
                         'your', 'skin', 'using', 'used', 'like', 'make', 'more', 'than', 
                         'help', 'also', 'into', 'when', 'just', 'really', 'very', 'super',
                         'going', 'because', 'about', 'would', 'could', 'should', 'their',
                         'there', 'where', 'these', 'those', 'them', 'then', 'which', 'here',
                         'know', 'think', 'thought', 'want', 'wanted', 'actually', 'pretty'}
            
            # Add brand name parts to stop words
            stop_words.update(brand_parts)
            
            filtered_words = [w for w in words if w not in stop_words]
            brand_mentions[brand]['terms'].update(filtered_words)
    
    # Convert to DataFrame format
    tiktok_data = []
    
    for brand, data in brand_mentions.items():
        # Calculate average sentiment
        avg_compound = np.mean(data['compound_scores']) if data['compound_scores'] else 0
        avg_positive = np.mean(data['positive_scores']) if data['positive_scores'] else 0
        avg_neutral = np.mean(data['neutral_scores']) if data['neutral_scores'] else 0
        avg_negative = np.mean(data['negative_scores']) if data['negative_scores'] else 0
        
        # Determine overall sentiment
        if avg_compound >= 0.05:
            sentiment_label = "Positive"
        elif avg_compound <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Get top terms
        top_terms = [term for term, count in data['terms'].most_common(10)]
        
        tiktok_data.append({
            'brand': brand,
            'original_name': data['original_name'],
            'mention_count': data['mention_count'],
            'avg_compound': avg_compound,
            'avg_positive': avg_positive,
            'avg_neutral': avg_neutral,
            'avg_negative': avg_negative,
            'sentiment_label': sentiment_label,
            'tiktok_terms': top_terms,
            'term_counts': {term: count for term, count in data['terms'].most_common(50)}
        })
    
    tiktok_df = pd.DataFrame(tiktok_data)
    print(f"Analyzed TikTok mentions for {len(tiktok_df)} brands")
    
    return tiktok_df

def merge_brand_data(sephora_brands, tiktok_brands):
    """Merge Sephora and TikTok data by brand"""
    print("Merging brand data...")
    
    # First check for each TikTok brand if there's a match in Sephora
    merged_data = []
    
    # Create sets of brands for faster lookups
    sephora_brand_set = set(sephora_brands['brand'])
    
    # First, add TikTok data with matches to Sephora data
    for _, tiktok_row in tiktok_brands.iterrows():
        tiktok_brand = tiktok_row['brand']
        
        # Look for exact match
        matched = False
        match_data = {}
        
        if tiktok_brand in sephora_brand_set:
            # Exact match found
            sephora_row = sephora_brands[sephora_brands['brand'] == tiktok_brand].iloc[0]
            match_data = sephora_row.to_dict()
            matched = True
        else:
            # Try partial match
            for sephora_brand in sephora_brand_set:
                if tiktok_brand in sephora_brand or sephora_brand in tiktok_brand:
                    sephora_row = sephora_brands[sephora_brands['brand'] == sephora_brand].iloc[0]
                    match_data = sephora_row.to_dict()
                    matched = True
                    print(f"Partial match: TikTok '{tiktok_brand}' with Sephora '{sephora_brand}'")
                    break
            
        # Combine data
        if matched:
            # Combine Sephora and TikTok data
            match_data.update(tiktok_row.to_dict())
            merged_data.append(match_data)
        else:
            # Only TikTok data, no Sephora match
            print(f"No Sephora match for TikTok brand: {tiktok_brand}")
            tiktok_data = tiktok_row.to_dict()
            # Add placeholder Sephora data
            tiktok_data.update({
                'avg_price': None,
                'key_characteristics': [],
                'total_loves': None,
                'avg_rating': None,
                'avg_loves': None,
                'product_count': 0
            })
            merged_data.append(tiktok_data)
    
    # Add Sephora brands that have no TikTok mentions
    tiktok_brand_set = set(tiktok_brands['brand'])
    for _, sephora_row in sephora_brands.iterrows():
        sephora_brand = sephora_row['brand']
        
        # Check if this brand was already added (exact or partial match)
        matched = False
        for data in merged_data:
            if data.get('brand') == sephora_brand or sephora_brand in str(data.get('brand', '')) or str(data.get('brand', '')) in sephora_brand:
                matched = True
                break
                
        if not matched:
            # Add Sephora data with no TikTok data
            sephora_data = sephora_row.to_dict()
            # Add placeholder TikTok data
            sephora_data.update({
                'mention_count': 0,
                'sentiment_label': "No mentions",
                'tiktok_terms': [],
                'avg_compound': None,
                'avg_positive': None,
                'avg_neutral': None,
                'avg_negative': None
            })
            merged_data.append(sephora_data)
    
    # Convert to DataFrame
    merged_df = pd.DataFrame(merged_data)
    
    # Fill NAs for mention count
    merged_df['mention_count'] = merged_df['mention_count'].fillna(0)
    
    # Use the original name with proper capitalization
    merged_df['display_brand'] = merged_df.apply(
        lambda row: row.get('original_name_x', '') if pd.notnull(row.get('original_name_x', None)) 
        else row.get('original_name_y', row.get('brand', 'Unknown')),
        axis=1
    )
    
    # Sort by total_loves
    if 'total_loves' in merged_df.columns:
        # Convert to numeric and handle NAs
        merged_df['total_loves'] = pd.to_numeric(merged_df['total_loves'], errors='coerce')
        merged_df = merged_df.sort_values('total_loves', ascending=False, na_position='last')
    
    print(f"Final dataset has {len(merged_df)} brands")
    
    return merged_df

def generate_visualizations(merged_df):
    """Generate visualizations based on the merged data"""
    print("Generating visualizations...")
    
    # Set style for visualizations
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Get top 20 brands by TikTok mentions
    top20_brands = merged_df[merged_df['mention_count'] > 0].nlargest(20, 'mention_count')
    top20_brand_names = set(top20_brands['brand'])
    
    # 1. Correlation between TikTok mentions and Loves Count (only top 20 mentioned brands)
    if 'mention_count' in top20_brands.columns and 'total_loves' in top20_brands.columns:
        plt.figure(figsize=(12, 10))
        # Filter to brands with some loves
        plot_df = top20_brands[pd.notnull(top20_brands['total_loves']) & (top20_brands['total_loves'] > 0)]
        
        if len(plot_df) > 3:  # Only plot if we have enough data
            # Create scatter plot
            plt.figure(figsize=(12, 10))
            
            # Use avg_price for point size if available
            if 'avg_price' in plot_df.columns:
                sizes = plot_df['avg_price'].fillna(10) * 3
                sizes = sizes.apply(lambda x: min(max(30, x), 300))  # Limit size range
            else:
                sizes = 100
            
            # Plot with logarithmic scale for better visibility
            plt.scatter(plot_df['total_loves'], plot_df['mention_count'], 
                      alpha=0.7, s=sizes, c=plot_df['avg_rating'], cmap='viridis')
            
            # Add colorbar for rating
            if 'avg_rating' in plot_df.columns:
                cbar = plt.colorbar()
                cbar.set_label('Average Rating')
            
            # Add brand labels
            for _, row in plot_df.iterrows():
                plt.text(row['total_loves'] * 1.05, row['mention_count'] * 1.05, 
                        row['display_brand'], fontsize=9)
            
            # Set axis to logarithmic scale
            plt.xscale('log')
            
            # Add title and labels
            plt.title('Relationship Between Loves Count and TikTok Mentions (Top 20 Brands)', fontsize=16)
            plt.xlabel('Loves Count (log scale)', fontsize=14)
            plt.ylabel('TikTok Mentions', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'loves_mentions_correlation.png'), dpi=300)
            plt.close()
    
    # 2. Sentiment distribution pie chart
    if 'sentiment_label' in top20_brands.columns:
        plt.figure(figsize=(10, 8))
        
        # Count sentiment distribution
        sentiment_counts = top20_brands['sentiment_label'].value_counts()
        
        # Filter out "No mentions" for pie chart
        if "No mentions" in sentiment_counts:
            sentiment_counts = sentiment_counts[sentiment_counts.index != "No mentions"]
        
        if not sentiment_counts.empty:
            # Create pie chart
            plt.figure(figsize=(10, 8))
            colors = ['#66c2a5', '#fc8d62', '#8da0cb']
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors, wedgeprops=dict(width=0.5, edgecolor='w'))
            
            plt.title('Sentiment Distribution for Top 20 Mentioned Brands', fontsize=16)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'sentiment_distribution.png'), dpi=300)
            plt.close()
    
    # 3. Top brands by mentions (horizontal bar chart)
    if 'mention_count' in top20_brands.columns and 'display_brand' in top20_brands.columns:
        plt.figure(figsize=(12, 14))
        
        if not top20_brands.empty:
            # Create horizontal bar chart
            plt.figure(figsize=(12, 14))
            
            # Use sentiment for color coding
            colors = []
            if 'sentiment_label' in top20_brands.columns:
                color_map = {'Positive': '#66c2a5', 'Neutral': '#8da0cb', 'Negative': '#fc8d62', 'No mentions': '#cccccc'}
                colors = [color_map.get(sentiment, '#cccccc') for sentiment in top20_brands['sentiment_label']]
            
            # Plot horizontal bars
            sns.barplot(x='mention_count', y='display_brand', data=top20_brands, palette=colors)
            
            # Add count labels
            for i, v in enumerate(top20_brands['mention_count']):
                plt.text(v + 0.1, i, str(int(v)), va='center')
            
            plt.title('Top 20 Brands by TikTok Mentions', fontsize=16)
            plt.xlabel('Number of Mentions', fontsize=14)
            plt.ylabel('Brand', fontsize=14)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'top_brands_by_mentions.png'), dpi=300)
            plt.close()
    
    # 4. Price vs. Rating bubble chart (with limited price range 0-100)
    if all(col in merged_df.columns for col in ['avg_price', 'avg_rating', 'total_loves']):
        plt.figure(figsize=(12, 10))
        
        # Filter to only top 20 mentioned brands with valid data points
        plot_df = top20_brands[(pd.notnull(top20_brands['avg_price'])) & 
                              (pd.notnull(top20_brands['avg_rating'])) & 
                              (pd.notnull(top20_brands['total_loves']))]
        
        if len(plot_df) > 3:  # Only plot if we have enough data
            # Create bubble chart
            plt.figure(figsize=(12, 10))
            
            # Normalize total_loves for bubble size
            sizes = np.sqrt(plot_df['total_loves']) / 30
            sizes = sizes.apply(lambda x: min(max(30, x), 500))  # Limit size range
            
            # Use mention_count for color intensity
            if 'mention_count' in plot_df.columns:
                colors = plot_df['mention_count']
                plt.scatter(plot_df['avg_price'], plot_df['avg_rating'], s=sizes, c=colors, 
                           alpha=0.7, cmap='YlOrRd', edgecolors='grey', linewidths=1)
                plt.colorbar(label='TikTok Mentions')
            else:
                plt.scatter(plot_df['avg_price'], plot_df['avg_rating'], s=sizes, 
                           alpha=0.7, edgecolors='grey', linewidths=1)
            
            # Add brand labels
            for _, row in plot_df.iterrows():
                plt.text(row['avg_price'] * 1.05, row['avg_rating'] * 1.001, 
                        row['display_brand'], fontsize=9)
            
            # Set x-axis limit to 0-100
            plt.xlim(0, 100)
            
            # Set y-axis limits with some padding
            y_min = max(0, plot_df['avg_rating'].min() - 0.2)
            y_max = min(5, plot_df['avg_rating'].max() + 0.2)
            plt.ylim(y_min, y_max)
            
            plt.title('Price vs. Rating for Top 20 Mentioned Brands', fontsize=16)
            plt.xlabel('Average Price ($)', fontsize=14)
            plt.ylabel('Average Rating', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'price_rating_bubble.png'), dpi=300)
            plt.close()
    
    # 5. Word cloud for most frequent terms in TikTok mentions
    if 'term_counts' in top20_brands.columns:
        # Combine all term counts
        all_terms = Counter()
        for term_dict in top20_brands['term_counts'].dropna():
            if isinstance(term_dict, dict):
                all_terms.update(term_dict)
        
        if all_terms:
            # Extended stopwords for beauty products
            makeup_stopwords = {
                'makeup', 'beauty', 'product', 'products', 'cosmetics', 'skincare', 
                'skin', 'face', 'look', 'looks', 'looking', 'apply', 'application',
                'makes', 'made', 'make', 'making', 'cosmetic', 'beauty', 'beautiful',
                'pretty', 'nice', 'good', 'great', 'amazing', 'awesome', 'perfect',
                'absolutely', 'totally', 'really', 'very', 'super', 'literally',
                'actually', 'basically', 'definitely', 'finally', 'probably',
                'using', 'uses', 'used', 'love', 'loved', 'loves', 'loving',
                'best', 'better', 'worst', 'favorite', 'colour', 'color', 'colors',
                'buying', 'bought', 'purchase', 'purchased', 'want', 'wanted',
                'needs', 'needed', 'need', 'sell', 'selling', 'sells', 'sold'
            }
            
            # Filter out stopwords
            filtered_terms = {k: v for k, v in all_terms.items() if k.lower() not in makeup_stopwords}
            
            # Create word cloud
            plt.figure(figsize=(12, 8))
            wordcloud = WordCloud(
                width=800, 
                height=400,
                max_words=100,
                background_color='white',
                colormap='viridis',
                contour_width=1,
                contour_color='steelblue'
            ).generate_from_frequencies(filtered_terms)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Terms in TikTok Mentions (Top 20 Brands)', fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'tiktok_terms_wordcloud.png'), dpi=300)
            plt.close()
    
    # 6. Word cloud of product characteristics from Sephora
    if 'characteristic_counts' in top20_brands.columns:
        # Combine all characteristic counts
        all_chars = Counter()
        for char_dict in top20_brands['characteristic_counts'].dropna():
            if isinstance(char_dict, dict):
                all_chars.update(char_dict)
        
        if all_chars:
            # Extended stopwords for beauty product characteristics
            char_stopwords = {
                'makeup', 'beauty', 'product', 'products', 'cosmetics', 'skincare', 
                'skin', 'face', 'look', 'looks', 'looking', 'apply', 'application',
                'makes', 'made', 'make', 'making', 'cosmetic', 'beauty', 'beautiful',
                'pretty', 'nice', 'good', 'great', 'amazing', 'awesome', 'perfect',
                'absolutely', 'totally', 'really', 'very', 'super', 'literally',
                'foundation', 'concealer', 'powder', 'blush', 'bronzer', 'highlighter',
                'eyeshadow', 'mascara', 'eyeliner', 'lipstick', 'lipgloss', 'liner',
                'primer', 'setting', 'spray', 'cream', 'lotion', 'serum', 'moisturizer',
                'cleanser', 'toner', 'essence', 'treatment', 'mask', 'exfoliator',
                'scrub', 'spf', 'sunscreen', 'balm', 'oil', 'formula', 'formulation',
                'application', 'applicator', 'bottle', 'tube', 'container', 'package',
                'packaging', 'compact', 'palette', 'shade', 'shades', 'color', 'colors',
                'undertone', 'finish', 'coverage', 'texture', 'consistency', 'formula'
            }
            
            # Filter out stopwords
            filtered_chars = {k: v for k, v in all_chars.items() if k.lower() not in char_stopwords}
            
            # Create word cloud
            plt.figure(figsize=(12, 8))
            wordcloud = WordCloud(
                width=800, 
                height=400,
                max_words=100,
                background_color='white',
                colormap='plasma',
                contour_width=1,
                contour_color='steelblue'
            ).generate_from_frequencies(filtered_chars)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Key Product Characteristics (Top 20 Brands)', fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'product_characteristics_wordcloud.png'), dpi=300)
            plt.close()
    
    # 7. Stacked bar chart for sentiment by top brands
    if all(col in top20_brands.columns for col in ['display_brand', 'avg_positive', 'avg_neutral', 'avg_negative']):
        if not top20_brands.empty:
            # Create stacked bar chart
            plt.figure(figsize=(12, 10))
            
            # Extract sentiment components
            brands = top20_brands['display_brand']
            positive = top20_brands['avg_positive']
            neutral = top20_brands['avg_neutral']
            negative = top20_brands['avg_negative']
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame({
                'Brand': brands,
                'Positive': positive,
                'Neutral': neutral,
                'Negative': negative
            })
            
            # Create stacked bar chart
            plot_data.set_index('Brand').plot(kind='barh', stacked=True, 
                                            color=['#66c2a5', '#8da0cb', '#fc8d62'],
                                            figsize=(12, 10))
            
            plt.title('Sentiment Composition for Top 20 Brands', fontsize=16)
            plt.xlabel('Sentiment Score', fontsize=14)
            plt.ylabel('Brand', fontsize=14)
            plt.legend(title='Sentiment Type')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'sentiment_by_brand.png'), dpi=300)
            plt.close()
    
    # 8. Mentions by Price Range (grouped bar chart instead of box plot)
    if 'mention_count' in top20_brands.columns and 'avg_price' in top20_brands.columns:
        plt.figure(figsize=(12, 8))
    
    # Filter to brands with valid price
    plot_df = top20_brands[pd.notnull(top20_brands['avg_price'])]
    
    if not plot_df.empty:
        # Create price ranges
        plot_df['price_range'] = pd.cut(
            plot_df['avg_price'],
            bins=[0, 20, 40, 60, 100, float('inf')],
            labels=['$0-$20', '$20-$40', '$40-$60', '$60-$100', '$100+']
        )
        
        # Group by price range and calculate total mentions and count
        price_range_stats = plot_df.groupby('price_range').agg({
            'mention_count': ['sum', 'count']
        }).reset_index()
        
        price_range_stats.columns = ['price_range', 'total_mentions', 'brand_count']
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        # Plot total mentions
        bars = plt.bar(price_range_stats['price_range'], price_range_stats['total_mentions'], 
                      color='#5975a4', alpha=0.8)
        
        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Add brand count as text above bars
        for i, row in price_range_stats.iterrows():
            plt.text(i, row['total_mentions'] + max(price_range_stats['total_mentions']) * 0.05, 
                    f'({int(row["brand_count"])} brands)', 
                    ha='center', va='bottom', color='#555555')
        
        # Set labels and title
        plt.xlabel('Price Range', fontsize=14)
        plt.ylabel('Total Mentions', fontsize=14)
        plt.title('TikTok Mentions by Price Range (Top 20 Brands)', fontsize=16)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'mentions_by_price_range.png'), dpi=300)
        plt.close()
    
    # Add paths to generated visualizations to be included in Excel
    return [
        os.path.join(VISUALIZATIONS_DIR, 'loves_mentions_correlation.png'),
        os.path.join(VISUALIZATIONS_DIR, 'sentiment_distribution.png'),
        os.path.join(VISUALIZATIONS_DIR, 'top_brands_by_mentions.png'),
        os.path.join(VISUALIZATIONS_DIR, 'price_rating_bubble.png'),
        os.path.join(VISUALIZATIONS_DIR, 'tiktok_terms_wordcloud.png'),
        os.path.join(VISUALIZATIONS_DIR, 'product_characteristics_wordcloud.png'),
        os.path.join(VISUALIZATIONS_DIR, 'sentiment_by_brand.png'),
        os.path.join(VISUALIZATIONS_DIR, 'mentions_by_price_range.png')
    ]

def save_brand_summary(merged_df, output_file, visualization_paths):
    """Save brand summary to Excel"""
    print(f"Saving brand summary to {output_file}...")
    
    try:
        # Create a copy with clean data for Excel
        excel_df = merged_df.copy()
        
        # Use display_brand as the primary brand column
        if 'display_brand' in excel_df.columns:
            excel_df['brand'] = excel_df['display_brand']
        
        # Convert list columns to strings
        for col in ['key_characteristics', 'tiktok_terms']:
            if col in excel_df.columns:
                excel_df[col] = excel_df[col].apply(
                    lambda x: ', '.join(map(str, x)) if isinstance(x, list) and x else ''
                )
        
        # Organize columns in the desired order
        final_cols = [
            'brand',
            'avg_price',
            'key_characteristics',
            'total_loves',
            'avg_rating',
            'tiktok_terms',
            'sentiment_label',
            'mention_count'
        ]
        
        # Keep only columns that exist
        output_cols = [col for col in final_cols if col in excel_df.columns]
        
        # Rename columns to match the requested format
        column_names = {
            'brand': 'Brand',
            'avg_price': 'Average Price',
            'key_characteristics': 'Characteristics',
            'total_loves': 'Loves Count',
            'avg_rating': 'Rating',
            'tiktok_terms': 'Mentioned Key TikTok Terms',
            'sentiment_label': 'Sentiment',
            'mention_count': 'TikTok Mentions'
        }
        
        final_df = excel_df[output_cols].rename(columns=column_names)
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Write main sheet
            final_df.to_excel(writer, sheet_name='Brand Summary', index=False)
            
            # Format worksheet
            workbook = writer.book
            worksheet = writer.sheets['Brand Summary']
            
            # Header format
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Apply formatting
            for col_num, value in enumerate(final_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
                # Set column widths
                if value in ['Characteristics', 'Mentioned Key TikTok Terms']:
                    worksheet.set_column(col_num, col_num, 40)  # Wider for text fields
                elif value == 'Brand':
                    worksheet.set_column(col_num, col_num, 20)  # Medium for brand names
                else:
                    worksheet.set_column(col_num, col_num, 15)  # Default for other columns
            
            # Add visualizations to a dedicated sheet
            if visualization_paths:
                viz_sheet = workbook.add_worksheet('Visualizations')
                
                # Add title
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 16,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                viz_sheet.merge_range('A1:J1', 'Brand Analysis Visualizations', title_format)
                
                # Add images
                row = 3
                for img_path in visualization_paths:
                    if os.path.exists(img_path):
                        try:
                            # Get image filename without extension
                            img_title = os.path.basename(img_path).replace('_', ' ').replace('.png', '')
                            
                            # Add title for this visualization
                            viz_sheet.write(f'B{row-1}', img_title.title(), workbook.add_format({'bold': True}))
                            
                            # Insert image
                            viz_sheet.insert_image(f'B{row}', img_path, {'x_scale': 0.7, 'y_scale': 0.7})
                            
                            # Move to next position
                            row += 25
                        except Exception as e:
                            print(f"Warning: Could not insert image {img_path}: {e}")
        
        print(f"Brand summary saved successfully to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        
        # Try to save as CSV as fallback
        try:
            csv_output = output_file.replace('.xlsx', '.csv')
            merged_df.to_csv(csv_output, index=False)
            print(f"Brand summary saved as CSV to {csv_output}")
            return True
        except Exception as csv_error:
            print(f"Error saving to CSV: {csv_error}")
            return False

def main():
    """Main function to run the analysis"""
    print("\n==== BRAND SUMMARY ANALYSIS ====\n")
    
    # Load data
    sephora_df, tiktok_df = load_data()
    
    # Preprocess data
    sephora_df, tiktok_df, price_col, sentence_col = preprocess_data(sephora_df, tiktok_df)
    
    # Calculate brand metrics from Sephora
    sephora_brands = calculate_brand_metrics(sephora_df, price_col)
    
    # Analyze TikTok mentions
    tiktok_brands = analyze_tiktok_mentions(tiktok_df, sentence_col)
    
    # Merge data
    merged_df = merge_brand_data(sephora_brands, tiktok_brands)
    
    # Generate visualizations
    visualization_paths = generate_visualizations(merged_df)
    
    # Save summary
    if not merged_df.empty:
        save_brand_summary(merged_df, OUTPUT_FILE, visualization_paths)
    else:
        print("Error: No brand data to save")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()