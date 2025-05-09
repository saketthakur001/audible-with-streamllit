import pandas as pd
import numpy as np
import ast
import urllib.parse
from datetime import datetime
import math

from flask import Flask, render_template_string, request, url_for
from markupsafe import Markup


# --- Configuration ---
DATA_PATH = 'books .csv'  # Ensure this file is in the same directory as the script
DEFAULT_DISPLAY_MODE = 'grid'
BOOKS_PER_PAGE = 24 # Books per page for pagination
DEFAULT_SORT_ORDER = 'popularity_desc' # New default sort

# --- Constants for Sliders ---
MAX_RATINGS_COUNT_FOR_SLIDER = 50000 # Max for the 'min_votes' slider
RATINGS_COUNT_SLIDER_STEP = 500
MAX_PAGES_FOR_SLIDER = 3000 # Max for the 'max_pages' slider
PAGES_SLIDER_STEP = 50


# --- Column Name Mapping (from CSV to internal names) ---
COLUMN_NAME_MAP = {
    'bookId': 'book_id_str', 'author': 'authors', 'rating': 'average_rating',
    'numRatings': 'ratings_count', 'pages': 'num_pages', 'language': 'language_code',
    'publishDate': 'publication_date_edition', 'firstPublishDate': 'first_publication_date',
}
REQUIRED_ORIGINAL_COLS = [
    'title', 'author', 'rating', 'numRatings', 'pages', 'language',
    'publisher', 'genres', 'coverImg'
]

# --- Data Loading and Cleaning Function ---
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

        for col in REQUIRED_ORIGINAL_COLS:
            if col not in df.columns:
                print(f"Error: Essential source column '{col}' not found.")
                return pd.DataFrame()

        df.rename(columns=COLUMN_NAME_MAP, inplace=True)

        df['ratings_count'] = pd.to_numeric(df.get('ratings_count'), errors='coerce').fillna(0).astype(int)
        df['num_pages'] = pd.to_numeric(df.get('num_pages'), errors='coerce').fillna(0).astype(int)
        df['average_rating'] = pd.to_numeric(df.get('average_rating'), errors='coerce')
        df['likedPercent'] = pd.to_numeric(df.get('likedPercent'), errors='coerce').fillna(0)
        df['price'] = pd.to_numeric(df.get('price'), errors='coerce').fillna(0)

        for col in ['title', 'authors', 'publisher', 'series', 'bookFormat', 'language_code', 'book_id_str']:
            df[col] = df.get(col, pd.Series(index=df.index, dtype='str')).fillna('Unknown')
        df['coverImg'] = df.get('coverImg', pd.Series(index=df.index, dtype='str')).fillna('')

        # Handle various date formats more robustly if possible, or ensure consistency in CSV
        # The original format='%m/%d/%y' is specific. If other formats exist, parsing will fail more often.
        df['first_publication_date_dt'] = pd.to_datetime(df.get('first_publication_date'), errors='coerce') #, format='%m/%d/%y', infer_datetime_format=False)
        df['publication_date_edition_dt'] = pd.to_datetime(df.get('publication_date_edition'), errors='coerce') #, format='%m/%d/%y', infer_datetime_format=False)
        df['publication_year'] = df['first_publication_date_dt'].dt.year.fillna(df['publication_date_edition_dt'].dt.year).fillna(0).astype(int)
        df['display_publication_date'] = df['first_publication_date_dt'].dt.strftime('%Y-%m-%d').fillna(df['publication_date_edition_dt'].dt.strftime('%Y-%m-%d')).fillna('Unknown')

        def parse_genres(genre_str):
            if isinstance(genre_str, str) and genre_str.startswith('[') and genre_str.endswith(']'):
                try: return ast.literal_eval(genre_str)
                except: return []
            return [] if not isinstance(genre_str, list) else genre_str
        df['genres_list'] = df['genres'].apply(parse_genres)
        df['genres_display_full'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else 'N/A')
        df['genres_display_short'] = df['genres_list'].apply(lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '') if x else 'N/A')

        def create_audible_link_url(title_str):
            if pd.isna(title_str) or title_str == '' or title_str == 'Unknown': return None
            base = "https://www.audible.in/search?"
            params = {'keywords': str(title_str), 'k': str(title_str)}
            return f"{base}{urllib.parse.urlencode(params)}"
        df['audible_link'] = df['title'].apply(create_audible_link_url)

        # Calculate Bayesian Rating for "Popularity" sort
        if not df.empty and 'average_rating' in df.columns and 'ratings_count' in df.columns:
            # Use a global mean for books with 0 ratings or if their own rating is NaN.
            # Filter out NaNs before calculating mean to avoid NaN propagation.
            valid_ratings = df['average_rating'].dropna()
            m_global_mean_rating = valid_ratings.mean() if not valid_ratings.empty else 3.0 # Fallback global mean

            # C is the "confidence" parameter - a prior number of ratings.
            # Higher C means a new book needs more ratings to shift from the global average.
            # Let's use a fixed value like 200, or a quantile of existing ratings counts.
            # C_prior_ratings_count = df['ratings_count'].quantile(0.50) # Median number of ratings
            C_prior_ratings_count = 200 # A fixed "typical" number of ratings for confidence

            def calculate_bayesian_rating(row, mean_rating, c_val):
                avg_r = row['average_rating']
                num_r = row['ratings_count']
                if pd.isna(avg_r) or pd.isna(num_r) or num_r == 0:
                    # If no rating or no votes, use its own average_rating if available (e.g. editorial rating),
                    # otherwise fallback to global mean. This helps books with an initial rating but 0 votes.
                    return avg_r if pd.notna(avg_r) else mean_rating
                return ((c_val * mean_rating) + (avg_r * num_r)) / (c_val + num_r)

            df['bayesian_rating'] = df.apply(lambda row: calculate_bayesian_rating(row, m_global_mean_rating, C_prior_ratings_count), axis=1)
        else:
            # Ensure column exists even if data is empty or required cols are missing
            df['bayesian_rating'] = pd.Series(index=df.index, dtype='float').fillna(3.0)


        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An critical error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Load data once when the app starts
BOOKS_DF = load_and_clean_data(DATA_PATH)
if not BOOKS_DF.empty:
    ALL_GENRES = sorted(list(set(g for sublist in BOOKS_DF['genres_list'] for g in sublist if g)))
    ALL_LANGUAGES = sorted([lang for lang in BOOKS_DF['language_code'].dropna().unique().tolist() if lang != 'Unknown'])
    ALL_FORMATS = sorted([fmt for fmt in BOOKS_DF['bookFormat'].dropna().unique().tolist() if fmt != 'Unknown'])
    # Ensure publication_year > 0 for min/max calculation
    valid_pub_years = BOOKS_DF['publication_year'][BOOKS_DF['publication_year'] > 1000] # Basic sanity check for year
    MIN_PUB_YEAR = int(valid_pub_years.min()) if not valid_pub_years.empty else 1800
    MAX_PUB_YEAR = int(BOOKS_DF['publication_year'].max()) if not BOOKS_DF['publication_year'].empty else datetime.now().year
else: # Fallbacks if data loading fails
    ALL_GENRES, ALL_LANGUAGES, ALL_FORMATS = [], [], []
    MIN_PUB_YEAR, MAX_PUB_YEAR = 1800, datetime.now().year


# --- Flask App Initialization ---
app = Flask(__name__)

# --- Utility Functions ---
def get_star_rating_html(rating_val, ratings_count=None, small=False):
    if pd.isna(rating_val) or rating_val == 0: return "<span class='stars-na'>N/A</span>"
    try:
        rating_val = float(rating_val)
        full_stars = int(rating_val)
        half_star_val = rating_val - full_stars
        half_star_char = ""
        if half_star_val >= 0.75:
            full_stars +=1 # Round up for .75 or more
        elif half_star_val >= 0.25:
            half_star_char = "½"

        empty_stars = 5 - full_stars - (1 if half_star_char else 0)
        if small:
            stars_html = f"<span class='stars'>{'★' * full_stars}{half_star_char}{'☆' * empty_stars}</span> <span class='rating-value-small'>({rating_val:.1f})</span>"
        else:
            stars_html = f"<span class='stars'>{'★' * full_stars}{half_star_char}{'☆' * empty_stars}</span> <span class='rating-value'>({rating_val:.2f})</span>"
        if ratings_count is not None and not small:
            stars_html += f" <span class='ratings-count'>({ratings_count:,} ratings)</span>"
        return Markup(stars_html)
    except ValueError: return "<span class='stars-na'>Error</span>"


# --- HTML Template (Embedded) ---
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Book Explorer Pro</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #121212;
            --bg-secondary: #1e1e1e;
            --bg-tertiary: #2a2a2a;
            --bg-interactive: #333333;
            --bg-interactive-hover: #404040;
            --text-primary: #e0e0e0;
            --text-secondary: #b3b3b3;
            --text-placeholder: #757575;
            --accent-primary: #00aeff; /* Brighter, more modern blue */
            --accent-primary-hover: #0095dd; /* Darker shade for hover */
            --border-color: #383838;
            --border-color-light: #4f4f4f;
            --star-color: #ffc107;
            --shadow-color: rgba(0, 0, 0, 0.3);
            --shadow-light-color: rgba(0,0,0,0.15);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            display: flex;
            min-height: 100vh;
            font-size: 15px;
            line-height: 1.6;
        }
        .sidebar {
            width: 300px;
            background-color: var(--bg-secondary);
            padding: 25px;
            border-right: 1px solid var(--border-color);
            overflow-y: auto;
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            box-shadow: 2px 0 10px var(--shadow-color);
            scrollbar-width: thin;
            scrollbar-color: var(--bg-interactive) var(--bg-secondary);
        }
        .main-content {
            margin-left: 320px; /* Sidebar width + some padding */
            padding: 25px 30px;
            width: calc(100% - 320px);
            overflow-y: auto;
        }
        h1, h2, h3 {
            color: var(--accent-primary);
            font-weight: 600;
        }
        h1 { text-align: center; margin-bottom: 25px; font-size: 2.2em; letter-spacing: -0.5px;}
        .sidebar h2 { margin-top:0; margin-bottom: 20px; font-size: 1.6em; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; }

        .filter-group {
            margin-bottom: 25px;
            padding: 15px;
            background-color: var(--bg-tertiary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        .filter-group h3 { margin-top: 0; font-size: 1.15em; color: var(--text-primary); border-bottom: 1px solid var(--border-color-light); padding-bottom: 8px; margin-bottom:12px; font-weight:500;}
        label { display: block; margin-bottom: 6px; font-size: 0.9em; font-weight: 500; color: var(--text-secondary); }

        input[type="text"], input[type="number"], select {
            width: calc(100% - 20px);
            padding: 10px;
            margin-bottom: 12px;
            border-radius: 5px;
            border: 1px solid var(--border-color-light);
            background-color: var(--bg-interactive);
            color: var(--text-primary);
            box-sizing: border-box;
            font-size: 0.95em;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px var(--accent-primary-hover);
        }
        select[multiple] { height: 120px; }

        input[type="range"] {
            width: 100%;
            margin-bottom: 0;
            -webkit-appearance: none;
            appearance: none;
            background: transparent;
            cursor: pointer;
        }
        input[type="range"]::-webkit-slider-runnable-track {
            background: var(--bg-interactive);
            height: 6px;
            border-radius: 3px;
        }
        input[type="range"]::-moz-range-track {
            background: var(--bg-interactive);
            height: 6px;
            border-radius: 3px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            margin-top: -5px; /* (track height - thumb height) / 2 assuming thumb height 16px */
            background-color: var(--accent-primary);
            height: 16px;
            width: 16px;
            border-radius: 50%;
            border: 2px solid var(--bg-secondary);
        }
        input[type="range"]::-moz-range-thumb {
            background-color: var(--accent-primary);
            height: 16px;
            width: 16px;
            border-radius: 50%;
            border: 2px solid var(--bg-secondary);
        }
        .slider-labels { display: flex; justify-content: space-between; font-size: 0.8em; margin-top: 2px; margin-bottom:10px; color: var(--text-secondary); }
        .range-value-display { font-weight: 500; color: var(--text-primary); }

        button, input[type="submit"] {
            background-color: var(--accent-primary);
            color: var(--bg-primary);
            border: none;
            padding: 12px 18px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1em;
            transition: background-color 0.2s ease, transform 0.1s ease;
            width: 100%;
        }
        button:hover, input[type="submit"]:hover { background-color: var(--accent-primary-hover); transform: translateY(-1px); }

        .view-switcher { text-align: center; margin-bottom: 25px; }
        .view-switcher a {
            text-decoration: none; color: var(--accent-primary);
            padding: 8px 15px; margin: 0 8px; border-radius: 5px;
            border: 1px solid var(--accent-primary);
            transition: background-color 0.2s, color 0.2s;
            font-weight: 500;
        }
        .view-switcher a.active { background-color: var(--accent-primary); color: var(--bg-primary); }
        .view-switcher a:not(.active):hover { background-color: var(--bg-interactive); }

        /* Grid View */
        .book-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); /* Wider cards */
            gap: 25px;
        }
        .book-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px; /* Reduced padding for more content space */
            display: flex;
            flex-direction: column;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .book-card:hover { transform: translateY(-5px); box-shadow: 0 8px 20px var(--shadow-color); border-color: var(--border-color-light); }
        .book-card img {
            width: 100%;
            height: 250px; /* Adjusted height */
            object-fit: contain;
            border-radius: 6px;
            margin-bottom: 12px;
            background-color: var(--bg-tertiary);
        }
        .book-card .title { font-size: 1.05em; font-weight: 600; color: var(--text-primary); margin-bottom: 5px; line-height:1.3; height: 2.6em; overflow:hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
        .book-card .author { font-size: 0.85em; color: var(--text-secondary); margin-bottom: 8px; height: 1.8em; overflow:hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical;}
        .book-card .rating { font-size: 0.9em; margin-bottom: 10px; }
        .stars { color: var(--star-color); }
        .stars-na { color: var(--text-placeholder); font-style: italic;}
        .rating-value, .ratings-count, .rating-value-small { color: var(--text-secondary); font-size:0.9em; }
        .book-card .details-toggle {
            font-size: 0.85em; color: var(--accent-primary); cursor: pointer; text-decoration: none;
            margin-top: auto; padding-top: 8px; font-weight:500;
        }
        .book-card .details-toggle:hover { text-decoration: underline; }
        .book-card .extra-details {
            display: none;
            font-size: 0.8em; text-align: left; margin-top: 10px;
            border-top: 1px solid var(--border-color-light); padding-top: 10px; color: var(--text-secondary);
        }
        .book-card .extra-details p { margin: 4px 0; }
        .book-card .extra-details strong { color: var(--text-primary); font-weight:500;}
        .book-card .audible-link { display:block; margin-top:10px; font-size:0.9em; color:var(--accent-primary); text-decoration:none; font-weight:500;}
        .book-card .audible-link:hover { text-decoration:underline; }

        /* List View */
        .book-list-item {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 12px;
            display: flex;
            align-items: flex-start; /* Align items to top for better text flow */
            transition: background-color 0.2s;
        }
        .book-list-item:hover { background-color: var(--bg-tertiary); border-color: var(--border-color-light); }
        .book-list-item img {
            width: 70px; height: 105px; object-fit: contain;
            border-radius: 4px; margin-right: 20px; background-color: var(--bg-tertiary);
            flex-shrink: 0;
        }
        .book-list-item .info { flex-grow: 1; }
        .book-list-item .title { font-size: 1.2em; font-weight: 600; color: var(--text-primary); margin-bottom:2px;}
        .book-list-item .author { font-size: 0.95em; color: var(--text-secondary); margin-bottom:6px;}
        .book-list-item .rating { font-size: 0.9em; margin-top: 5px; }
        .book-list-item .meta { font-size: 0.85em; color: var(--text-secondary); margin-top:8px; line-height:1.5; }
        .book-list-item .meta a {color: var(--accent-primary); text-decoration:none;}
        .book-list-item .meta a:hover {text-decoration:underline;}

        .no-results { text-align:center; padding: 40px; font-size: 1.25em; color: var(--text-secondary); background-color: var(--bg-secondary); border-radius:8px;}
        .results-summary { text-align: center; margin-bottom: 20px; font-size: 1em; color: var(--text-secondary);}

        .pagination {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }
        .pagination a, .pagination span {
            margin: 0 4px;
            padding: 8px 12px;
            text-decoration: none;
            color: var(--text-primary);
            background-color: var(--bg-interactive);
            border: 1px solid var(--border-color-light);
            border-radius: 4px;
            transition: background-color 0.2s, color 0.2s;
        }
        .pagination a:hover {
            background-color: var(--accent-primary-hover);
            color: var(--bg-primary);
            border-color: var(--accent-primary-hover);
        }
        .pagination .current-page {
            background-color: var(--accent-primary);
            color: var(--bg-primary);
            border-color: var(--accent-primary);
            font-weight: 600;
        }
        .pagination .disabled {
            color: var(--text-placeholder);
            background-color: var(--bg-tertiary);
            cursor: not-allowed;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>📚 Book Filters</h2>
        <form method="GET" action="/">
            <input type="hidden" name="view" value="{{ current_view }}">
            <input type="hidden" name="page" value="1"> <div class="filter-group">
                <h3>Search & Sort</h3>
                <label for="search_query">Search Term:</label>
                <input type="text" name="search_query" id="search_query" value="{{ filters.search_query or '' }}" placeholder="Title, author, series...">

                <label for="sort_by">Sort by:</label>
                <select name="sort_by" id="sort_by">
                    {% for val, display in sort_options.items() %}
                    <option value="{{ val }}" {% if filters.sort_by == val %}selected{% endif %}>{{ display }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="filter-group">
                <h3>Quality & Engagement</h3>
                <label for="min_rating">Min Avg. Rating: <span class="range-value-display" id="min_rating_val">{{ filters.min_rating or 0.0 }}</span></label>
                <input type="range" name="min_rating" id="min_rating" min="0" max="5" step="0.1" value="{{ filters.min_rating or 0.0 }}" oninput="document.getElementById('min_rating_val').textContent = this.value">
                <div class="slider-labels"><span>0</span><span>5</span></div>

                <label for="min_votes">Min Ratings Count: <span class="range-value-display" id="min_votes_val">{{ filters.min_votes or 0 }}</span></label>
                <input type="range" name="min_votes" id="min_votes" min="0" max="{{max_ratings_slider}}" step="{{ratings_slider_step}}" value="{{ filters.min_votes or 0 }}" oninput="document.getElementById('min_votes_val').textContent = this.value">
                <div class="slider-labels"><span>0</span><span>{{max_ratings_slider}}+</span></div>

                <label for="min_liked">Min Liked Percent: <span class="range-value-display" id="min_liked_val">{{ filters.min_liked or 0 }}</span>%</label>
                <input type="range" name="min_liked" id="min_liked" min="0" max="100" step="1" value="{{ filters.min_liked or 0 }}" oninput="document.getElementById('min_liked_val').textContent = this.value + '%'">
                <div class="slider-labels"><span>0%</span><span>100%</span></div>
            </div>

            <div class="filter-group">
                <h3>Content Attributes</h3>
                <label for="genres">Genres (select multiple):</label>
                <select name="genres" id="genres" multiple>
                    {% for genre in all_genres %}
                    <option value="{{ genre }}" {% if genre in filters.genres %}selected{% endif %}>{{ genre }}</option>
                    {% endfor %}
                </select>

                <label for="language">Language:</label>
                <select name="language" id="language">
                    <option value="">All Languages</option>
                    {% for lang in all_languages %}
                    <option value="{{ lang }}" {% if filters.language == lang %}selected{% endif %}>{{ lang }}</option>
                    {% endfor %}
                </select>

                <label for="book_format">Format:</label>
                <select name="book_format" id="book_format">
                    <option value="">All Formats</option>
                    {% for fmt in all_formats %}
                    <option value="{{ fmt }}" {% if filters.book_format == fmt %}selected{% endif %}>{{ fmt }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="filter-group">
                <h3>Publication & Length</h3>
                <label for="pub_year_min">Min Pub. Year: <span class="range-value-display" id="pub_year_min_val">{{ filters.pub_year_min or min_pub_year }}</span></label>
                <input type="range" name="pub_year_min" id="pub_year_min" min="{{min_pub_year}}" max="{{max_pub_year}}" step="1" value="{{ filters.pub_year_min or min_pub_year }}" oninput="document.getElementById('pub_year_min_val').textContent = this.value">
                <div class="slider-labels"><span>{{min_pub_year}}</span><span>{{max_pub_year}}</span></div>

                <label for="pub_year_max">Max Pub. Year: <span class="range-value-display" id="pub_year_max_val">{{ filters.pub_year_max or max_pub_year }}</span></label>
                <input type="range" name="pub_year_max" id="pub_year_max" min="{{min_pub_year}}" max="{{max_pub_year}}" step="1" value="{{ filters.pub_year_max or max_pub_year }}" oninput="document.getElementById('pub_year_max_val').textContent = this.value">
                <div class="slider-labels"><span>{{min_pub_year}}</span><span>{{max_pub_year}}</span></div>

                <label for="max_pages">Max Pages: <span class="range-value-display" id="max_pages_val">{{ filters.max_pages or max_pages_slider }}</span></label>
                <input type="range" name="max_pages" id="max_pages" min="0" max="{{max_pages_slider}}" step="{{pages_slider_step}}" value="{{ filters.max_pages or max_pages_slider }}" oninput="document.getElementById('max_pages_val').textContent = this.value">
                <div class="slider-labels"><span>0</span><span>{{max_pages_slider}}+</span></div>
            </div>

            <input type="submit" value="Apply Filters">
        </form>
    </div>

    <div class="main-content">
        <h1>Book Explorer Pro</h1>
        <div class="view-switcher">
            <a href="{{ url_for('index', **request.args.to_dict()|update_query_param('view', 'grid')) }}" class="{{ 'active' if current_view == 'grid' else '' }}">Grid View</a>
            <a href="{{ url_for('index', **request.args.to_dict()|update_query_param('view', 'list')) }}" class="{{ 'active' if current_view == 'list' else '' }}">List View</a>
        </div>

        <p class="results-summary">
            {% if total_filtered_books > 0 %}
                Showing books {{ (current_page - 1) * books_per_page + 1 }} - {{ (current_page * books_per_page) if (current_page * books_per_page) < total_filtered_books else total_filtered_books }} of {{ total_filtered_books }} results.
            {% elif books %}
                 Showing {{ books|length }} books.
            {% else %}
                No books match your filters.
            {% endif %}
            (Total {{ total_books_unfiltered }} books in library)
        </p>

        {% if books %}
            {% if current_view == 'grid' %}
            <div class="book-grid">
                {% for book in books %}
                <div class="book-card">
                    <img src="{{ book.coverImg if book.coverImg else 'https://via.placeholder.com/200x300.png?text=No+Cover' }}" alt="{{ book.title }} Cover" onerror="this.onerror=null;this.src='https://via.placeholder.com/200x300.png?text=No+Cover';">
                    <div class="title" title="{{ book.title }}">{{ book.title }}</div>
                    <div class="author" title="{{ book.authors }}">{{ book.authors if book.authors != 'Unknown' else 'Author N/A' }}</div>
                    <div class="rating">{{ get_star_rating_html(book.average_rating, book.ratings_count) }}</div>
                    <div class="details-toggle" onclick="toggleDetails(this)">Show Details ▼</div>
                    <div class="extra-details">
                        <p><strong>Series:</strong> {{ book.series if book.series != 'Unknown' else 'N/A' }}</p>
                        <p><strong>Popularity Score:</strong> {{ "%.2f"|format(book.bayesian_rating) if book.bayesian_rating else 'N/A' }}</p>
                        <p><strong>Genres:</strong> {{ book.genres_display_short }}</p>
                        <p><strong>Pages:</strong> {{ book.num_pages if book.num_pages > 0 else 'N/A' }}</p>
                        <p><strong>Published:</strong> {{ book.display_publication_date }} ({{book.publication_year if book.publication_year > 0 else 'N/A' }})</p>
                        <p><strong>Format:</strong> {{ book.bookFormat if book.bookFormat != 'Unknown' else 'N/A' }}</p>
                        <p><strong>Language:</strong> {{ book.language_code if book.language_code != 'Unknown' else 'N/A' }}</p>
                        {% if book.audible_link %}
                            <a href="{{ book.audible_link }}" target="_blank" class="audible-link">Listen on Audible 🎧</a>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
            {% elif current_view == 'list' %}
            <div class="book-list">
                {% for book in books %}
                <div class="book-list-item">
                    <img src="{{ book.coverImg if book.coverImg else 'https://via.placeholder.com/70x105.png?text=N/A' }}" alt="{{ book.title }} Cover" onerror="this.onerror=null;this.src='https://via.placeholder.com/70x105.png?text=N/A';">
                    <div class="info">
                        <div class="title">{{ book.title }}</div>
                        <div class="author">{{ book.authors if book.authors != 'Unknown' else 'Author N/A' }}</div>
                        <div class="rating">{{ get_star_rating_html(book.average_rating, book.ratings_count, small=True) }}
                            <span class="ratings-count">({{book.ratings_count}} votes)</span>
                            <span class="rating-value-small">| Pop: {{ "%.2f"|format(book.bayesian_rating) if book.bayesian_rating else 'N/A' }}</span>
                        </div>
                        <div class="meta">
                            {{ book.num_pages if book.num_pages > 0 else 'N/A' }} pages | Format: {{ book.bookFormat if book.bookFormat != 'Unknown' else 'N/A' }} | Published: {{ book.display_publication_date }} ({{book.publication_year if book.publication_year > 0 else 'N/A' }})
                            {% if book.audible_link %}
                                | <a href="{{ book.audible_link }}" target="_blank">Audible 🎧</a>
                            {% endif %}
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}

            {% if total_pages > 1 %}
            <div class="pagination">
                {% if current_page > 1 %}
                    <a href="{{ url_for('index', **request.args.to_dict()|update_query_param('page', current_page - 1)) }}">« Prev</a>
                {% else %}
                    <span class="disabled">« Prev</span>
                {% endif %}

                {% for page_num in pagination_window %}
                    {% if page_num == '...' %}
                        <span class="disabled">...</span>
                    {% elif page_num == current_page %}
                        <span class="current-page">{{ page_num }}</span>
                    {% else %}
                        <a href="{{ url_for('index', **request.args.to_dict()|update_query_param('page', page_num)) }}">{{ page_num }}</a>
                    {% endif %}
                {% endfor %}

                {% if current_page < total_pages %}
                    <a href="{{ url_for('index', **request.args.to_dict()|update_query_param('page', current_page + 1)) }}">Next »</a>
                {% else %}
                    <span class="disabled">Next »</span>
                {% endif %}
            </div>
            {% endif %}

        {% else %}
            <p class="no-results">🙁 No books found matching your criteria. Try adjusting the filters! Perhaps widen your search?</p>
        {% endif %}
    </div>

    <script>
        function toggleDetails(element) {
            const extraDetails = element.nextElementSibling;
            if (extraDetails.style.display === "none" || extraDetails.style.display === "") {
                extraDetails.style.display = "block";
                element.textContent = "Hide Details ▲";
            } else {
                extraDetails.style.display = "none";
                element.textContent = "Show Details ▼";
            }
        }
        // Persist range slider values visually
        document.addEventListener('DOMContentLoaded', function() {
            const sliders = [
                {id: 'min_rating', suffix: ''},
                {id: 'min_votes', suffix: ''},
                {id: 'min_liked', suffix: '%'},
                {id: 'pub_year_min', suffix: ''},
                {id: 'pub_year_max', suffix: ''},
                {id: 'max_pages', suffix: ''}
            ];
            sliders.forEach(item => {
                const slider = document.getElementById(item.id);
                const valDisplay = document.getElementById(item.id + '_val');
                if (slider && valDisplay) {
                    valDisplay.textContent = slider.value + item.suffix;
                }
            });
        });
    </script>
</body>
</html>
"""

# --- Jinja Custom Filter for pagination/view links ---
def update_query_params(query_args_dict, key, value):
    """Utility to update a key in a copy of request.args dictionary."""
    args = query_args_dict.copy()
    args[key] = value
    return args

app.jinja_env.filters['update_query_param'] = update_query_params


# --- Flask Route ---
@app.route('/', methods=['GET'])
def index():
    if BOOKS_DF.empty:
        return "Error: Book data could not be loaded. Please check the console, the data file path, and file integrity."

    current_view = request.args.get('view', DEFAULT_DISPLAY_MODE)
    current_page = request.args.get('page', 1, type=int)
    if current_page < 1: current_page = 1
    
    filters = {
        'search_query': request.args.get('search_query', '').strip(),
        'sort_by': request.args.get('sort_by', DEFAULT_SORT_ORDER),
        'min_rating': request.args.get('min_rating', type=float, default=0.0),
        'min_votes': request.args.get('min_votes', type=int, default=0),
        'min_liked': request.args.get('min_liked', type=int, default=0),
        'genres': request.args.getlist('genres'),
        'language': request.args.get('language', ''),
        'book_format': request.args.get('book_format', ''),
        'pub_year_min': request.args.get('pub_year_min', type=int, default=MIN_PUB_YEAR),
        'pub_year_max': request.args.get('pub_year_max', type=int, default=MAX_PUB_YEAR),
        'max_pages': request.args.get('max_pages', type=int, default=MAX_PAGES_FOR_SLIDER)
    }

    filtered_df = BOOKS_DF.copy()

    if filters['search_query']:
        query = filters['search_query'].lower()
        search_cols = ['title', 'authors', 'publisher', 'series'] # Add 'genres_display_full' if you want to search genres text
        filtered_df = filtered_df[
            filtered_df[search_cols].apply(lambda row: row.astype(str).str.lower().str.contains(query, regex=False, na=False).any(), axis=1)
        ]
    
    if filters['min_rating'] > 0:
        filtered_df = filtered_df[filtered_df['average_rating'].fillna(0) >= filters['min_rating']]
    if filters['min_votes'] > 0:
        filtered_df = filtered_df[filtered_df['ratings_count'] >= filters['min_votes']]
    if filters['min_liked'] > 0:
        filtered_df = filtered_df[filtered_df['likedPercent'].fillna(0) >= filters['min_liked']]
    
    if filters['genres']:
        # Ensure genres_list has lists, not NaN, for the filter to work correctly
        clean_genres_list = filtered_df['genres_list'].apply(lambda x: x if isinstance(x, list) else [])
        filtered_df = filtered_df[clean_genres_list.apply(lambda x_genres: any(sg in x_genres for sg in filters['genres']))]
    
    if filters['language']:
        filtered_df = filtered_df[filtered_df['language_code'] == filters['language']]
    if filters['book_format']:
        filtered_df = filtered_df[filtered_df['bookFormat'] == filters['book_format']]

    # Ensure year filters are logical before applying
    actual_min_year = min(filters['pub_year_min'], filters['pub_year_max'])
    actual_max_year = max(filters['pub_year_min'], filters['pub_year_max'])
    
    # Apply publication year filter only if changed from default min/max range, or if they are valid
    if actual_min_year > MIN_PUB_YEAR or actual_max_year < MAX_PUB_YEAR:
         filtered_df = filtered_df[
            (filtered_df['publication_year'] >= actual_min_year) & \
            (filtered_df['publication_year'] <= actual_max_year) & \
            (filtered_df['publication_year'] > 0) # Only include valid years
        ]
    
    if filters['max_pages'] < MAX_PAGES_FOR_SLIDER: # Apply only if not the default max
        filtered_df = filtered_df[filtered_df['num_pages'].fillna(0) <= filters['max_pages']]


    sort_options_map = {
        'popularity_desc': ('bayesian_rating', False, 'ratings_count', False), # Primary: Bayesian, Secondary: ratings_count
        'ratings_count_desc': ('ratings_count', False, 'average_rating', False),
        'average_rating_desc': ('average_rating', False, 'ratings_count', False),
        'title_asc': ('title', True, 'bayesian_rating', False),
        'liked_percent_desc': ('likedPercent', False, 'ratings_count', False),
        'pub_year_desc': ('publication_year', False, 'bayesian_rating', False),
        'pub_year_asc': ('publication_year', True, 'bayesian_rating', False),
        'num_pages_asc': ('num_pages', True, 'bayesian_rating', False),
        'num_pages_desc': ('num_pages', False, 'bayesian_rating', False),
    }
    
    sort_params = sort_options_map.get(filters['sort_by'], sort_options_map[DEFAULT_SORT_ORDER])
    primary_sort_col, primary_asc, secondary_sort_col, secondary_asc = sort_params
    
    if not filtered_df.empty:
        # Ensure sort columns are appropriate types and handle NaNs
        # For numeric, fillna with a value that sorts them last/first as desired or use na_position
        # For string, fillna with empty string for consistent sorting
        if pd.api.types.is_numeric_dtype(filtered_df[primary_sort_col]):
            filtered_df[primary_sort_col] = pd.to_numeric(filtered_df[primary_sort_col], errors='coerce').fillna(-1 if not primary_asc else float('inf')) # Push NaNs to end
        else:
            filtered_df[primary_sort_col] = filtered_df[primary_sort_col].astype(str).fillna('')

        if pd.api.types.is_numeric_dtype(filtered_df[secondary_sort_col]):
            filtered_df[secondary_sort_col] = pd.to_numeric(filtered_df[secondary_sort_col], errors='coerce').fillna(-1 if not secondary_asc else float('inf'))
        else:
            filtered_df[secondary_sort_col] = filtered_df[secondary_sort_col].astype(str).fillna('')

        if primary_sort_col == 'title': # Special handling for case-insensitive title sort
             filtered_df = filtered_df.sort_values(
                by=[primary_sort_col, secondary_sort_col],
                ascending=[primary_asc, secondary_asc],
                key=lambda col: col.str.lower() if col.name == primary_sort_col else col
            )
        else:
            filtered_df = filtered_df.sort_values(
                by=[primary_sort_col, secondary_sort_col],
                ascending=[primary_asc, secondary_asc]
            )


    total_filtered_books = len(filtered_df)
    total_pages = (total_filtered_books + BOOKS_PER_PAGE - 1) // BOOKS_PER_PAGE
    if current_page > total_pages and total_pages > 0:
        current_page = total_pages # Adjust if current page is out of bounds after filtering

    start_index = (current_page - 1) * BOOKS_PER_PAGE
    end_index = start_index + BOOKS_PER_PAGE
    paginated_books_df = filtered_df.iloc[start_index:end_index]

    # Pagination window logic (e.g., 1 ... 4 5 6 ... 10)
    window_size = 2 # number of pages around current page
    pagination_window = []
    if total_pages <= 1:
        pagination_window = []
    elif total_pages <= 5 + (window_size * 2) : # show all if not too many pages
        pagination_window = list(range(1, total_pages + 1))
    else:
        pagination_window.append(1)
        if current_page > window_size + 2: pagination_window.append('...')
        
        for i in range(max(2, current_page - window_size), min(total_pages, current_page + window_size + 1)):
            if i not in pagination_window : pagination_window.append(i)

        if current_page < total_pages - (window_size + 1) : pagination_window.append('...')
        if total_pages not in pagination_window: pagination_window.append(total_pages)


    sort_options_display = {
        'popularity_desc': 'Popularity (Best Match)',
        'ratings_count_desc': 'Ratings Count (High to Low)',
        'average_rating_desc': 'Average Rating (High to Low)',
        'liked_percent_desc': 'Liked Percent (High to Low)',
        'pub_year_desc': 'Publication Year (Newest)',
        'pub_year_asc': 'Publication Year (Oldest)',
        'num_pages_desc': 'Pages (Longest)',
        'num_pages_asc': 'Pages (Shortest)',
        'title_asc': 'Title (A-Z)'
    }

    return render_template_string(
        HTML_TEMPLATE,
        books=paginated_books_df.to_dict('records'),
        current_view=current_view,
        filters=filters,
        sort_options=sort_options_display,
        all_genres=ALL_GENRES,
        all_languages=ALL_LANGUAGES,
        all_formats=ALL_FORMATS,
        min_pub_year=MIN_PUB_YEAR,
        max_pub_year=MAX_PUB_YEAR,
        max_ratings_slider=MAX_RATINGS_COUNT_FOR_SLIDER,
        ratings_slider_step=RATINGS_COUNT_SLIDER_STEP,
        max_pages_slider=MAX_PAGES_FOR_SLIDER,
        pages_slider_step=PAGES_SLIDER_STEP,
        get_star_rating_html=get_star_rating_html,
        total_books_unfiltered=len(BOOKS_DF),
        total_filtered_books=total_filtered_books,
        current_page=current_page,
        total_pages=total_pages,
        books_per_page=BOOKS_PER_PAGE,
        pagination_window=pagination_window,
        request=request # Pass request object for args in template
    )

if __name__ == '__main__':
    if BOOKS_DF.empty:
        print("Halting app start: Book data failed to load. Check 'books .csv' file and console output.")
    else:
        print(f"Successfully loaded {len(BOOKS_DF)} books.")
        print(f"Min publication year: {MIN_PUB_YEAR}, Max: {MAX_PUB_YEAR}")
        print(f"Found {len(ALL_GENRES)} unique genres. {len(ALL_LANGUAGES)} languages. {len(ALL_FORMATS)} formats.")
        print(f"Default sort order: {DEFAULT_SORT_ORDER}")
        app.run(debug=True)