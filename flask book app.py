import pandas as pd
import numpy as np
import ast
import urllib.parse
from datetime import datetime

from flask import Flask, render_template_string, request
from markupsafe import Markup


# --- Configuration ---
DATA_PATH = 'books .csv'  # Ensure this file is in the same directory as the script
DEFAULT_DISPLAY_MODE = 'grid'
BOOKS_PER_PAGE = 30 # For pagination, if implemented (not in this initial version for simplicity)

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

# --- Data Loading and Cleaning Function (Adapted from Streamlit) ---
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

        df['first_publication_date_dt'] = pd.to_datetime(df.get('first_publication_date'), errors='coerce', format='%m/%d/%y', infer_datetime_format=False)
        df['publication_date_edition_dt'] = pd.to_datetime(df.get('publication_date_edition'), errors='coerce', format='%m/%d/%y', infer_datetime_format=False)
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

        # Audible Link
        def create_audible_link_url(title_str):
            if pd.isna(title_str) or title_str == '' or title_str == 'Unknown': return None
            base = "https://www.audible.in/search?"
            params = {'keywords': str(title_str), 'k': str(title_str)}
            return f"{base}{urllib.parse.urlencode(params)}"
        df['audible_link'] = df['title'].apply(create_audible_link_url)
        
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An critical error occurred during data loading: {e}")
        return pd.DataFrame()

# Load data once when the app starts
BOOKS_DF = load_and_clean_data(DATA_PATH)
if not BOOKS_DF.empty:
    ALL_GENRES = sorted(list(set(g for sublist in BOOKS_DF['genres_list'] for g in sublist if g)))
    ALL_LANGUAGES = sorted([lang for lang in BOOKS_DF['language_code'].dropna().unique().tolist() if lang != 'Unknown'])
    ALL_FORMATS = sorted([fmt for fmt in BOOKS_DF['bookFormat'].dropna().unique().tolist() if fmt != 'Unknown'])
    MIN_PUB_YEAR = int(BOOKS_DF['publication_year'][BOOKS_DF['publication_year'] > 0].min() if not BOOKS_DF[BOOKS_DF['publication_year'] > 0].empty else 1800)
    MAX_PUB_YEAR = int(BOOKS_DF['publication_year'].max() if not BOOKS_DF.empty else datetime.now().year)
else: # Fallbacks if data loading fails
    ALL_GENRES, ALL_LANGUAGES, ALL_FORMATS = [], [], []
    MIN_PUB_YEAR, MAX_PUB_YEAR = 1800, datetime.now().year


# --- Flask App Initialization ---
app = Flask(__name__)

# --- Utility Functions ---
def get_star_rating_html(rating_val, ratings_count=None):
    if pd.isna(rating_val): return "<span class='stars-na'>N/A</span>"
    try:
        rating_val = float(rating_val)
        full_stars = int(rating_val)
        half_star = "★" if rating_val - full_stars >= 0.75 else ("½" if rating_val - full_stars >= 0.25 else "") # Adjusted half star logic
        empty_stars = 5 - full_stars - (1 if half_star else 0)
        stars_html = f"<span class='stars'>{'★' * full_stars}{half_star}{'☆' * empty_stars}</span> <span class='rating-value'>({rating_val:.2f})</span>"
        if ratings_count is not None:
            stars_html += f" <span class='ratings-count'>({ratings_count:,} ratings)</span>"
        return Markup(stars_html)
    except: return "<span class='stars-na'>Error</span>"


# --- HTML Template (Embedded) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Book Explorer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            background-color: #1a1a1a; /* Darker background */
            color: #e0e0e0; /* Light text */
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 280px;
            background-color: #2c2c2c; /* Slightly lighter dark for sidebar */
            padding: 20px;
            border-right: 1px solid #444;
            overflow-y: auto;
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
        }
        .main-content {
            margin-left: 300px; /* Adjust based on sidebar width + padding */
            padding: 20px;
            width: calc(100% - 300px);
            overflow-y: auto;
        }
        h1, h2, h3 {
            color: #61dafb; /* Light blue accent */
        }
        h1 { text-align: center; margin-bottom: 20px; }
        .filter-group {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #333;
            border-radius: 6px;
        }
        .filter-group h3 { margin-top: 0; font-size: 1.1em; color: #82aaff; border-bottom: 1px solid #555; padding-bottom: 5px;}
        label { display: block; margin-bottom: 5px; font-size: 0.9em; }
        input[type="text"], input[type="number"], select {
            width: calc(100% - 16px);
            padding: 8px;
            margin-bottom: 10px;
            border-radius: 4px;
            border: 1px solid #555;
            background-color: #444;
            color: #e0e0e0;
            box-sizing: border-box;
        }
        input[type="range"] { width: calc(100% - 16px); }
        .slider-labels { display: flex; justify-content: space-between; font-size: 0.8em; margin-top: -5px; margin-bottom:10px;}

        button, input[type="submit"] {
            background-color: #61dafb;
            color: #1a1a1a;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.2s;
            width: 100%;
        }
        button:hover, input[type="submit"]:hover { background-color: #82aaff; }
        
        .view-switcher { text-align: center; margin-bottom: 20px; }
        .view-switcher a {
            text-decoration: none; color: #61dafb;
            padding: 8px 12px; margin: 0 5px; border-radius: 4px;
            border: 1px solid #61dafb;
        }
        .view-switcher a.active { background-color: #61dafb; color: #1a1a1a; }

        /* Grid View */
        .book-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); /* Responsive grid */
            gap: 20px;
        }
        .book-card {
            background-color: #2c2c2c;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .book-card:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.3); }
        .book-card img {
            max-width: 100%;
            height: 200px; /* Fixed height for cover */
            object-fit: contain; /* Use contain to see whole image */
            border-radius: 4px;
            margin-bottom: 10px;
            background-color: #383838; /* Placeholder bg for images */
        }
        .book-card .title { font-size: 1em; font-weight: bold; color: #e0e0e0; margin-bottom: 5px; line-height:1.2; height: 2.4em; overflow:hidden; }
        .book-card .author { font-size: 0.85em; color: #bbb; margin-bottom: 8px; height: 2em; overflow:hidden;}
        .book-card .rating { font-size: 0.8em; margin-bottom: 8px; }
        .stars { color: #ffc107; } /* Yellow stars */
        .stars-na, .rating-value, .ratings-count { color: #aaa; font-size:0.9em; }
        .book-card .details-toggle {
            font-size: 0.8em; color: #61dafb; cursor: pointer; text-decoration: underline;
            margin-top: auto; padding-top: 5px;
        }
        .book-card .extra-details {
            display: none; /* Hidden by default */
            font-size: 0.8em; text-align: left; margin-top: 10px;
            border-top: 1px solid #444; padding-top: 8px; color: #ccc;
        }
        .book-card .extra-details p { margin: 3px 0; }
        .book-card .audible-link { display:block; margin-top:10px; font-size:0.85em;}


        /* List View */
        .book-list-item {
            background-color: #2c2c2c;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 10px 15px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .book-list-item img {
            width: 60px; height: 90px; object-fit: contain;
            border-radius: 4px; margin-right: 15px; background-color: #383838;
        }
        .book-list-item .info { flex-grow: 1; }
        .book-list-item .title { font-size: 1.1em; font-weight: bold; color: #e0e0e0; }
        .book-list-item .author { font-size: 0.9em; color: #bbb; }
        .book-list-item .rating { font-size: 0.85em; margin-top: 5px; }
        .book-list-item .meta { font-size: 0.8em; color: #999; margin-top:5px; }

        .no-results { text-align:center; padding: 30px; font-size: 1.2em; color: #888;}
        .results-summary { text-align: center; margin-bottom: 15px; font-size: 0.9em; color: #aaa;}
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>📚 Book Filters</h2>
        <form method="GET" action="/">
            <input type="hidden" name="view" value="{{ current_view }}">

            <div class="filter-group">
                <h3>Search & Sort</h3>
                <label for="search_query">Search Term:</label>
                <input type="text" name="search_query" id="search_query" value="{{ filters.search_query or '' }}">
                
                <label for="sort_by">Sort by:</label>
                <select name="sort_by" id="sort_by">
                    {% for val, display in sort_options.items() %}
                    <option value="{{ val }}" {% if filters.sort_by == val %}selected{% endif %}>{{ display }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="filter-group">
                <h3>Quality & Engagement</h3>
                <label for="min_rating">Min Avg. Rating: <span id="min_rating_val">{{ filters.min_rating or 0.0 }}</span></label>
                <input type="range" name="min_rating" id="min_rating" min="0" max="5" step="0.1" value="{{ filters.min_rating or 0.0 }}" oninput="document.getElementById('min_rating_val').textContent = this.value">
                
                <label for="min_votes">Min Ratings Count: <span id="min_votes_val">{{ filters.min_votes or 0 }}</span></label>
                <input type="range" name="min_votes" id="min_votes" min="0" max="1000" step="10" value="{{ filters.min_votes or 0 }}" oninput="document.getElementById('min_votes_val').textContent = this.value">
                 <div class="slider-labels"><span>0</span><span>1000+</span></div>

                <label for="min_liked">Min Liked Percent: <span id="min_liked_val">{{ filters.min_liked or 0 }}</span>%</label>
                <input type="range" name="min_liked" id="min_liked" min="0" max="100" step="1" value="{{ filters.min_liked or 0 }}" oninput="document.getElementById('min_liked_val').textContent = this.value">
                 <div class="slider-labels"><span>0%</span><span>100%</span></div>
            </div>

            <div class="filter-group">
                <h3>Content Attributes</h3>
                <label for="genres">Genres (select multiple):</label>
                <select name="genres" id="genres" multiple size="5">
                    {% for genre in all_genres %}
                    <option value="{{ genre }}" {% if genre in filters.genres %}selected{% endif %}>{{ genre }}</option>
                    {% endfor %}
                </select>

                <label for="language">Language:</label>
                <select name="language" id="language">
                    <option value="">All</option>
                    {% for lang in all_languages %}
                    <option value="{{ lang }}" {% if filters.language == lang %}selected{% endif %}>{{ lang }}</option>
                    {% endfor %}
                </select>

                <label for="book_format">Format:</label>
                <select name="book_format" id="book_format">
                    <option value="">All</option>
                    {% for fmt in all_formats %}
                    <option value="{{ fmt }}" {% if filters.book_format == fmt %}selected{% endif %}>{{ fmt }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <div class="filter-group">
                <h3>Publication & Length</h3>
                <label for="pub_year_min">Min Pub. Year: <span id="pub_year_min_val">{{ filters.pub_year_min or min_pub_year }}</span></label>
                <input type="range" name="pub_year_min" id="pub_year_min" min="{{min_pub_year}}" max="{{max_pub_year}}" step="1" value="{{ filters.pub_year_min or min_pub_year }}" oninput="document.getElementById('pub_year_min_val').textContent = this.value">
                <div class="slider-labels"><span>{{min_pub_year}}</span><span>{{max_pub_year}}</span></div>

                <label for="pub_year_max">Max Pub. Year: <span id="pub_year_max_val">{{ filters.pub_year_max or max_pub_year }}</span></label>
                <input type="range" name="pub_year_max" id="pub_year_max" min="{{min_pub_year}}" max="{{max_pub_year}}" step="1" value="{{ filters.pub_year_max or max_pub_year }}" oninput="document.getElementById('pub_year_max_val').textContent = this.value">
                 <div class="slider-labels"><span>{{min_pub_year}}</span><span>{{max_pub_year}}</span></div>

                <label for="max_pages">Max Pages: <span id="max_pages_val">{{ filters.max_pages or 1000 }}</span></label>
                <input type="range" name="max_pages" id="max_pages" min="0" max="2000" step="50" value="{{ filters.max_pages or 1000 }}" oninput="document.getElementById('max_pages_val').textContent = this.value">
                 <div class="slider-labels"><span>0</span><span>2000+</span></div>
            </div>

            <input type="submit" value="Apply Filters">
        </form>
    </div>

    <div class="main-content">
        <h1>Book Explorer</h1>
        <div class="view-switcher">
            <a href="{{ url_for('index', view='grid', **query_params_no_view) }}" class="{{ 'active' if current_view == 'grid' else '' }}">Grid View</a>
            <a href="{{ url_for('index', view='list', **query_params_no_view) }}" class="{{ 'active' if current_view == 'list' else '' }}">List View</a>
        </div>
        <p class="results-summary">Showing {{ books|length }} of {{ total_books_unfiltered }} books.</p>

        {% if books %}
            {% if current_view == 'grid' %}
            <div class="book-grid">
                {% for book in books %}
                <div class="book-card">
                    <img src="{{ book.coverImg if book.coverImg else 'https://via.placeholder.com/150x220.png?text=No+Cover' }}" alt="{{ book.title }} Cover" onerror="this.onerror=null;this.src='https://via.placeholder.com/150x220.png?text=No+Cover';">
                    <div class="title" title="{{ book.title }}">{{ book.title }}</div>
                    <div class="author" title="{{ book.authors }}">{{ book.authors }}</div>
                    <div class="rating">{{ get_star_rating_html(book.average_rating, book.ratings_count) }}</div>
                    <div class="details-toggle" onclick="toggleDetails(this)">Show Details ▼</div>
                    <div class="extra-details">
                        <p><strong>Series:</strong> {{ book.series if book.series != 'Unknown' else 'N/A' }}</p>
                        <p><strong>Genres:</strong> {{ book.genres_display_short }}</p>
                        <p><strong>Pages:</strong> {{ book.num_pages }}</p>
                        <p><strong>Published:</strong> {{ book.display_publication_date }}</p>
                        <p><strong>Format:</strong> {{ book.bookFormat }}</p>
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
                    <img src="{{ book.coverImg if book.coverImg else 'https://via.placeholder.com/60x90.png?text=N/A' }}" alt="{{ book.title }} Cover" onerror="this.onerror=null;this.src='https://via.placeholder.com/60x90.png?text=N/A';">
                    <div class="info">
                        <div class="title">{{ book.title }}</div>
                        <div class="author">{{ book.authors }}</div>
                        <div class="rating">{{ get_star_rating_html(book.average_rating) }} ({{book.ratings_count}} votes)</div>
                        <div class="meta">
                            {{ book.num_pages }} pages | Format: {{ book.bookFormat }} | Published: {{ book.display_publication_date }}
                            {% if book.audible_link %}
                                | <a href="{{ book.audible_link }}" target="_blank">Audible 🎧</a>
                            {% endif %}
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        {% else %}
            <p class="no-results">No books match your current filters. Try adjusting them!</p>
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
        // Persist range slider values visually on page load if set
        document.addEventListener('DOMContentLoaded', function() {
            const sliders = ['min_rating', 'min_votes', 'min_liked', 'pub_year_min', 'pub_year_max', 'max_pages'];
            sliders.forEach(id => {
                const slider = document.getElementById(id);
                const valDisplay = document.getElementById(id + '_val');
                if (slider && valDisplay) {
                    valDisplay.textContent = slider.value + (id === 'min_liked' ? '%' : '');
                }
            });
        });
    </script>
</body>
</html>
"""

# --- Flask Route ---
@app.route('/', methods=['GET'])
def index():
    if BOOKS_DF.empty:
        return "Error: Book data could not be loaded. Please check the console and the data file."

    current_view = request.args.get('view', DEFAULT_DISPLAY_MODE)
    
    # --- Get filter values from request ---
    filters = {
        'search_query': request.args.get('search_query', ''),
        'sort_by': request.args.get('sort_by', 'ratings_count_desc'),
        'min_rating': request.args.get('min_rating', type=float, default=0.0),
        'min_votes': request.args.get('min_votes', type=int, default=0),
        'min_liked': request.args.get('min_liked', type=int, default=0),
        'genres': request.args.getlist('genres'), # For multi-select
        'language': request.args.get('language', ''),
        'book_format': request.args.get('book_format', ''),
        'pub_year_min': request.args.get('pub_year_min', type=int, default=MIN_PUB_YEAR),
        'pub_year_max': request.args.get('pub_year_max', type=int, default=MAX_PUB_YEAR),
        'max_pages': request.args.get('max_pages', type=int, default=2000) # Assuming a practical max
    }

    # --- Apply filters ---
    filtered_df = BOOKS_DF.copy()

    if filters['search_query']:
        query = filters['search_query'].lower()
        search_cols = ['title', 'authors', 'publisher', 'series']
        filtered_df = filtered_df[
            filtered_df[search_cols].apply(lambda row: row.astype(str).str.lower().str.contains(query, regex=False).any(), axis=1)
        ]
    
    if filters['min_rating'] > 0:
        filtered_df = filtered_df[filtered_df['average_rating'].fillna(0) >= filters['min_rating']]
    if filters['min_votes'] > 0:
        filtered_df = filtered_df[filtered_df['ratings_count'] >= filters['min_votes']]
    if filters['min_liked'] > 0:
        filtered_df = filtered_df[filtered_df['likedPercent'].fillna(0) >= filters['min_liked']]
    
    if filters['genres']:
        filtered_df = filtered_df[filtered_df['genres_list'].apply(lambda x_genres: any(sg in x_genres for sg in filters['genres']))]
    
    if filters['language']:
        filtered_df = filtered_df[filtered_df['language_code'] == filters['language']]
    if filters['book_format']:
        filtered_df = filtered_df[filtered_df['bookFormat'] == filters['book_format']]

    if filters['pub_year_min'] > MIN_PUB_YEAR or filters['pub_year_max'] < MAX_PUB_YEAR :
        filtered_df = filtered_df[(filtered_df['publication_year'] >= filters['pub_year_min']) & (filtered_df['publication_year'] <= filters['pub_year_max'])]
    
    if filters['max_pages'] < 2000 : # Assuming 2000 is a "show all" default
        filtered_df = filtered_df[filtered_df['num_pages'] <= filters['max_pages']]


    # --- Apply Sorting ---
    sort_options_map = {
        'ratings_count_desc': ('ratings_count', False), 'average_rating_desc': ('average_rating', False),
        'title_asc': ('title', True), 'liked_percent_desc': ('likedPercent', False),
        'pub_year_desc': ('publication_year', False), 'pub_year_asc': ('publication_year', True),
        'num_pages_asc': ('num_pages', True), 'num_pages_desc': ('num_pages', False)
    }
    sort_col, sort_asc = sort_options_map.get(filters['sort_by'], ('ratings_count', False))
    
    if not filtered_df.empty:
        # Handle NaNs in sort column: place them last when ascending, first when descending
        na_position = 'last' if sort_asc else 'first'
        if pd.api.types.is_numeric_dtype(filtered_df[sort_col]):
             filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc, na_position=na_position)
        else: # For string columns, fillna before sort to avoid type errors with na_position
             filtered_df[sort_col] = filtered_df[sort_col].astype(str) # Ensure string type
             filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc, na_position=na_position, key=lambda col: col.str.lower())


    # Prepare query parameters for view switcher links (to keep filters)
    query_params_no_view = {k: v for k, v in request.args.items() if k != 'view'}


    # Sort options for the dropdown
    sort_options_display = {
        'ratings_count_desc': 'Ratings Count (High to Low)', 'average_rating_desc': 'Average Rating (High to Low)',
        'liked_percent_desc': 'Liked Percent (High to Low)', 'pub_year_desc': 'Publication Year (Newest)',
        'pub_year_asc': 'Publication Year (Oldest)', 'num_pages_desc': 'Pages (Longest)',
        'num_pages_asc': 'Pages (Shortest)', 'title_asc': 'Title (A-Z)'
    }

    return render_template_string(
        HTML_TEMPLATE,
        books=filtered_df.to_dict('records'),
        current_view=current_view,
        filters=filters,
        sort_options=sort_options_display,
        all_genres=ALL_GENRES,
        all_languages=ALL_LANGUAGES,
        all_formats=ALL_FORMATS,
        min_pub_year=MIN_PUB_YEAR,
        max_pub_year=MAX_PUB_YEAR,
        get_star_rating_html=get_star_rating_html, # Pass utility function to template
        total_books_unfiltered=len(BOOKS_DF),
        query_params_no_view=query_params_no_view
    )

if __name__ == '__main__':
    if BOOKS_DF.empty:
        print("Halting app start: Book data failed to load.")
    else:
        print(f"Successfully loaded {len(BOOKS_DF)} books.")
        print(f"Min publication year: {MIN_PUB_YEAR}, Max: {MAX_PUB_YEAR}")
        print(f"Found {len(ALL_GENRES)} unique genres.")
        app.run(debug=True)
# ```

# **How to Use:**

# 1.  **Save:** Save the entire code block above as a single Python file (e.g., `app.py`).
# 2.  **Data File:** Make sure your `books .csv` file (with the space in the name) is in the *same directory* as `app.py`.
# 3.  **Install Flask & Pandas:** If you don't have them, install them:
#     ```bash
#     pip install Flask pandas
#     ```
# 4.  **Run:** Open your terminal or command prompt, navigate to the directory where you saved the file, and run:
#     ```bash
#     python app.py
#     ```
# 5.  **Access:** Open your web browser and go to `http://127.0.0.1:5000/`.

# **Features Implemented:**

# * **Single File:** All Flask, HTML, CSS, and JS are in one `.py` file.
# * **UI:** A dark-themed, responsive UI with a fixed sidebar for filters and a main content area for books.
# * **Filters:**
#     * Search (title, author, publisher, series).
#     * Sort by various criteria.
#     * Minimum Average Rating (range slider).
#     * Minimum Ratings Count (range slider).
#     * Minimum Liked Percent (range slider).
#     * Genres (multi-select).
#     * Language (dropdown).
#     * Book Format (dropdown).
#     * Publication Year Min/Max (range sliders).
#     * Max Pages (range slider).
# * **Display Views:**
#     * **Grid View (Default):** Shows book covers, titles, authors, and star ratings in a responsive grid (aims for many cards per row, adjusts with screen size). Includes a "Show Details" toggle for more info within each card.
#     * **List View:** A more compact list with smaller cover images and key details.
#     * View switching is done via links that reload the page with a `view` query parameter.
# * **Star Ratings:** A utility function generates HTML for star ratings.
# * **Audible Links:** Included in both grid (details) and list views.
# * **JavaScript:** Minimal JS is used for:
#     * Updating the displayed values for range sliders.
#     * Toggling the "extra details" section in the grid view cards.
#     * Ensuring range slider values are visually updated on page load if they were set by query params.

# This is a comprehensive solution that should give you a good starting point for your Flask-based book explorer! You can further customize the styling and add more advanced features as need