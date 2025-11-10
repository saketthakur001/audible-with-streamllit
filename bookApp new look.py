import streamlit as st
import pandas as pd
import re
import urllib.parse
import numpy as np
import ast # For safely evaluating string representations of lists (like genres)
from datetime import datetime

# --- Constants and Configuration ---
DATA_PATH = 'books .csv' # Updated CSV filename with space
DEFAULT_MIN_VOTES_THRESHOLD = 50
DEFAULT_MIN_RATING_THRESHOLD = 3.5
DEFAULT_LIKED_PERCENT_THRESHOLD = 75
DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES = 100
DEFAULT_VOTES_POWER = 1.0

COLUMN_NAME_MAP = {
    'bookId': 'book_id_str', 'author': 'authors', 'rating': 'average_rating',
    'numRatings': 'ratings_count', 'pages': 'num_pages', 'language': 'language_code',
    'publishDate': 'publication_date_edition', 'firstPublishDate': 'first_publication_date',
}
REQUIRED_ORIGINAL_COLS = [
    'title', 'author', 'rating', 'numRatings', 'pages', 'language',
    'publisher', 'genres', 'coverImg'
]

@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
        for col in REQUIRED_ORIGINAL_COLS:
            if col not in df.columns:
                st.error(f"Error: Essential source column '{col}' not found.")
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
        df['genres_display'] = df['genres_list'].apply(lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '') if x else 'N/A') # Show top 3 genres

        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An critical error occurred during data loading: {e}")
        return pd.DataFrame()

# --- Page Setup and Styling ---
st.set_page_config(layout="wide", page_title="📚 Advanced Book Portal 📖")

st.markdown("""
<style>
    body {
        color: #EAEAEA; background-color: #121212; /* Darker background, lighter text */
        font-family: 'Roboto', 'Segoe UI', sans-serif;
    }
    /* Dataframe image styling */
    .stDataFrame img {
        max-height: 120px; /* Increased from 80px for table view */
        width: auto; object-fit: contain; display: block; margin: auto;
        border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .st-emotion-cache-1cypcdb { /* Sidebar */
        background-color: #1E1E1E; border-right: 1px solid #333; padding: 20px;
    }
    .st-emotion-cache-1jm9le { padding: 25px; } /* Main content area */

    h1, h2, h3, h4, h5, h6 { color: #BB86FC; } /* Primary accent for headers (Material Design Dark Purple) */
    h1 { color: #03DAC6; } /* Secondary accent for main title (Material Design Dark Teal) */
    hr { border-top: 1px solid #333; }

    /* Input widgets more integrated with dark theme */
    .stTextInput input, .stSlider div[data-baseweb="slider"], .stMultiSelect div[data-baseweb="select"] > div, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 6px; border: 1px solid #444; background-color: #2C2C2C; color: #EAEAEA;
    }
    .stMultiSelect div[data-baseweb="select"] > div, .stSelectbox div[data-baseweb="select"] > div { padding: 6px; }
    .stSlider span[role="slider"] { background-color: #BB86FC !important; }
    .stRadio div[data-baseweb="radio"] { background-color: #2C2C2C; padding:8px; border-radius:6px;}


    .stButton>button {
        border: 1px solid #03DAC6; background-color: transparent; color: #03DAC6;
        padding: 8px 16px; border-radius: 6px; font-weight: bold;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background-color: #03DAC6; color: #121212; transform: translateY(-2px); }
    .stButton>button:active { transform: translateY(0px); }
    
    a { color: #81D4FA; } a:hover { color: #B3E5FC; }

    /* Card Styling for Grid/Card Views */
    .book-grid-container { display: flex; flex-wrap: wrap; gap: 20px; }
    .book-card {
        background-color: #1E1E1E; border: 1px solid #333; border-radius: 8px;
        padding: 15px; margin-bottom:0px; /* margin handled by gap */
        display: flex; flex-direction: column;
        height: 100%; /* Critical for consistent height in columns */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
    }
    .book-card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.5); }
    .book-card-image-container { text-align: center; margin-bottom: 12px; }
    .book-card img {
        max-width: 100%; width: auto; max-height: 200px; /* Good size for cards */
        object-fit: contain; border-radius: 4px;
    }
    .card-title { font-size: 1.1em; font-weight: bold; color: #03DAC6; margin-bottom: 4px; line-height: 1.3; }
    .card-author { font-size: 0.9em; color: #B0B0B0; margin-bottom: 8px; }
    .card-info { font-size: 0.85em; color: #AAAAAA; margin-bottom: 3px; }
    .card-info strong { color: #CFCECE; }
    .card-actions { margin-top: auto; padding-top:10px; /* Pushes actions to bottom */ }
</style>
""", unsafe_allow_html=True)

st.title("📚 Advanced Book Portal 📖")

df_original = load_and_clean_data(DATA_PATH)
if df_original.empty: st.stop()
df = df_original.copy()
C = df['average_rating'].mean() if pd.notna(df['average_rating'].mean()) else 0.0

# --- Sidebar ---
st.sidebar.header("📜 Filters & Scrolls 📜")
# --- Display Mode Selector ---
display_mode = st.sidebar.radio(
    "🖼️ Display Mode",
    ["Enhanced Table", "Image Grid", "Card View (Details on Click)"],
    index=1, # Default to Image Grid
    key="display_mode_selector",
    help="Choose how to view the book listings."
)
st.sidebar.markdown("---")

search_query = st.sidebar.text_input("🔍 Search Titles, Authors, Publishers", help="Case-insensitive search.").lower()
st.sidebar.markdown("---")

st.sidebar.subheader("🪄 Sort Your Findings")
sort_options_map = {
    'Custom Score (Rating * Votes^p) [Recommended]': 'rating_votes_power_score',
    'Weighted Score (IMDb Style)': 'weighted_score', 'Average Rating (High to Low)': 'average_rating',
    'Ratings Count (High to Low)': 'ratings_count', 'Liked Percentage (High to Low)': 'likedPercent',
    'Publication Year (Newest First)': 'pub_year_newest', 'Publication Year (Oldest First)': 'pub_year_oldest',
    'Page Count (Shortest First)': 'num_pages_asc', 'Page Count (Longest First)': 'num_pages_desc',
    'Price (Low to High)': 'price_asc', 'Price (High to Low)': 'price_desc',
}
default_sort_key = 'Custom Score (Rating * Votes^p) [Recommended]'
sort_by_display_name = st.sidebar.selectbox(
    "Sort by", options=list(sort_options_map.keys()),
    index=list(sort_options_map.keys()).index(default_sort_key)
)
selected_sort_method = sort_options_map[sort_by_display_name]

with st.sidebar.expander("🔧 Sorting Parameters", expanded=False):
    p = st.slider("Votes Power ('p') for Custom Score", 0.0, 2.0, DEFAULT_VOTES_POWER, 0.05)
    m_val = st.slider("Anchor Votes ('m') for Weighted Score", 0, 1000, DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES, 10)

st.sidebar.markdown("---")
with st.sidebar.expander("🌟 Quality & Engagement", expanded=True):
    min_votes_threshold = st.slider("Min Ratings Count", 0, int(df['ratings_count'].max() if not df.empty else 1000), DEFAULT_MIN_VOTES_THRESHOLD, 10)
    min_rating_threshold = st.slider("Min Average Rating", 0.0, 5.0, DEFAULT_MIN_RATING_THRESHOLD, 0.1)
    min_liked_percent = st.slider("Min Liked Percent", 0, 100, DEFAULT_LIKED_PERCENT_THRESHOLD, 1)

with st.sidebar.expander("📚 Content Attributes"):
    all_genres_flat = sorted(list(set(g for sublist in df['genres_list'] for g in sublist if g)))
    selected_genres = st.multiselect("Filter by Genres (ANY selected)", options=all_genres_flat, default=[])
    
    unique_languages = sorted(df['language_code'].dropna().unique().tolist())
    if unique_languages and not (len(unique_languages) == 1 and unique_languages[0] == 'Unknown'):
        all_languages_option = ['All'] + [lang for lang in unique_languages if lang != 'Unknown']
        selected_languages_multiselect = st.multiselect("Language", options=all_languages_option, default=['All'])

    unique_formats = sorted(df['bookFormat'].dropna().unique().tolist())
    if unique_formats and not (len(unique_formats) == 1 and unique_formats[0] == 'Unknown'):
         selected_formats = st.multiselect("Book Format", options=[fmt for fmt in unique_formats if fmt != 'Unknown'], default=[])

with st.sidebar.expander("📖 Publication & Length"):
    min_year = int(df['publication_year'][df['publication_year'] > 0].min() if not df[df['publication_year'] > 0].empty else 1800)
    max_year = int(df['publication_year'].max() if not df.empty else datetime.now().year)
    if min_year < max_year : # ensure valid range
      selected_year_range = st.slider("Publication Year Range", min_year, max_year, (min_year, max_year))
    else: # Fallback if data is sparse for years
      st.info("Publication year data not sufficient for range filter.")
      selected_year_range = (min_year, max_year) # still assign for downstream logic

    max_pg = int(df['num_pages'].max() if not df.empty else 1000)
    min_pg_sel, max_pg_selected = st.slider("Page Count Range", 0, max_pg, (0, max_pg), 10)

if 'price' in df.columns and df['price'].nunique() > 1 and df['price'].sum() > 0:
    with st.sidebar.expander("💰 Price"):
        min_price = float(df['price'][df['price'] > 0].min(skipna=True) if not df[df['price'] > 0].empty else 0)
        max_price = float(df['price'].max(skipna=True) if not df.empty else 100)
        if max_price <= min_price: max_price = min_price + 10 if min_price > 0 else 100
        selected_price_range = st.slider("Price Range", min_price, max_price, (min_price, max_price), max(0.1, (max_price-min_price)/100))


# --- Score Calculations (Simplified for brevity, full logic from previous step assumed) ---
df['weighted_score'] = ((df['ratings_count'] / (df['ratings_count'] + m_val)) * df['average_rating'] + (m_val / (df['ratings_count'] + m_val)) * C) if m_val > 0 else df['average_rating']
df.loc[df['average_rating'].isna(), 'weighted_score'] = np.nan

df['rating_votes_power_score'] = (df['average_rating'] * (df['ratings_count'] ** p)) if p > 0 else df['average_rating']
df.loc[df['average_rating'].isna() | ((df['ratings_count'] == 0) & (p > 0)), 'rating_votes_power_score'] = 0
df['rating_votes_power_score'] = df['rating_votes_power_score'].fillna(0)


# --- Apply Filters (Simplified for brevity, full logic from previous step assumed) ---
current_mask = pd.Series([True] * len(df), index=df.index)
if search_query:
    search_cols = ['title', 'authors', 'publisher', 'series']
    current_mask &= df[search_cols].apply(lambda row: row.astype(str).str.lower().str.contains(search_query, regex=False).any(), axis=1)
current_mask &= (df['ratings_count'] >= min_votes_threshold) & \
                (df['average_rating'].fillna(0) >= min_rating_threshold) & \
                (df['likedPercent'].fillna(0) >= min_liked_percent)
if selected_genres: current_mask &= df['genres_list'].apply(lambda x_genres: any(sg in x_genres for sg in selected_genres))
if 'selected_languages_multiselect' in locals() and 'All' not in selected_languages_multiselect and selected_languages_multiselect:
    current_mask &= df['language_code'].isin(selected_languages_multiselect)
if 'selected_formats' in locals() and selected_formats: current_mask &= df['bookFormat'].isin(selected_formats)
if 'selected_year_range' in locals(): current_mask &= (df['publication_year'] >= selected_year_range[0]) & (df['publication_year'] <= selected_year_range[1])
current_mask &= (df['num_pages'] >= min_pg_sel) & (df['num_pages'] <= max_pg_selected)
if 'selected_price_range' in locals() and 'price' in df.columns: current_mask &= (df['price'] >= selected_price_range[0]) & (df['price'] <= selected_price_range[1])
filtered_df = df[current_mask].copy()

# --- Apply Sorting (Simplified for brevity, full logic from previous step assumed) ---
# ... (Full sorting logic as in previous version should be here) ...
# For this example, just a basic sort to ensure 'sorted_df' exists
if not filtered_df.empty:
    # Simplified sorting - replace with full logic from previous answer
    sort_col = selected_sort_method
    ascending_order = True
    if selected_sort_method in ['rating_votes_power_score', 'weighted_score', 'average_rating', 'ratings_count', 'likedPercent', 'pub_year_newest', 'num_pages_desc', 'price_desc']:
        ascending_order = False

    if selected_sort_method == 'pub_year_newest': sort_col = 'publication_year'
    elif selected_sort_method == 'pub_year_oldest': sort_col = 'publication_year'
    elif selected_sort_method == 'num_pages_asc': sort_col = 'num_pages'
    elif selected_sort_method == 'num_pages_desc': sort_col = 'num_pages'
    elif selected_sort_method == 'price_asc': sort_col = 'price'
    elif selected_sort_method == 'price_desc': sort_col = 'price'
        
    sorted_df = filtered_df.sort_values(by=sort_col, ascending=ascending_order, na_position='last' if ascending_order else 'first')
else:
    sorted_df = filtered_df # empty dataframe

if not sorted_df.empty and 'title' in sorted_df.columns:
    sorted_df['Audible Link URL'] = sorted_df['title'].apply(lambda t: f"https://www.audible.in/search?keywords={urllib.parse.quote_plus(str(t))}" if pd.notna(t) and t != 'Unknown' else None)
elif 'Audible Link URL' not in sorted_df.columns:
     sorted_df['Audible Link URL'] = pd.Series(dtype='str')

# --- Star Rating Function ---
def get_star_rating(rating_val):
    if pd.isna(rating_val): return "N/A"
    try:
        rating_val = float(rating_val)
        full_stars = int(rating_val)
        half_star = "½" if rating_val - full_stars >= 0.5 else ""
        empty_stars = 5 - full_stars - (1 if half_star else 0)
        return f"{'★' * full_stars}{half_star}{'☆' * empty_stars} ({rating_val:.2f})"
    except: return "N/A"


# --- Display Functions for Each Mode ---
def display_enhanced_table(df_to_display):
    st.markdown("##### Enhanced Table View")
    display_cols_table = ['coverImg', 'title', 'authors', 'series', 'average_rating', 'ratings_count', 'likedPercent', 'genres_display', 'bookFormat', 'num_pages', 'display_publication_date', 'Audible Link URL']
    if 'price' in df_to_display.columns and df_to_display['price'].sum() > 0 : display_cols_table.insert(-1, 'price')
    
    final_display_cols_table = [col for col in display_cols_table if col in df_to_display.columns]
    
    column_configs_table = {
        "coverImg": st.column_config.ImageColumn("Cover", width="small"), "title": st.column_config.TextColumn("Title", width="medium"),
        "authors": st.column_config.TextColumn("Author(s)"), "series": st.column_config.TextColumn("Series", width="small"),
        "average_rating": st.column_config.NumberColumn("Avg Rating", format="%.2f"),
        "ratings_count": st.column_config.NumberColumn("Ratings #", format="%d"),
        "likedPercent": st.column_config.NumberColumn("Liked", format="%d%%"),
        "genres_display": st.column_config.TextColumn("Genres",width="small"), "bookFormat": st.column_config.TextColumn("Format",width="small"),
        "num_pages": st.column_config.NumberColumn("Pages"), "display_publication_date": st.column_config.TextColumn("Published"),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Audible Link URL": st.column_config.LinkColumn("Audible", display_text="🎧", width="small")
    }
    active_configs_table = {k: v for k, v in column_configs_table.items() if k in final_display_cols_table}
    st.dataframe(df_to_display[final_display_cols_table], use_container_width=True, hide_index=True, column_config=active_configs_table, height=600)

def display_image_grid(df_to_display, columns_per_row=3):
    st.markdown("##### Image Grid View")
    if df_to_display.empty:
        st.info("No books to display in the grid.")
        return

    cols = st.columns(columns_per_row)
    for i, row in enumerate(df_to_display.itertuples()):
        col_index = i % columns_per_row
        with cols[col_index]:
            # Use markdown to structure the card with custom CSS
            card_html = f"""
            <div class="book-card">
                <div class="book-card-image-container">
                    <img src="{row.coverImg if pd.notna(row.coverImg) and row.coverImg else 'https://via.placeholder.com/150x220.png?text=No+Cover'}" alt="{row.title}">
                </div>
                <div class="card-title">{row.title}</div>
                <div class="card-author">{row.authors}</div>
                <div class="card-info"><strong>Rating:</strong> {get_star_rating(row.average_rating)}</div>
                <div class="card-info"><strong>Genres:</strong> {row.genres_display}</div>
                <div class="card-actions">
            """
            # Audible Link as a button-like link
            if hasattr(row, 'Audible_Link_URL') and pd.notna(row.Audible_Link_URL):
                card_html += f'<a href="{row.Audible_Link_URL}" target="_blank" class="stButton" style="text-decoration:none; display:inline-block; margin-top:5px;"><button>Listen on Audible 🎧</button></a>'
            
            card_html += "</div></div>"
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("---") # Visual separator between cards in a column


def display_card_view_with_popover(df_to_display, columns_per_row=3):
    st.markdown("##### Card View with Details")
    if df_to_display.empty:
        st.info("No books to display as cards.")
        return
        
    cols = st.columns(columns_per_row)
    for i, row in enumerate(df_to_display.itertuples()):
        col_index = i % columns_per_row
        with cols[col_index]:
            # Main card content (visible)
            card_html_visible = f"""
            <div class="book-card" style="margin-bottom:0;"> <div class="book-card-image-container">
                    <img src="{row.coverImg if pd.notna(row.coverImg) and row.coverImg else 'https://via.placeholder.com/150x220.png?text=No+Cover'}" alt="{row.title}">
                </div>
                <div class="card-title">{row.title}</div>
                <div class="card-author" style="margin-bottom:10px;">{row.authors}</div>
            """
            st.markdown(card_html_visible, unsafe_allow_html=True)

            # Popover for details
            with st.popover("View Details", use_container_width=True):
                pop_html = f"""
                <h4>{row.title}</h4>
                <p><strong>Author(s):</strong> {row.authors}</p>
                <p><strong>Series:</strong> {row.series if hasattr(row, 'series') else 'N/A'}</p>
                <p><strong>Rating:</strong> {get_star_rating(row.average_rating)} ({row.ratings_count:,} ratings)</p>
                <p><strong>Liked:</strong> {row.likedPercent:.0f}%</p>
                <p><strong>Genres:</strong> {getattr(row, 'genres_display', 'N/A')}</p>
                <p><strong>Format:</strong> {row.bookFormat}</p>
                <p><strong>Pages:</strong> {row.num_pages}</p>
                <p><strong>Published:</strong> {row.display_publication_date}</p>
                <p><strong>Publisher:</strong> {row.publisher}</p>
                """
                if 'price' in df.columns and hasattr(row, 'price') and pd.notna(row.price) and row.price > 0:
                    pop_html += f"<p><strong>Price:</strong> ${row.price:.2f}</p>"
                
                st.markdown(pop_html, unsafe_allow_html=True)
                if hasattr(row, 'Audible_Link_URL') and pd.notna(row.Audible_Link_URL):
                    st.link_button("Listen on Audible 🎧", row.Audible_Link_URL, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True) # Close the book-card div opened in card_html_visible
            st.markdown("---") # Visual separator

# --- Main Display Logic ---
st.subheader(f"Found {len(sorted_df)} / {len(df_original)} Books")
if sorted_df.empty:
    st.info("No books match your current filter scrolls. Try adjusting them, brave explorer!")
else:
    if display_mode == "Enhanced Table":
        display_enhanced_table(sorted_df)
    elif display_mode == "Image Grid":
        display_image_grid(sorted_df, columns_per_row=st.slider("Grid Columns", 2, 5, 3, key="grid_cols_slider")) # Allow user to change grid columns
    elif display_mode == "Card View (Details on Click)":
        display_card_view_with_popover(sorted_df, columns_per_row=st.slider("Card Columns", 2, 4, 3, key="card_cols_slider"))


st.sidebar.markdown("---")
st.sidebar.markdown("Crafted with 魔法 by the Book Portal Archmage!")