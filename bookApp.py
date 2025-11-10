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
DEFAULT_MIN_RATING_THRESHOLD = 3.5 # Slightly lowered for broader initial results
DEFAULT_LIKED_PERCENT_THRESHOLD = 75
DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES = 100
DEFAULT_VOTES_POWER = 1.0

# Column name mapping from new CSV to internal consistent names
COLUMN_NAME_MAP = {
    'bookId': 'book_id_str',
    'author': 'authors',
    'rating': 'average_rating',
    'numRatings': 'ratings_count',
    'pages': 'num_pages',
    'language': 'language_code',
    'publishDate': 'publication_date_edition', # Specific edition's publication
    'firstPublishDate': 'first_publication_date', # Original work's publication
    # Columns to keep as is: title, series, publisher, genres, bookFormat, likedPercent, coverImg, price
}

# Define required original columns from the CSV
# These are needed for the app to function with the new dataset
REQUIRED_ORIGINAL_COLS = [
    'title', 'author', 'rating', 'numRatings', 'pages', 'language',
    'publisher', 'genres', 'coverImg' # 'publishDate' or 'firstPublishDate' also important for sorting
]


@st.cache_data
def load_and_clean_data(file_path):
    try:
        # Use index_col=0 if the first column in CSV is just an unnamed index
        df = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

        # Check for required original columns before renaming
        for col in REQUIRED_ORIGINAL_COLS:
            if col not in df.columns:
                st.error(f"Error: Essential source column '{col}' not found in the CSV. Please check the dataset structure.")
                return pd.DataFrame()

        # Rename columns for consistency with previous logic where applicable
        df.rename(columns=COLUMN_NAME_MAP, inplace=True)

        # --- Data Type Conversions and Cleaning ---
        # Numeric columns
        df['ratings_count'] = pd.to_numeric(df.get('ratings_count'), errors='coerce').fillna(0).astype(int)
        df['num_pages'] = pd.to_numeric(df.get('num_pages'), errors='coerce').fillna(0).astype(int)
        df['average_rating'] = pd.to_numeric(df.get('average_rating'), errors='coerce') # Keep as float
        df['likedPercent'] = pd.to_numeric(df.get('likedPercent'), errors='coerce').fillna(0)
        df['price'] = pd.to_numeric(df.get('price'), errors='coerce').fillna(0) # Assuming price is numeric

        # String columns
        for col in ['title', 'authors', 'publisher', 'series', 'bookFormat', 'language_code', 'book_id_str']:
            df[col] = df.get(col, pd.Series(index=df.index, dtype='str')).fillna('Unknown')
        
        df['coverImg'] = df.get('coverImg', pd.Series(index=df.index, dtype='str')).fillna('')

        # Date columns - attempt to parse both, prioritize first_publication_date
        df['first_publication_date_dt'] = pd.to_datetime(df.get('first_publication_date'), errors='coerce', format='%m/%d/%y', infer_datetime_format=False)
        df['publication_date_edition_dt'] = pd.to_datetime(df.get('publication_date_edition'), errors='coerce', format='%m/%d/%y', infer_datetime_format=False)

        # Create a single 'publication_year' column for filtering, prioritizing first publication
        df['publication_year'] = df['first_publication_date_dt'].dt.year.fillna(df['publication_date_edition_dt'].dt.year).fillna(0).astype(int)
        
        # For display, keep original date strings if needed, or format from datetime
        df['display_publication_date'] = df['first_publication_date_dt'].dt.strftime('%Y-%m-%d').fillna(
                                           df['publication_date_edition_dt'].dt.strftime('%Y-%m-%d')).fillna('Unknown')


        # Genre parsing (string representation of list)
        def parse_genres(genre_str):
            if isinstance(genre_str, str) and genre_str.startswith('[') and genre_str.endswith(']'):
                try:
                    return ast.literal_eval(genre_str)
                except (ValueError, SyntaxError):
                    return [] # Or handle malformed strings appropriately
            elif isinstance(genre_str, list): # Already a list
                return genre_str
            return []
        df['genres_list'] = df['genres'].apply(parse_genres)
        df['genres_display'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else 'N/A')


        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An critical error occurred during data loading and cleaning: {e}")
        return pd.DataFrame()

# --- Page Setup and Styling ---
st.set_page_config(layout="wide", page_title="✨ Wizard Book Explorer ✨")

st.markdown("""
<style>
    body {
        color: #E0E0E0; /* Lighter text for dark mode feel */
        background-color: #1E1E1E; /* Dark background */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stDataFrame { /* Target Streamlit dataframe */
        background-color: #2A2A2A;
    }
    .stDataFrame img { /* Ensure images in dataframe cells are not too large */
        max-height: 80px;
        object-fit: contain;
    }
    .st-emotion-cache-1cypcdb { /* Sidebar */
        background-color: #2A2A2A; /* Darker sidebar */
        padding: 20px;
        border-right: 1px solid #444;
    }
    .st-emotion-cache-1jm9le { /* Main content area */
        padding: 25px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00A0B0; /* Teal accent for headers */
        margin-bottom: 0.5em;
    }
    h1 { color: #FF6B6B; } /* Main title a bit different */

    hr { border-top: 1px solid #444; }

    /* Input widgets */
    .stTextInput input, .stSlider div[data-baseweb="slider"], .stMultiSelect div[data-baseweb="select"] > div {
        border-radius: 5px;
        border: 1px solid #555;
        background-color: #333;
        color: #E0E0E0;
    }
    .stMultiSelect div[data-baseweb="select"] > div { padding: 5px; }
    .stSlider span[role="slider"] { background-color: #00A0B0 !important; }


    /* Expander styling */
    .st-emotion-cache-1h9usn1 p { /* Expander header text */
        color: #00A0B0;
        font-weight: bold;
    }
    
    .stButton>button { /* Button styling */
        border: 2px solid #00A0B0;
        background-color: transparent;
        color: #00A0B0;
        padding: 8px 16px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00A0B0;
        color: #1E1E1E;
    }
    .st-emotion-cache-zq5wz9, .st-emotion-cache-10y5m8g {
         margin-bottom: 0.5rem; padding-bottom: 0.5rem;
    }
    a { color: #66D9EF; text-decoration: none; } /* Links */
    a:hover { text-decoration: underline; color: #A6E22E; }
</style>
""", unsafe_allow_html=True)

st.title("✨ Wizard Book Explorer ✨")
st.write("""
Dive into a magical realm of books! Filter by genre, publication year, ratings, and more.
Discover your next great read with powerful sorting options and cover previews.
""")

df_original = load_and_clean_data(DATA_PATH)

if df_original.empty:
    st.warning("The Book Tome is empty or could not be summoned. Please check the data source and error messages.")
    st.stop()

df = df_original.copy()

# Calculate Overall Average Rating (C)
C = df['average_rating'].mean() if pd.notna(df['average_rating'].mean()) else 0.0

# --- Sidebar ---
st.sidebar.header("📜 Filters & Scrolls 📜")
st.sidebar.markdown("---")

# --- Search and Sort (Top of Sidebar) ---
search_query = st.sidebar.text_input("🔍 Search Titles, Authors, Publishers", help="Case-insensitive search.").lower()
st.sidebar.markdown("---")

st.sidebar.subheader("🪄 Sort Your Findings")
sort_options_map = {
    'Custom Score (Rating * Votes^p) [Recommended]': 'rating_votes_power_score',
    'Weighted Score (IMDb Style)': 'weighted_score',
    'Average Rating (High to Low)': 'average_rating',
    'Ratings Count (High to Low)': 'ratings_count',
    'Liked Percentage (High to Low)': 'likedPercent',
    'Publication Year (Newest First)': 'pub_year_newest',
    'Publication Year (Oldest First)': 'pub_year_oldest',
    'Page Count (Shortest First)': 'num_pages_asc',
    'Page Count (Longest First)': 'num_pages_desc',
    'Price (Low to High)': 'price_asc',
    'Price (High to Low)': 'price_desc',
}
default_sort_key = 'Custom Score (Rating * Votes^p) [Recommended]'
sort_by_display_name = st.sidebar.selectbox(
    "Sort by",
    options=list(sort_options_map.keys()),
    index=list(sort_options_map.keys()).index(default_sort_key)
)
selected_sort_method = sort_options_map[sort_by_display_name]

with st.sidebar.expander("🔧 Sorting Parameters (Advanced)", expanded=False):
    st.markdown("Adjust parameters for specific sorting methods.")
    st.subheader("Parameter 'p' for Custom Score")
    p = st.slider(
        "Votes Power ('p')", 0.0, 2.0, DEFAULT_VOTES_POWER, 0.05,
        help="Rating * Votes^p. p=0: score=rating. p=1: score=rating*votes."
    )
    st.subheader("Parameter 'm' for Weighted Score")
    st.write(f"Overall Avg Rating (C) ≈ {C:.2f}")
    m_val = st.slider(
        "Weighted Score Anchor Votes ('m')", 0, 1000, DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES, 10,
        help="'m' virtual ratings at value C. Higher 'm' pulls scores to C."
    )

st.sidebar.markdown("---")

# --- Filter Groups ---
with st.sidebar.expander("🌟 Primary Quality & Engagement", expanded=True):
    min_votes_threshold = st.slider(
        "Minimum Ratings Count", 0, int(df['ratings_count'].max() if not df.empty else 1000),
        DEFAULT_MIN_VOTES_THRESHOLD, 10
    )
    min_rating_threshold = st.slider(
        "Minimum Average Rating", 0.0, 5.0, DEFAULT_MIN_RATING_THRESHOLD, 0.1
    )
    min_liked_percent = st.slider(
        "Minimum Liked Percent", 0, 100, DEFAULT_LIKED_PERCENT_THRESHOLD, 1,
        help="Minimum percentage of readers who liked the book."
    )

with st.sidebar.expander("📚 Content Attributes"):
    # Genre Filter
    all_genres_flat = sorted(list(set(g for sublist in df['genres_list'] for g in sublist if g)))
    selected_genres = st.multiselect(
        "Filter by Genres (ANY selected)",
        options=all_genres_flat,
        default=[],
        help="Show books that have ANY of the selected genres."
    )
    # Language Filter
    unique_languages = sorted(df['language_code'].dropna().unique().tolist())
    # Remove 'Unknown' if it's the only one or provide 'All'
    if 'Unknown' in unique_languages and len(unique_languages) == 1 and not all_genres_flat : # if unknown is only lang and no genres selected
        pass # Don't show language filter if not diverse
    elif unique_languages:
        all_languages_option = ['All'] + [lang for lang in unique_languages if lang != 'Unknown']
        selected_languages_multiselect = st.multiselect(
            "Language", options=all_languages_option, default=['All']
        )

    # Book Format Filter
    unique_formats = sorted(df['bookFormat'].dropna().unique().tolist())
    if unique_formats:
         selected_formats = st.multiselect(
            "Book Format", options=[fmt for fmt in unique_formats if fmt != 'Unknown'], default=[]
        )


with st.sidebar.expander("📖 Publication & Length"):
    min_year = int(df['publication_year'][df['publication_year'] > 0].min() if not df[df['publication_year'] > 0].empty else 1800)
    max_year = int(df['publication_year'].max() if not df.empty else datetime.now().year)
    selected_year_range = st.slider(
        "Publication Year Range (First Pub.)",
        min_year, max_year, (min_year, max_year)
    )
    max_pg = int(df['num_pages'].max() if not df.empty else 1000)
    min_pg, max_pg_selected = st.slider(
        "Page Count Range", 0, max_pg, (0, max_pg), 10
    )

if 'price' in df.columns and df['price'].nunique() > 1 : # only show if price data is meaningful
    with st.sidebar.expander("💰 Price (if available)"):
        min_price = float(df['price'][df['price'] > 0].min() if not df[df['price'] > 0].empty else 0)
        max_price = float(df['price'].max() if not df.empty else 100)
        # Ensure min_price is less than max_price for slider
        if min_price >= max_price and max_price > 0: max_price = min_price + 1 # simple adjustment
        elif min_price >=max_price and max_price == 0 : max_price = 100 # default if no price data

        if max_price > min_price: # Only show slider if range is valid
            selected_price_range = st.slider(
                "Price Range",
                min_price, max_price, (min_price, max_price), step=max(0.1, (max_price-min_price)/100) # Dynamic step
            )
        else:
            st.info("Price data not sufficient for range filter.")


# --- Score Calculations ---
df['weighted_score'] = np.nan
valid_ratings_mask_ws = df['average_rating'].notna()
if m_val == 0:
    df.loc[valid_ratings_mask_ws, 'weighted_score'] = df.loc[valid_ratings_mask_ws, 'average_rating']
else:
    df.loc[valid_ratings_mask_ws, 'weighted_score'] = (
        (df.loc[valid_ratings_mask_ws, 'ratings_count'] / (df.loc[valid_ratings_mask_ws, 'ratings_count'] + m_val)) * df.loc[valid_ratings_mask_ws, 'average_rating'] +
        (m_val / (df.loc[valid_ratings_mask_ws, 'ratings_count'] + m_val)) * C
    )

df['rating_votes_power_score'] = 0.0
valid_ratings_mask_rp = df['average_rating'].notna()
if p == 0:
    df.loc[valid_ratings_mask_rp, 'rating_votes_power_score'] = df.loc[valid_ratings_mask_rp, 'average_rating'].fillna(0)
else:
    has_rating_pos_votes = valid_ratings_mask_rp & (df['ratings_count'] > 0)
    df.loc[has_rating_pos_votes, 'rating_votes_power_score'] = (
        df.loc[has_rating_pos_votes, 'average_rating'] *
        (df.loc[has_rating_pos_votes, 'ratings_count'] ** p)
    )
df['rating_votes_power_score'] = df['rating_votes_power_score'].fillna(0)


# --- Apply Filters ---
current_mask = pd.Series([True] * len(df), index=df.index)

if search_query:
    search_cols = ['title', 'authors', 'publisher', 'series'] # Include series in search
    current_mask &= df[search_cols].apply(lambda row: row.astype(str).str.lower().str.contains(search_query, regex=False).any(), axis=1)

current_mask &= (df['ratings_count'] >= min_votes_threshold)
current_mask &= (df['average_rating'].fillna(0) >= min_rating_threshold) # handle NaN ratings for filter
current_mask &= (df['likedPercent'].fillna(0) >= min_liked_percent)

if selected_genres:
    current_mask &= df['genres_list'].apply(lambda x_genres: any(sg in x_genres for sg in selected_genres))

if 'selected_languages_multiselect' in locals() and 'All' not in selected_languages_multiselect and selected_languages_multiselect:
    current_mask &= df['language_code'].isin(selected_languages_multiselect)

if 'selected_formats' in locals() and selected_formats:
    current_mask &= df['bookFormat'].isin(selected_formats)

current_mask &= (df['publication_year'] >= selected_year_range[0]) & (df['publication_year'] <= selected_year_range[1])
current_mask &= (df['num_pages'] >= min_pg) & (df['num_pages'] <= max_pg_selected)

if 'selected_price_range' in locals() and 'price' in df.columns:
     current_mask &= (df['price'] >= selected_price_range[0]) & (df['price'] <= selected_price_range[1])


filtered_df = df[current_mask].copy()

# --- Apply Sorting ---
sorted_df = pd.DataFrame(columns=filtered_df.columns) # Init empty
if not filtered_df.empty:
    sort_ascending = True
    na_pos = 'first'
    
    if selected_sort_method in ['rating_votes_power_score', 'weighted_score', 'average_rating', 'ratings_count', 'likedPercent', 'pub_year_newest', 'num_pages_desc', 'price_desc']:
        sort_ascending = False
        na_pos = 'last'

    if selected_sort_method == 'pub_year_newest':
        sorted_df = filtered_df.sort_values(by='publication_year', ascending=False, na_position='last')
    elif selected_sort_method == 'pub_year_oldest':
        sorted_df = filtered_df.sort_values(by='publication_year', ascending=True, na_position='first')
    elif selected_sort_method == 'price_asc':
         sorted_df = filtered_df.sort_values(by='price', ascending=True, na_position='first')
    elif selected_sort_method == 'price_desc':
         sorted_df = filtered_df.sort_values(by='price', ascending=False, na_position='last')
    elif selected_sort_method in ['num_pages_asc', 'num_pages_desc']:
        col_to_sort = 'num_pages'
        sorted_df = filtered_df.sort_values(by=col_to_sort, ascending=sort_ascending, na_position=na_pos)
    elif selected_sort_method in sort_options_map.values():
        # Handles 'rating_votes_power_score', 'weighted_score', 'average_rating', 'ratings_count', 'likedPercent'
        sorted_df = filtered_df.sort_values(by=selected_sort_method, ascending=sort_ascending, na_position=na_pos)
    else:
        sorted_df = filtered_df # Fallback


# --- Audible Link ---
def create_audible_link_url(title_str):
    if pd.isna(title_str) or title_str == '' or title_str == 'Unknown':
        return None
    base = "https://www.audible.in/search?"
    params = {'keywords': str(title_str), 'k': str(title_str)}
    return f"{base}{urllib.parse.urlencode(params)}"

if not sorted_df.empty and 'title' in sorted_df.columns:
    sorted_df['Audible Link URL'] = sorted_df['title'].apply(create_audible_link_url)
elif 'Audible Link URL' not in sorted_df.columns : # Ensure column exists for empty df
     sorted_df['Audible Link URL'] = pd.Series(dtype='str')


# --- Display Results ---
st.subheader(f"✨ Found {len(sorted_df)} / {len(df_original)} Books ✨")

display_cols = [
    'coverImg', 'title', 'authors', 'series', 'average_rating', 'ratings_count', 'likedPercent',
    'genres_display', 'bookFormat', 'num_pages', 'display_publication_date', 'publisher'
]
if 'price' in sorted_df.columns and sorted_df['price'].sum() > 0: # show price if data exists
    display_cols.append('price')


# Add score columns if used for sorting
if selected_sort_method == 'rating_votes_power_score':
    display_cols.insert(display_cols.index('average_rating'), 'rating_votes_power_score')
elif selected_sort_method == 'weighted_score':
    display_cols.insert(display_cols.index('average_rating'), 'weighted_score')

# Add Audible link
if 'Audible Link URL' in sorted_df.columns:
    display_cols.append('Audible Link URL')

# Filter display_cols to only those that exist in sorted_df
final_display_cols = [col for col in display_cols if col in sorted_df.columns]


column_configs = {
    "coverImg": st.column_config.ImageColumn("Cover", help="Book Cover Image", width="small"),
    "title": st.column_config.TextColumn("Title", help="Book Title", width="medium"),
    "authors": st.column_config.TextColumn("Author(s)", width="medium"),
    "series": st.column_config.TextColumn("Series", width="small"),
    "average_rating": st.column_config.NumberColumn("Avg Rating", format="%.2f", help="Average user rating"),
    "rating_votes_power_score": st.column_config.NumberColumn("Custom Score", format="%.2f"),
    "weighted_score": st.column_config.NumberColumn("Weighted Score", format="%.2f"),
    "ratings_count": st.column_config.NumberColumn("Ratings #", format="%d", help="Total ratings count"),
    "likedPercent": st.column_config.NumberColumn("Liked %", format="%d%%", help="Percentage of readers who liked it"),
    "genres_display": st.column_config.TextColumn("Genres", width="medium"),
    "bookFormat": st.column_config.TextColumn("Format", width="small"),
    "num_pages": st.column_config.NumberColumn("Pages", format="%d"),
    "display_publication_date": st.column_config.TextColumn("Published", help="Primarily First Publication Date"),
    "publisher": st.column_config.TextColumn("Publisher", width="medium"),
    "price": st.column_config.NumberColumn("Price", format="$%.2f", help="Price if available"), # Adjust format as needed
    "Audible Link URL": st.column_config.LinkColumn("Audible", display_text="Search 🎧", help="Search on Audible.in")
}

active_column_configs = {k: v for k, v in column_configs.items() if k in final_display_cols}

if not sorted_df.empty:
    st.dataframe(
        sorted_df[final_display_cols],
        use_container_width=True,
        hide_index=True,
        column_config=active_column_configs,
        height=600 # Set a fixed height for the dataframe viewport
    )
else:
    st.info("No books match your current filter scrolls. Try adjusting them!")

st.sidebar.markdown("---")
st.sidebar.info("Crafted with ✨ by the Book Wizard!")