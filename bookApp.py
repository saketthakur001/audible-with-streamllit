import streamlit as st
import pandas as pd
import re
import urllib.parse
import numpy as np

DATA_PATH = 'books.csv'
DEFAULT_MIN_VOTES_THRESHOLD = 50
DEFAULT_MIN_RATING_THRESHOLD = 4.0 # User can adjust; high default for "quality"
DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES = 100
DEFAULT_VOTES_POWER = 1.0  # Changed default to 1.0 to give more weight to votes initially

@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        # Clean column names (strip whitespace, remove BOM if present)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

        required_cols = ['title', 'authors', 'average_rating', 'ratings_count', 'num_pages', 'language_code', 'publication_date', 'publisher']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Error: Required column '{col}' not found in the CSV.")
                return pd.DataFrame()

        # Data type conversions and cleaning
        df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce').fillna(0).astype(int)
        df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce').fillna(0).astype(int)
        df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce') # Keep as float, NaNs handled in scoring

        df['language_code'] = df['language_code'].fillna('Unknown')
        df['title'] = df['title'].fillna('Unknown Title')
        df['authors'] = df['authors'].fillna('Unknown Author')
        df['publisher'] = df['publisher'].fillna('Unknown Publisher')
        # Ensure publication_date is string for consistent processing later
        df['publication_date'] = df['publication_date'].astype(str).fillna('Unknown Date')


        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure '{file_path}' is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading and cleaning: {e}")
        return pd.DataFrame()

st.set_page_config(layout="wide", page_title="📚 Advanced Book Explorer")

# --- Custom CSS (mostly as provided by user) ---
st.markdown("""
<style>
    body {
        color: #333;
        background-color: #f0f2f6;
        font-family: sans-serif;
    }
    .st-emotion-cache-1cypcdb { /* Sidebar */
        background-color: #ffffff;
        padding: 20px;
        border-right: 1px solid #ddd;
    }
    .st-emotion-cache-1jm9le { /* Main content area */
        padding: 20px;
    }
    h1 {
        color: #ff4b4b; /* A vibrant red for the main title */
        margin-bottom: 10px;
    }
    h2, h3, h4, h5, h6 {
        color: #333;
        margin-top: 15px;
        margin-bottom: 8px;
    }
    hr {
        border-top: 1px solid #bbb;
    }
    /* Styling for st.dataframe headers (Note: st.dataframe has limited CSS control) */
    .stDataFrame .col_heading {
        background-color: #e9e9e9 !important;
        color: #333 !important;
        font-weight: bold;
    }
    /* These .ag-* classes are for ag-grid, not directly for st.dataframe */
    .ag-header {
        background-color: #e9e9e9 !important;
        color: #333 !important;
        font-weight: bold;
    }
    .ag-row:hover {
        background-color: #f0f0f0 !important;
    }
    .ag-cell-value a {
        color: #007bff;
        text-decoration: none;
        font-weight: normal;
    }
    .ag-cell-value a:hover {
        text-decoration: underline;
    }
    .st-emotion-cache-zq5wz9 { /* Specific Streamlit component margin adjustment */
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
     .st-emotion-cache-10y5m8g { /* Specific Streamlit component margin adjustment */
         margin-top: 0.5rem;
         margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.header("📚 Advanced Book Explorer")
st.write("""
Explore a comprehensive book dataset. Filter by various criteria and use advanced sorting options
to find exactly what you're looking for. The recommended sort now prioritizes a combination of
high ratings and a substantial number of votes.
""")

df_original = load_and_clean_data(DATA_PATH)

if df_original.empty:
    st.warning("Data could not be loaded. Please check the data source and error messages above.")
    st.stop()

# Make a copy for filtering and manipulation to preserve the original dataframe
df = df_original.copy()

# Calculate Overall Average Rating (C) - used for Weighted Score
# Ensure C is calculated only from valid ratings and is a scalar
C = df['average_rating'].mean() if pd.notna(df['average_rating'].mean()) else 0.0

# --- Sidebar for Filters and Ranking ---
st.sidebar.header("Filters and Ranking")
st.sidebar.markdown("---")

st.sidebar.subheader("Minimum Engagement & Quality")
st.sidebar.write("Books must meet *both* thresholds to be displayed.")
min_votes_threshold = st.sidebar.slider(
    "Minimum Number of Ratings",
    min_value=0,
    max_value=int(df['ratings_count'].max() if not df['ratings_count'].empty else 1000), # Handle empty df
    value=DEFAULT_MIN_VOTES_THRESHOLD,
    step=10,
    help="Filters out books with fewer than this many ratings."
)
min_rating_threshold = st.sidebar.slider(
    "Minimum Average Rating",
    min_value=0.0,
    max_value=5.0,
    value=DEFAULT_MIN_RATING_THRESHOLD,
    step=0.1,
    help="Filters out books with an average rating below this value."
)

st.sidebar.markdown("---")
with st.sidebar.expander("Additional Filters"):
    max_pages_val = int(df['num_pages'].max() if not df['num_pages'].empty else 1000)
    min_pages, max_pages_selected = st.slider(
        "Page Count Range",
        min_value=0,
        max_value=max_pages_val,
        value=(0, max_pages_val),
        step=10
    )

    unique_languages = sorted(df['language_code'].dropna().unique().tolist())
    all_languages_option = ['All'] + unique_languages
    selected_languages_multiselect = st.multiselect(
        "Language Code",
        options=all_languages_option,
        default=['All'],
        help="Select languages. If 'All' is chosen, language filter is ignored. If empty, no books will match."
    )

st.sidebar.markdown("---")
search_query = st.sidebar.text_input("Search (Title, Authors, Publisher)", help="Case-insensitive search.").lower()

st.sidebar.markdown("---")
st.sidebar.subheader("Sorting Options")

# --- Define Sort Options ---
# Naming convention changed for clarity
sort_options_map = {
    'Custom Score (Rating * Votes^p) [Recommended]': 'rating_votes_power_score',
    'Weighted Score (IMDb Style)': 'weighted_score',
    'Average Rating (High to Low)': 'average_rating',
    'Ratings Count (High to Low)': 'ratings_count',
    'Average Rating then Ratings Count': 'rating_then_votes',
    'Page Count (Shortest First)': 'num_pages_asc',
    'Page Count (Longest First)': 'num_pages_desc',
    'Publication Date (Newest First)': 'pub_date_newest',
    'Publication Date (Oldest First)': 'pub_date_oldest',
}
# Make the new recommended the default
default_sort_key = 'Custom Score (Rating * Votes^p) [Recommended]'

sort_by_display_name = st.sidebar.selectbox(
    "Sort by",
    options=list(sort_options_map.keys()),
    index=list(sort_options_map.keys()).index(default_sort_key) # Set default dynamically
)
selected_sort_method = sort_options_map[sort_by_display_name]


# --- Sorting Parameter Sliders ---
st.sidebar.markdown("---")
with st.sidebar.expander("Sorting Parameters (Advanced)", expanded=False):
    st.markdown("Adjust parameters for specific sorting methods.")

    st.subheader("Parameter 'p' for Custom Score")
    st.write("Applies to: 'Custom Score (Rating * Votes^p)'")
    p = st.slider(
        "Votes Power ('p')",
        min_value=0.0,
        max_value=2.0,
        value=DEFAULT_VOTES_POWER,
        step=0.05,
        help="Controls vote influence in 'Rating * Votes^p'.\np=0: score is rating.\np=1: score is rating * votes.\np>1: votes have amplified effect."
    )
    st.markdown("---")
    st.subheader("Parameter 'm' for Weighted Score")
    st.write("Applies to: 'Weighted Score (IMDb Style)'")
    st.write(f"Overall Average Rating of all books (C) ≈ {C:.2f}")
    m = st.slider(
        "Weighted Score Anchor Votes ('m')",
        min_value=0,
        max_value=1000, # Increased max for more flexibility
        value=DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES,
        step=10,
        help="'m' is the number of 'virtual' ratings at value C added to each book's score. Higher 'm' pulls scores closer to the overall average C, especially for books with few actual ratings."
    )

# --- Calculate Scores ---

# 1. Calculate Weighted Score (IMDb style)
# R = average for the movie (mean)
# v = number of votes for the movie
# m = minimum votes required to be listed (now a configurable parameter)
# C = the mean vote across the whole report
# weighted rating (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C
df['weighted_score'] = np.nan # Initialize column
valid_ratings_mask_ws = df['average_rating'].notna() # Mask for rows with valid average_rating

# Calculate for rows with valid ratings
# Handle m=0 case to avoid division by zero if ratings_count is also 0
if m == 0:
    df.loc[valid_ratings_mask_ws, 'weighted_score'] = df.loc[valid_ratings_mask_ws, 'average_rating']
else:
    df.loc[valid_ratings_mask_ws, 'weighted_score'] = (
        (df.loc[valid_ratings_mask_ws, 'ratings_count'] / (df.loc[valid_ratings_mask_ws, 'ratings_count'] + m)) * df.loc[valid_ratings_mask_ws, 'average_rating'] +
        (m / (df.loc[valid_ratings_mask_ws, 'ratings_count'] + m)) * C
    )
    # For items with 0 votes and m > 0, their score should effectively be C.
    # The formula above might give NaN if ratings_count is 0 because average_rating could be NaN.
    # Explicitly set score to C for 0-vote items if they had a rating initially (or handle if it was NaN)
    # However, if average_rating is NaN, their weighted_score should remain NaN or be C if we impute.
    # The current calculation should yield C if average_rating is valid and ratings_count is 0.
    # If average_rating itself is NaN, weighted_score will be NaN, which is fine.

# 2. Calculate Custom Score: Rating * Votes Power Score
df['rating_votes_power_score'] = 0.0  # Initialize to 0.0 for all

valid_ratings_mask_rp = df['average_rating'].notna()

if p == 0:
    # If p is 0, score is simply the average_rating (or 0 if rating is NaN)
    df.loc[valid_ratings_mask_rp, 'rating_votes_power_score'] = df.loc[valid_ratings_mask_rp, 'average_rating']
else: # p > 0
    # For books with ratings_count > 0 and a valid average_rating
    has_rating_and_positive_votes_mask = valid_ratings_mask_rp & (df['ratings_count'] > 0)
    df.loc[has_rating_and_positive_votes_mask, 'rating_votes_power_score'] = (
        df.loc[has_rating_and_positive_votes_mask, 'average_rating'] *
        (df.loc[has_rating_and_positive_votes_mask, 'ratings_count'] ** p)
    )
    # For books with ratings_count == 0 (and p > 0) and a valid average_rating,
    # score is average_rating * (0^p) = 0. This is covered by initialization.
    # Books with no average_rating (NaN) will also have a score of 0.0.

# --- Apply Filters ---
# Start with a mask that includes all books
current_mask = pd.Series([True] * len(df), index=df.index)

# Page count filter
current_mask &= (df['num_pages'] >= min_pages) & (df['num_pages'] <= max_pages_selected)

# Language filter
if not selected_languages_multiselect: # User cleared all selections
    current_mask &= pd.Series([False] * len(df), index=df.index) # Match no books
elif 'All' not in selected_languages_multiselect: # Specific languages selected
    current_mask &= df['language_code'].isin(selected_languages_multiselect)
# If 'All' is in selected_languages_multiselect, no language filter is applied (mask remains unchanged for this step)

# Search query filter (applies to title, authors, publisher)
if search_query:
    search_mask = (
        df['title'].astype(str).str.lower().str.contains(search_query, regex=False) |
        df['authors'].astype(str).str.lower().str.contains(search_query, regex=False) |
        df['publisher'].astype(str).str.lower().str.contains(search_query, regex=False)
    )
    current_mask &= search_mask

# Apply initial filters
filtered_df = df[current_mask].copy() # .copy() to avoid SettingWithCopyWarning

# Apply minimum engagement & quality filters
filtered_df = filtered_df[
    (filtered_df['average_rating'].fillna(0) >= min_rating_threshold) &
    (filtered_df['ratings_count'] >= min_votes_threshold)
].copy()


# --- Apply Sorting ---
# Default to an empty dataframe if no sort method matches (should not happen with selectbox)
sorted_df = pd.DataFrame(columns=filtered_df.columns)

if not filtered_df.empty:
    if selected_sort_method == 'rating_then_votes':
        sorted_df = filtered_df.sort_values(by=['average_rating', 'ratings_count'], ascending=[False, False], na_position='last')
    elif selected_sort_method == 'rating_votes_power_score':
        sorted_df = filtered_df.sort_values(by='rating_votes_power_score', ascending=False, na_position='last')
    elif selected_sort_method == 'average_rating':
        sorted_df = filtered_df.sort_values(by='average_rating', ascending=False, na_position='last')
    elif selected_sort_method == 'ratings_count':
        sorted_df = filtered_df.sort_values(by='ratings_count', ascending=False, na_position='last')
    elif selected_sort_method == 'num_pages_asc':
        sorted_df = filtered_df.sort_values(by='num_pages', ascending=True, na_position='last')
    elif selected_sort_method == 'num_pages_desc':
        sorted_df = filtered_df.sort_values(by='num_pages', ascending=False, na_position='last')
    elif selected_sort_method == 'pub_date_newest':
        filtered_df['publication_date_dt'] = pd.to_datetime(filtered_df['publication_date'], errors='coerce')
        sorted_df = filtered_df.sort_values(by='publication_date_dt', ascending=False, na_position='last').drop(columns='publication_date_dt')
    elif selected_sort_method == 'pub_date_oldest':
        filtered_df['publication_date_dt'] = pd.to_datetime(filtered_df['publication_date'], errors='coerce')
        sorted_df = filtered_df.sort_values(by='publication_date_dt', ascending=True, na_position='first').drop(columns='publication_date_dt')
    elif selected_sort_method == 'weighted_score':
        sorted_df = filtered_df.sort_values(by='weighted_score', ascending=False, na_position='last')
    else: # Fallback, though selectbox should prevent this
        sorted_df = filtered_df

# --- Generate Audible Link Column ---
def create_audible_link_url(title):
    if pd.isna(title) or title == '' or title == 'Unknown Title':
        return None
    base_url = "https://www.audible.in/search?"
    # More robust encoding for search query parameters
    params = {'keywords': str(title), 'k': str(title), 'i': 'eu-audible-in'}
    full_url = f"{base_url}{urllib.parse.urlencode(params)}"
    return full_url

if not sorted_df.empty:
    sorted_df['Audible Link URL'] = sorted_df['title'].apply(create_audible_link_url)
else:
    # Ensure the column exists even if sorted_df is empty to avoid errors in st.dataframe
    sorted_df['Audible Link URL'] = pd.Series(dtype='str')


# --- Display Results ---
st.write(f"#### Displaying {len(sorted_df)} of {len(df_original)} books based on your criteria.")

# Define columns to display, ensuring they exist
base_display_columns = [
    'title', 'authors', 'average_rating', 'ratings_count',
    'num_pages', 'language_code', 'publication_date', 'publisher'
]
# Ensure only existing columns are selected from sorted_df
display_columns = [col for col in base_display_columns if col in sorted_df.columns]


# Dynamically add score columns for display if they were used for sorting or are relevant
if selected_sort_method == 'rating_votes_power_score' and 'rating_votes_power_score' in sorted_df.columns:
    display_columns.insert(2, 'rating_votes_power_score') # Insert after authors
elif selected_sort_method == 'weighted_score' and 'weighted_score' in sorted_df.columns:
    display_columns.insert(2, 'weighted_score') # Insert after authors

# Add Audible link at the end if the column exists
if 'Audible Link URL' in sorted_df.columns:
    display_columns.append('Audible Link URL')


# Column configurations for st.dataframe
column_configs = {
    "title": st.column_config.TextColumn("Title", help="Book Title", width="medium"),
    "authors": st.column_config.TextColumn("Authors", help="Book Author(s)", width="medium"),
    "average_rating": st.column_config.NumberColumn("Avg Rating", help="Average user rating (0-5)", format="%.2f"),
    "ratings_count": st.column_config.NumberColumn("Ratings Count", help="Total number of ratings", format="%d"),
    "num_pages": st.column_config.NumberColumn("# Pages", help="Number of pages", format="%d"),
    "language_code": st.column_config.TextColumn("Language"),
    "publication_date": st.column_config.TextColumn("Pub Date"),
    "publisher": st.column_config.TextColumn("Publisher", width="medium"),
    "weighted_score": st.column_config.NumberColumn("Weighted Score", help="IMDb-style weighted score", format="%.2f"),
    "rating_votes_power_score": st.column_config.NumberColumn("Custom Score", help="Rating * Votes^p Score", format="%.2f"),
    "Audible Link URL": st.column_config.LinkColumn(
        "Audible",
        display_text="Search 🔎",
        help="Search this title on Audible.in (results may vary)"
    )
}

# Filter column_configs to only include keys present in display_columns
active_column_configs = {k: v for k, v in column_configs.items() if k in display_columns}

if not sorted_df.empty:
    st.dataframe(
        sorted_df[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config=active_column_configs
    )
else:
    st.info("No books match your current filter criteria.")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Adjust the 'Votes Power (p)' under 'Sorting Parameters' to fine-tune how much the number of votes influences the 'Custom Score'.")