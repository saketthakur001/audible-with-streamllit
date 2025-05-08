import streamlit as st
import pandas as pd
import re
import urllib.parse
import numpy as np

# --- Configuration ---
DATA_PATH = 'books.csv' # Use the new file path
DEFAULT_MIN_VOTES_THRESHOLD = 50 # Adjusted default threshold for book ratings_count
DEFAULT_MIN_RATING_THRESHOLD = 4.0 # Default filter threshold for average_rating
DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES = 100 # 'm' parameter for weighted score calculation (adjusted for potentially higher vote counts)

# --- Helper Functions ---

@st.cache_data
def load_and_clean_data(file_path):
    """Loads data from CSV and cleans relevant columns."""
    try:
        df = pd.read_csv(file_path)

        # Clean column names (remove leading/trailing spaces and potential hidden characters)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

        # Ensure relevant columns exist and are in correct format
        required_cols = ['title', 'authors', 'average_rating', 'ratings_count', 'num_pages', 'language_code', 'publication_date', 'publisher']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Error: Required column '{col}' not found in the CSV file.")
                return pd.DataFrame() # Return empty DataFrame if a required column is missing

        # Convert ratings_count and num_pages to integer, handle errors
        df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce').fillna(0).astype(int)
        df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce').fillna(0).astype(int)

        # Convert average_rating to numeric, handle errors
        df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')

        # Handle potential NaNs in language_code
        df['language_code'] = df['language_code'].fillna('Unknown')

        # Handle potential NaNs in other text columns used for search
        df['title'] = df['title'].fillna('')
        df['authors'] = df['authors'].fillna('')
        df['publisher'] = df['publisher'].fillna('')


        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading and cleaning: {e}")
        return pd.DataFrame()

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Book Explorer")

st.markdown("""
<style>
    body {
        color: #333;
        background-color: #f0f2f6;
        font-family: sans-serif;
    }
    .st-emotion-cache-1cypcdb {
        background-color: #ffffff;
        padding: 20px;
        border-right: 1px solid #ddd;
    }
    .st-emotion-cache-1jm9le {
        padding: 20px;
    }
    h1 {
        color: #ff4b4b;
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
    /* Adjust spacing for smaller header */
    .st-emotion-cache-zq5wz9 { /* Target header container */
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
     .st-emotion-cache-10y5m8g { /* Target markdown p tag */
         margin-top: 0.5rem;
         margin-bottom: 1rem;
    }


</style>
""", unsafe_allow_html=True)

# Using st.header for a smaller title
st.header("📚 Book Explorer")

st.write("""
Explore the book dataset with custom filters and ranking based on average rating and number of ratings.
""")

df = load_and_clean_data(DATA_PATH)

if df.empty:
    st.stop()

# Calculate the overall mean rating for the weighted score calculation
# Calculate only on non-null ratings
C = df['average_rating'].mean()

st.sidebar.header("Filters and Ranking")

st.sidebar.markdown("---")
st.sidebar.subheader("Minimum Engagement & Quality Filters")
st.sidebar.write("Titles must meet *both* thresholds.")
min_votes_threshold = st.sidebar.slider(
    "Minimum Number of Ratings (Filter)",
    min_value=0,
    max_value=int(df['ratings_count'].max() if not df['ratings_count'].empty else 0),
    value=DEFAULT_MIN_VOTES_THRESHOLD,
    step=1
)
min_rating_threshold = st.sidebar.slider(
    "Minimum Average Rating (Filter)",
    min_value=0.0,
    max_value=5.0,
    value=DEFAULT_MIN_RATING_THRESHOLD,
    step=0.1
)

st.sidebar.markdown("---")
# Use expander for 'Other Filters'
with st.sidebar.expander("Other Filters"):

    # Page Count Filter (instead of Time)
    max_pages = int(df['num_pages'].max() if not df['num_pages'].empty else 1)
    min_pages, max_pages_selected = st.slider(
        "Page Count Range",
        min_value=0,
        max_value=max_pages,
        value=(0, max_pages),
        step=10
    )

    # Language Code Filter (instead of Language)
    all_languages = ['All'] + sorted(df['language_code'].dropna().unique().tolist())
    selected_languages = st.multiselect(
        "Language Code",
        options=all_languages,
        default=['All']
    )
    if 'All' in selected_languages:
        languages_to_filter = df['language_code'].dropna().unique().tolist()
    elif selected_languages:
         languages_to_filter = selected_languages
    else:
         languages_to_filter = []

    # Price Filter (Not available in books.csv, removing)
    # if df['price'].min() is not None ...

st.sidebar.markdown("---")
# Search Box - Keep outside expander for easy access
search_query = st.sidebar.text_input("Search (Title, Authors, Publisher)").lower()


st.sidebar.markdown("---")
st.sidebar.subheader("Sorting")
sort_options = [
    'Average Rating & Ratings Count (Recommended)',
    'Average Rating (High to Low)',
    'Ratings Count (High to Low)',
    'Page Count (Shortest First)',
    'Page Count (Longest First)',
    'Publication Date (Newest First)',
    'Publication Date (Oldest First)',
    'Weighted Score (Optional)' # Keep weighted score as an option
]
sort_by = st.sidebar.selectbox(
    "Sort by",
    options=sort_options
)

# --- Calculate Weighted Score (Optional, for sorting) ---
# Formula: ((v * R) + (m * C)) / (v + m)
# v = ratings_count, R = average_rating, m = Weighted Score Anchor Votes, C = Overall Average Rating
st.sidebar.markdown("---")
with st.sidebar.expander("Weighted Score Parameter"):
    st.write("Parameter for 'Weighted Score (Optional)' sort.")
    st.write(f"Overall Average Rating (C) ≈ {C:.2f}")
    m = st.slider(
        "Weighted Score Anchor Votes ('m')",
        min_value=0,
        max_value=500, # Adjusted range for books
        value=DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES,
        step=1,
        help="'m' votes means weighted score is avg of item rating and C."
    )

df['weighted_score'] = np.nan
valid_ratings_mask = df['average_rating'].notna()
df.loc[valid_ratings_mask, 'weighted_score'] = (
    (df.loc[valid_ratings_mask, 'ratings_count'] * df.loc[valid_ratings_mask, 'average_rating']) + (m * C)
) / (df.loc[valid_ratings_mask, 'ratings_count'] + m)

if m > 0:
    zero_votes_mask = (df['ratings_count'] == 0) & valid_ratings_mask
    df.loc[zero_votes_mask, 'weighted_score'] = C


# --- Apply Filters ---

# Initial mask based on numerical ranges and language
initial_mask = (
    (df['num_pages'] >= min_pages) &
    (df['num_pages'] <= max_pages_selected)
)

# Apply language filter to the mask
if languages_to_filter:
    initial_mask = initial_mask & (df['language_code'].isin(languages_to_filter))
else:
     initial_mask = initial_mask & (df['language_code'].isna()) # or False

# Apply search filter to the mask
if search_query:
    search_mask = (
        df['title'].astype(str).str.lower().str.contains(search_query) |
        df['authors'].astype(str).str.lower().str.contains(search_query) |
        df['publisher'].astype(str).str.lower().str.contains(search_query)
    )
    initial_mask = initial_mask & search_mask

# Apply all initial filters
filtered_df = df[initial_mask].copy()

# Now apply the minimum engagement/quality filter thresholds to the already filtered data
filtered_df = filtered_df[
    (filtered_df['average_rating'].fillna(0) >= min_rating_threshold) &
    (filtered_df['ratings_count'] >= min_votes_threshold)
].copy()


# --- Apply Sorting ---

if sort_by == 'Average Rating & Ratings Count (Recommended)':
    # Sort primarily by average_rating, secondarily by ratings_count
    sorted_df = filtered_df.sort_values(by=['average_rating', 'ratings_count'], ascending=[False, False], na_position='last')
elif sort_by == 'Average Rating (High to Low)':
     sorted_df = filtered_df.sort_values(by='average_rating', ascending=False, na_position='last')
elif sort_by == 'Ratings Count (High to Low)':
     sorted_df = filtered_df.sort_values(by='ratings_count', ascending=False)
elif sort_by == 'Page Count (Shortest First)':
    sorted_df = filtered_df.sort_values(by='num_pages', ascending=True)
elif sort_by == 'Page Count (Longest First)':
    sorted_df = filtered_df.sort_values(by='num_pages', ascending=False)
elif sort_by == 'Publication Date (Newest First)':
    # Convert to datetime for proper sorting, handle errors
    sorted_df['publication_date_dt'] = pd.to_datetime(sorted_df['publication_date'], errors='coerce')
    sorted_df = sorted_df.sort_values(by='publication_date_dt', ascending=False, na_position='last').drop(columns='publication_date_dt')
elif sort_by == 'Publication Date (Oldest First)':
    sorted_df['publication_date_dt'] = pd.to_datetime(sorted_df['publication_date'], errors='coerce')
    sorted_df = sorted_df.sort_values(by='publication_date_dt', ascending=True, na_position='first').drop(columns='publication_date_dt')
elif sort_by == 'Weighted Score (Optional)':
    sorted_df = filtered_df.sort_values(by='weighted_score', ascending=False, na_position='last')


# --- Generate Audible Link Column (Returning URL string) ---
def create_audible_link_url(title):
    if pd.isna(title) or title == '':
        return None
    base_url = "https://www.audible.in/search?"
    encoded_title = urllib.parse.quote_plus(str(title))
    full_url = f"{base_url}keywords={encoded_title}&k={encoded_title}&i=eu-audible-in"
    return full_url

sorted_df['Audible Link URL'] = sorted_df['title'].apply(create_audible_link_url)


# --- Display Results ---

st.write(f"Showing {len(sorted_df)} out of {len(df)} books")

# Select columns to display - adjusted for books.csv
display_columns = [
    'bookID', 'title', 'authors', 'average_rating', 'ratings_count',
    'num_pages', 'language_code', 'publication_date', 'publisher', 'Audible Link URL'
]

# Display the data table using column_config for the link and formatting
st.dataframe(
    sorted_df[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Audible Link URL": st.column_config.LinkColumn(
            "Audible Link",
            display_text="Search on Audible 🔎",
            help="Click to search for this title on Audible.in (May not find book editions)"
        ),
        "average_rating": st.column_config.NumberColumn(
             "Average Rating", format="%.2f"
         ),
         "ratings_count": st.column_config.NumberColumn(
              "Ratings Count", format="%d"
         ),
         "num_pages": st.column_config.NumberColumn(
              "Number of Pages", format="%d"
         ),
         # Add configs for other columns for better display
         "bookID": st.column_config.NumberColumn("Book ID"),
         "title": st.column_config.TextColumn("Title"),
         "authors": st.column_config.TextColumn("Authors"),
         "language_code": st.column_config.TextColumn("Language Code"),
         "publication_date": st.column_config.TextColumn("Publication Date"),
         "publisher": st.column_config.TextColumn("Publisher"),
         # Weighted score column is not displayed by default, but could be added here if needed
         # "weighted_score": st.column_config.NumberColumn("Weighted Score", format="%.2f"),
    }
)
