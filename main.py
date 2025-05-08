import streamlit as st
import pandas as pd
import re
import urllib.parse
import numpy as np

# --- Configuration ---
DATA_PATH = 'audiobooks.csv'
DEFAULT_MIN_VOTES_THRESHOLD = 10
DEFAULT_MIN_RATING_THRESHOLD = 4.0
DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES = 30

# --- Helper Functions ---

@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        def parse_stars(stars_str):
            if pd.isna(stars_str) or str(stars_str).strip() == 'Not rated yet':
                return None, 0
            rating_match = re.search(r'(\d+(\.\d+)?)\s+out of 5 stars', str(stars_str))
            votes_match = re.search(r'(\d+)\s+ratings', str(stars_str))
            rating = float(rating_match.group(1)) if rating_match else None
            votes = int(votes_match.group(1)) if votes_match else 0
            return rating, votes

        parsed_data = df['stars'].apply(parse_stars)
        df['rating'] = parsed_data.apply(lambda x: x[0])
        df['votes'] = parsed_data.apply(lambda x: x[1])
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)

        def parse_time(time_str):
            if pd.isna(time_str):
                return 0
            parts = str(time_str).replace(' and ', ' ').replace('mins', '').split()
            total_minutes = 0
            try:
                i = 0
                while i < len(parts):
                    if parts[i].isdigit():
                        value = int(parts[i])
                        if i + 1 < len(parts):
                            unit = parts[i+1].lower().strip()
                            if 'hr' in unit or 'hour' in unit:
                                total_minutes += value * 60
                                i += 2
                            elif 'min' in unit:
                                total_minutes += value
                                i += 2
                            else:
                                total_minutes += value
                                i += 1
                        else:
                             total_minutes += value
                             i += 1
                    else:
                        i += 1
            except Exception as e:
                return 0
            return total_minutes

        df['total_minutes'] = df['time'].apply(parse_time)
        df['price'] = df['price'].astype(str).str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading and cleaning: {e}")
        return pd.DataFrame()

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Audible Audiobook Explorer")

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

# Changed st.title to st.header for smaller size
st.header("🎧 Audible Audiobook Explorer")

st.write("""
Explore the audiobook dataset with custom filters and a ranking system
based on a weighted score considering both star ratings and the number of votes.
""")

df = load_and_clean_data(DATA_PATH)

if df.empty:
    st.stop()

C = df['rating'].mean()

st.sidebar.header("Filters and Ranking")

st.sidebar.markdown("---")
st.sidebar.subheader("Weighted Score Parameter ('m')")
st.sidebar.write("Influences weight of overall average vs. item rating.")
st.sidebar.write(f"Overall Average Rating (C) ≈ {C:.2f}")

m = st.sidebar.slider(
    "Weighted Score Anchor Votes ('m')",
    min_value=0,
    max_value=200,
    value=DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES,
    step=1,
    help="'m' votes means weighted score is avg of item rating and C."
)

df['weighted_score'] = np.nan
valid_ratings_mask = df['rating'].notna()
df.loc[valid_ratings_mask, 'weighted_score'] = (
    (df.loc[valid_ratings_mask, 'votes'] * df.loc[valid_ratings_mask, 'rating']) + (m * C)
) / (df.loc[valid_ratings_mask, 'votes'] + m)

if m > 0:
    zero_votes_mask = (df['votes'] == 0) & valid_ratings_mask
    df.loc[zero_votes_mask, 'weighted_score'] = C

st.sidebar.markdown("---")
st.sidebar.subheader("Minimum Engagement & Quality Filters")
st.sidebar.write("Titles must meet *both* thresholds.")
min_votes_threshold = st.sidebar.slider(
    "Minimum Number of Votes (Filter)",
    min_value=0,
    max_value=int(df['votes'].max() if not df['votes'].empty else 0),
    value=DEFAULT_MIN_VOTES_THRESHOLD,
    step=1
)
min_rating_threshold = st.sidebar.slider(
    "Minimum Star Rating (Filter)",
    min_value=0.0,
    max_value=5.0,
    value=DEFAULT_MIN_RATING_THRESHOLD,
    step=0.1
)

st.sidebar.markdown("---")
with st.sidebar.expander("Other Filters"):
    if df['price'].min() is not None and df['price'].max() is not None and not df['price'].isnull().all():
        min_price_val = float(df['price'].min())
        max_price_val = float(df['price'].max())
        min_price, max_price = st.slider(
            "Price Range",
            min_value=min_price_val,
            max_value=max_price_val,
            value=(min_price_val, max_price_val),
            step=10.0
        )
    else:
        min_price, max_price = 0.0, 1000.0
        st.warning("Price data not available.")

    max_time_hours = int(df['total_minutes'].max() / 60) + 1 if df['total_minutes'].max() > 0 else 1
    min_time_hours, max_time_hours_selected = st.slider(
        "Time Range (Hours)",
        min_value=0,
        max_value=max_time_hours,
        value=(0, max_time_hours),
        step=1
    )
    min_time_minutes_filter = min_time_hours * 60
    max_time_minutes_filter = max_time_hours_selected * 60

    all_languages = ['All'] + sorted(df['language'].dropna().unique().tolist())
    selected_languages = st.multiselect(
        "Language",
        options=all_languages,
        default=['All']
    )
    if 'All' in selected_languages:
        languages_to_filter = df['language'].dropna().unique().tolist()
    elif selected_languages:
         languages_to_filter = selected_languages
    else:
         languages_to_filter = []

st.sidebar.markdown("---")
search_query = st.sidebar.text_input("Search (Title, Author, Narrator)").lower()

st.sidebar.markdown("---")
st.sidebar.subheader("Sorting")
sort_by = st.sidebar.selectbox(
    "Sort by",
    options=['Weighted Score (Recommended)', 'Rating (High to Low)', 'Votes (High to Low)', 'Price (Low to High)', 'Price (High to Low)', 'Time (Shortest First)', 'Time (Longest First)']
)

# --- Apply Filters ---
initial_mask = (
    (df['price'] >= min_price) &
    (df['price'] <= max_price) &
    (df['total_minutes'] >= min_time_minutes_filter) &
    (df['total_minutes'] <= max_time_minutes_filter)
)

if languages_to_filter:
    initial_mask = initial_mask & (df['language'].isin(languages_to_filter))
else:
     initial_mask = initial_mask & (df['language'].isna())

if search_query:
    search_mask = (
        df['name'].astype(str).str.lower().str.contains(search_query) |
        df['author'].astype(str).str.lower().str.contains(search_query) |
        df['narrator'].astype(str).str.lower().str.contains(search_query)
    )
    initial_mask = initial_mask & search_mask

filtered_df = df[initial_mask].copy()

filtered_df = filtered_df[
    (filtered_df['rating'].fillna(0) >= min_rating_threshold) &
    (filtered_df['votes'] >= min_votes_threshold)
].copy()

# --- Apply Sorting ---
if sort_by == 'Weighted Score (Recommended)':
    sorted_df = filtered_df.sort_values(by='weighted_score', ascending=False, na_position='last')
elif sort_by == 'Rating (High to Low)':
     sorted_df = filtered_df.sort_values(by='rating', ascending=False, na_position='last')
elif sort_by == 'Votes (High to Low)':
     sorted_df = filtered_df.sort_values(by='votes', ascending=False)
elif sort_by == 'Price (Low to High)':
    sorted_df = filtered_df.sort_values(by='price', ascending=True)
elif sort_by == 'Price (High to Low)':
    sorted_df = filtered_df.sort_values(by='price', ascending=False)
elif sort_by == 'Time (Shortest First)':
    sorted_df = filtered_df.sort_values(by='total_minutes', ascending=True)
elif sort_by == 'Time (Longest First)':
    sorted_df = filtered_df.sort_values(by='total_minutes', ascending=False)

# --- Generate Audible Link Column (Returning URL string) ---
def create_audible_link_url(title):
    if pd.isna(title):
        return None
    base_url = "https://www.audible.in/search?"
    encoded_title = urllib.parse.quote_plus(str(title))
    full_url = f"{base_url}keywords={encoded_title}&k={encoded_title}&i=eu-audible-in"
    return full_url

sorted_df['Audible Link URL'] = sorted_df['name'].apply(create_audible_link_url)

# --- Display Results ---
st.write(f"Showing {len(sorted_df)} out of {len(df)} audiobooks")

display_columns = [
    'name', 'author', 'narrator', 'time', 'releasedate',
    'language', 'rating', 'votes', 'weighted_score', 'price', 'Audible Link URL'
]

st.dataframe(
    sorted_df[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Audible Link URL": st.column_config.LinkColumn(
            "Audible Link",
            display_text="Search on Audible 🔎",
            help="Click to search for this title on Audible.in"
        ),
        "weighted_score": st.column_config.NumberColumn(
             "Weighted Score", format="%.2f"
         ),
         "price": st.column_config.NumberColumn(
             "Price", format="%.2f"
         ),
         "rating": st.column_config.NumberColumn(
             "Rating", format="%.2f"
         ),
         "votes": st.column_config.NumberColumn(
              "Votes", format="%d"
         ),
         "total_minutes": st.column_config.NumberColumn(
              "Total Minutes", format="%d",
              help="Total duration in minutes"
         ),
         "name": st.column_config.TextColumn("Name", help="Book Title"),
         "author": st.column_config.TextColumn("Author"),
         "narrator": st.column_config.TextColumn("Narrator"),
         "time": st.column_config.TextColumn("Time"),
         "releasedate": st.column_config.TextColumn("Release Date"),
         "language": st.column_config.TextColumn("Language"),
    }
)
