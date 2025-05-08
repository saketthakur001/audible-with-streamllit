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
    """Loads data from CSV and cleans 'stars' and 'time' columns."""
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

# --- Add some basic styling ---
st.markdown("""
<style>
    /* Basic body styling */
    body {
        color: #333;
        background-color: #f0f2f6;
        font-family: sans-serif;
    }
    /* Sidebar styling */
    .st-emotion-cache-1cypcdb { /* Target the sidebar container */
        background-color: #ffffff;
        padding: 20px;
        border-right: 1px solid #ddd;
    }
     /* Main content area padding */
    .st-emotion-cache-1jm9le { /* Target the main content container */
        padding: 20px;
    }
     /* Style the header */
    h1 {
        color: #ff4b4b; /* Streamlit's primary color */
        margin-bottom: 10px;
    }
    h2, h3, h4, h5, h6 {
        color: #333;
        margin-top: 15px;
        margin-bottom: 8px;
    }
    /* Style markdown horizontal rule */
    hr {
        border-top: 1px solid #bbb;
    }
    /* Adjust dataframe header background */
    .ag-header {
        background-color: #e9e9e9 !important;
        color: #333 !important;
        font-weight: bold;
    }
    /* Adjust dataframe row hover */
    .ag-row:hover {
        background-color: #f0f0f0 !important;
    }
     /* Style the clickable link text */
    .ag-cell-value a {
        color: #007bff; /* Link color */
        text-decoration: none; /* Remove underline */
        font-weight: normal;
    }
     .ag-cell-value a:hover {
        text-decoration: underline; /* Add underline on hover */
    }


</style>
""", unsafe_allow_html=True)


st.title("🎧 Audible Audiobook Explorer")

st.write("""
Explore the audiobook dataset with custom filters and a ranking system
based on a weighted score considering both star ratings and the number of votes.
""")

# Load data
df = load_and_clean_data(DATA_PATH)

if df.empty:
    st.stop()

# Calculate the overall mean rating for the weighted score calculation
C = df['rating'].mean()

# --- Sidebar Filters ---
st.sidebar.header("Filters and Ranking")

# Weighted Score Parameter
st.sidebar.markdown("---")
st.sidebar.subheader("Weighted Score Parameter ('m')")
st.sidebar.write("This value influences how much weight is given to the overall average rating vs. the item's own rating.")
st.sidebar.write(f"Overall Average Rating (C) ≈ {C:.2f}")

m = st.sidebar.slider(
    "Weighted Score Anchor Votes ('m')",
    min_value=0,
    max_value=200,
    value=DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES,
    step=1,
    help="Items with 'm' votes will have their weighted score be roughly the average of their rating and the overall average rating (C)."
)

# Calculate the weighted score for all items
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
st.sidebar.write("Only titles meeting *both* thresholds after other filters will be shown.")
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
        st.warning("Price data not available or clean.")


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
        return None # Return None if title is missing
    base_url = "https://www.audible.in/search?"
    encoded_title = urllib.parse.quote_plus(str(title))
    full_url = f"{base_url}keywords={encoded_title}&k={encoded_title}&i=eu-audible-in"
    return full_url # Return ONLY the URL string

sorted_df['Audible Link URL'] = sorted_df['name'].apply(create_audible_link_url)


# --- Display Results ---

st.write(f"Showing {len(sorted_df)} out of {len(df)} audiobooks")

# Select columns to display, using the new 'Audible Link URL' column
display_columns = [
    'name', 'author', 'narrator', 'time', 'releasedate',
    'language', 'rating', 'votes', 'weighted_score', 'price', 'Audible Link URL' # Use the URL column
]

# Display the data table using column_config for the link and formatting
st.dataframe(
    sorted_df[display_columns], # Pass the DataFrame with the URL column
    use_container_width=True,
    hide_index=True,
    column_config={
        "Audible Link URL": st.column_config.LinkColumn(
            "Audible Link", # Header text for the column
            display_text="Search on Audible 🔎", # Text to display in the cell
            help="Click to search for this title on Audible.in"
        ),
        # Configure other columns for better display formatting
        "weighted_score": st.column_config.NumberColumn(
             "Weighted Score", format="%.2f" # Format as two decimal places
         ),
         "price": st.column_config.NumberColumn(
             "Price", format="%.2f" # Format price
         ),
         "rating": st.column_config.NumberColumn(
             "Rating", format="%.2f" # Format rating
         ),
         "votes": st.column_config.NumberColumn(
              "Votes", format="%d" # Format as integer
         ),
         "total_minutes": st.column_config.NumberColumn(
              "Total Minutes", format="%d", # Not displayed by default, but good config
              help="Total duration in minutes"
         ),
         # Configure text columns to prevent potential wrapping issues or add tooltips
         "name": st.column_config.TextColumn("Name", help="Book Title"),
         "author": st.column_config.TextColumn("Author"),
         "narrator": st.column_config.TextColumn("Narrator"),
         "time": st.column_config.TextColumn("Time"), # Original time string
         "releasedate": st.column_config.TextColumn("Release Date"),
         "language": st.column_config.TextColumn("Language"),
    }
)