import streamlit as st
import pandas as pd
import re
import urllib.parse
import numpy as np # Import numpy for mean calculation and handling NaNs

# --- Configuration ---
DATA_PATH = 'audiobooks.csv' # Make sure your data is in a CSV file named audiobooks.csv
DEFAULT_MIN_VOTES_THRESHOLD = 10 # Default filter threshold for votes
DEFAULT_MIN_RATING_THRESHOLD = 4.0 # Default filter threshold for rating
DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES = 30 # 'm' parameter for weighted score calculation

# --- Helper Functions ---

@st.cache_data # Cache the data loading and cleaning
def load_and_clean_data(file_path):
    """Loads data from CSV and cleans 'stars' and 'time' columns."""
    try:
        df = pd.read_csv(file_path)

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # --- Clean 'stars' column to get 'rating' and 'votes' ---
        def parse_stars(stars_str):
            if pd.isna(stars_str) or stars_str == 'Not rated yet':
                return None, 0 # Use None for rating if not rated
            rating_match = re.search(r'(\d+(\.\d+)?)\s+out of 5 stars', str(stars_str))
            votes_match = re.search(r'(\d+)\s+ratings', str(stars_str))

            rating = float(rating_match.group(1)) if rating_match else None
            votes = int(votes_match.group(1)) if votes_match else 0
            return rating, votes

        # Apply the parsing function
        parsed_data = df['stars'].apply(parse_stars)
        df['rating'] = parsed_data.apply(lambda x: x[0])
        df['votes'] = parsed_data.apply(lambda x: x[1])

        # Convert votes to integer, handling potential NaNs from original data
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)


        # --- Clean 'time' column to get 'total_minutes' ---
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
                            elif 'min' in unit: # handle cases like '30 min'
                                total_minutes += value
                                i += 2
                            else: # Handle cases like '10' without 'hrs' or 'mins' after
                                total_minutes += value # Assume minutes if no unit follows? Or hours? Let's assume minutes for simplicity unless 'hr' is present.
                                i += 1
                        else: # Just a number at the end? Assume minutes
                             total_minutes += value
                             i += 1
                    else:
                        i += 1 # Skip non-digit/unit parts
            except Exception as e:
                # st.warning(f"Could not parse time string '{time_str}': {e}") # Avoid warnings during caching
                return 0 # Return 0 on error


            return total_minutes

        df['total_minutes'] = df['time'].apply(parse_time)

        # Convert price to numeric, handle potential errors and commas
        df['price'] = df['price'].astype(str).str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Ensure rating is numeric, coercing errors. Keep None for not rated.
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An error occurred during data loading and cleaning: {e}")
        return pd.DataFrame()

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Audible Audiobook Explorer")

st.title("🎧 Audible Audiobook Explorer")

st.write("""
Explore the audiobook dataset with custom filters and a ranking system
based on a weighted score considering both star ratings and the number of votes.
""")

# Load data
df = load_and_clean_data(DATA_PATH)

if df.empty:
    st.stop() # Stop the app if data loading failed

# Calculate the overall mean rating for the weighted score calculation
# Calculate only on non-null ratings
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
    max_value=200, # Allow a reasonable range for 'm'
    value=DEFAULT_WEIGHTED_SCORE_ANCHOR_VOTES,
    step=1,
    help="Items with 'm' votes will have their weighted score be roughly the average of their rating and the overall average rating (C)."
)


# Calculate the weighted score for all items
# Formula: ((v * R) + (m * C)) / (v + m)
# Handle cases where rating is NaN - these will result in NaN weighted score
# Use C for items with 0 votes where rating is not NaN, otherwise NaN
df['weighted_score'] = np.nan # Initialize with NaN
# Apply the formula where rating is not NaN
valid_ratings_mask = df['rating'].notna()
df.loc[valid_ratings_mask, 'weighted_score'] = (
    (df.loc[valid_ratings_mask, 'votes'] * df.loc[valid_ratings_mask, 'rating']) + (m * C)
) / (df.loc[valid_ratings_mask, 'votes'] + m)

# For items with 0 votes but a valid rating, the formula simplifies, or we can set it explicitly if m=0 or votes=0
# If votes is 0, score is C (if m > 0). If m is 0, score is R (if votes > 0).
# The current formula handles votes=0 correctly if m>0 -> ((0*R) + m*C) / (0+m) = m*C / m = C
# The current formula handles m=0 correctly if votes>0 -> ((v*R) + 0*C) / (v+0) = v*R / v = R
# If votes=0 and m=0, it's division by zero, but filtered out by min_votes_threshold usually.
# Let's explicitly set score to C for items with 0 votes but valid rating, provided m > 0
if m > 0:
    zero_votes_mask = (df['votes'] == 0) & valid_ratings_mask
    df.loc[zero_votes_mask, 'weighted_score'] = C


st.sidebar.markdown("---")
st.sidebar.subheader("Minimum Engagement & Quality Filters")
st.sidebar.write("Only titles meeting *both* thresholds after other filters will be shown.")
min_votes_threshold = st.sidebar.slider(
    "Minimum Number of Votes (Filter)",
    min_value=0,
    max_value=int(df['votes'].max()),
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
st.sidebar.subheader("Other Filters")

# Price Filter
if df['price'].min() is not None and df['price'].max() is not None and not df['price'].isnull().all():
    min_price_val = float(df['price'].min())
    max_price_val = float(df['price'].max())
    min_price, max_price = st.sidebar.slider(
        "Price Range",
        min_value=min_price_val,
        max_value=max_price_val,
        value=(min_price_val, max_price_val),
        step=10.0
    )
else:
    min_price, max_price = 0.0, 1000.0 # Default range if price data is missing or all NaN
    st.sidebar.warning("Price data not available or clean.")


# Time Filter (in hours for user friendly input)
max_time_hours = int(df['total_minutes'].max() / 60) + 1 if df['total_minutes'].max() > 0 else 1 # Approx max hours
min_time_hours, max_time_hours_selected = st.sidebar.slider(
    "Time Range (Hours)",
    min_value=0,
    max_value=max_time_hours,
    value=(0, max_time_hours),
    step=1
)
min_time_minutes_filter = min_time_hours * 60
max_time_minutes_filter = max_time_hours_selected * 60


# Language Filter
all_languages = ['All'] + sorted(df['language'].dropna().unique().tolist()) # Add 'All' option
selected_languages = st.sidebar.multiselect(
    "Language",
    options=all_languages,
    default=['All'] # Select 'All' by default
)
# Handle 'All' selection
if 'All' in selected_languages:
    languages_to_filter = df['language'].dropna().unique().tolist()
elif selected_languages: # If 'All' not selected but some languages are
     languages_to_filter = selected_languages
else: # If 'All' not selected and no languages are picked (empty list)
     languages_to_filter = [] # This will filter out everything


# Search Box
search_query = st.sidebar.text_input("Search (Title, Author, Narrator)").lower()

st.sidebar.markdown("---")
st.sidebar.subheader("Sorting")
sort_by = st.sidebar.selectbox(
    "Sort by",
    options=['Weighted Score (Recommended)', 'Rating (High to Low)', 'Votes (High to Low)', 'Price (Low to High)', 'Price (High to Low)', 'Time (Shortest First)', 'Time (Longest First)']
)
# Default ascending based on sort_by, but weighted_score/rating/votes usually descending
sort_ascending = False
if 'Price' in sort_by or 'Time (Shortest First)' in sort_by:
    sort_ascending = True


# --- Apply Filters ---

# Start with the original DataFrame and apply filters sequentially or using combined masks

# Initial mask based on numerical ranges and language
initial_mask = (
    (df['price'] >= min_price) &
    (df['price'] <= max_price) &
    (df['total_minutes'] >= min_time_minutes_filter) &
    (df['total_minutes'] <= max_time_minutes_filter)
)

# Apply language filter to the mask
if languages_to_filter:
    initial_mask = initial_mask & (df['language'].isin(languages_to_filter))
else: # If no languages selected and 'All' wasn't default, and list is empty
     initial_mask = initial_mask & (df['language'].isna()) # or False, ensures nothing passes if no language selected


# Apply search filter to the mask
if search_query:
    search_mask = (
        df['name'].astype(str).str.lower().str.contains(search_query) |
        df['author'].astype(str).str.lower().str.contains(search_query) |
        df['narrator'].astype(str).str.lower().str.contains(search_query)
    )
    initial_mask = initial_mask & search_mask


# Apply all initial filters
filtered_df = df[initial_mask].copy()

# Now apply the minimum engagement/quality filter thresholds to the already filtered data
# Use fillna(0) for safety when comparing rating, although NaNs should have been handled by weighted score calc
filtered_df = filtered_df[
    (filtered_df['rating'].fillna(0) >= min_rating_threshold) &
    (filtered_df['votes'] >= min_votes_threshold)
].copy() # Final copy after all filters


# --- Apply Sorting ---

if sort_by == 'Weighted Score (Recommended)':
    # Sort by weighted score, handling NaNs (put NaNs at the end)
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


# --- Generate Audible Link Column ---

def create_audible_link(title):
    if pd.isna(title):
        return "N/A"
    base_url = "https://www.audible.in/search?"
    # Encode the title for URL parameters
    encoded_title = urllib.parse.quote_plus(str(title))
    # Construct the URL matching the example structure
    # Using both 'keywords' and 'k' as seen in the example
    full_url = f"{base_url}keywords={encoded_title}&k={encoded_title}&i=eu-audible-in"
    # Return as a markdown link
    return f"[Audible Link]({full_url})"

sorted_df['Audible Link'] = sorted_df['name'].apply(create_audible_link)


# --- Display Results ---

st.write(f"Showing {len(sorted_df)} out of {len(df)} audiobooks")

# Select columns to display - include weighted score
display_columns = [
    'name', 'author', 'narrator', 'time', 'releasedate',
    'language', 'stars', 'rating', 'votes', 'weighted_score', 'price', 'Audible Link'
]

# Display the data table
# Format weighted_score and price for better readability
display_df = sorted_df[display_columns].copy()
# Apply formatting only if the column exists and is numeric
if 'weighted_score' in display_df.columns:
    display_df['weighted_score'] = display_df['weighted_score'].map('{:.2f}'.format, na_action='ignore')
if 'price' in display_df.columns:
     display_df['price'] = display_df['price'].map('{:.2f}'.format, na_action='ignore')


st.dataframe(display_df, use_container_width=True, hide_index=True)