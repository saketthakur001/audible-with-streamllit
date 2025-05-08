import streamlit as st
import pandas as pd
import re
import urllib.parse

# --- Configuration ---
DATA_PATH = 'audiobooks.csv' # Make sure your data is in a CSV file named audiobooks.csv
DEFAULT_MIN_VOTES_THRESHOLD = 10 # A default threshold to start with
DEFAULT_MIN_RATING_THRESHOLD = 4.0 # A default threshold to start with

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
            if stars_str == 'Not rated yet':
                return None, 0 # Use None for rating if not rated
            match = re.search(r'(\d+(\.\d+)?)\s+out of 5 stars\s+(\d+)\s+ratings', stars_str)
            if match:
                rating = float(match.group(1))
                votes = int(match.group(3))
                return rating, votes
            return None, 0 # Return None, 0 for any unhandled format

        df[['rating', 'votes']] = df['stars'].apply(parse_stars).tolist()

        # --- Clean 'time' column to get 'total_minutes' ---
        def parse_time(time_str):
            if pd.isna(time_str):
                return 0
            parts = time_str.replace(' and ', ' ').replace('mins', '').split()
            total_minutes = 0
            try:
                i = 0
                while i < len(parts):
                    if parts[i].isdigit():
                        value = int(parts[i])
                        if i + 1 < len(parts):
                            unit = parts[i+1].lower().strip()
                            if 'hr' in unit:
                                total_minutes += value * 60
                            elif 'min' in unit:
                                total_minutes += value
                            i += 2
                        else: # Just a number at the end? Unlikely format, handle defensively
                             total_minutes += value # Assume minutes if no unit follows
                             i += 1
                    else:
                        i += 1 # Skip non-digit/unit parts
            except Exception as e:
                st.warning(f"Could not parse time string '{time_str}': {e}")
                return 0 # Return 0 on error


            return total_minutes

        df['total_minutes'] = df['time'].apply(parse_time)

        # Convert price to numeric, handle potential errors
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Ensure rating and votes are numeric, coercing errors
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)


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
based on star ratings and the number of votes.
""")

# Load data
df = load_and_clean_data(DATA_PATH)

if df.empty:
    st.stop() # Stop the app if data loading failed

# --- Sidebar Filters ---
st.sidebar.header("Filters and Ranking")

# Rating and Vote Thresholds (Our Custom Ranking System)
st.sidebar.markdown("---")
st.sidebar.subheader("Minimum Engagement & Quality")
st.sidebar.write("Titles must meet *both* thresholds to be shown.")
min_votes_threshold = st.sidebar.slider(
    "Minimum Number of Votes",
    min_value=0,
    max_value=int(df['votes'].max()),
    value=DEFAULT_MIN_VOTES_THRESHOLD,
    step=1
)
min_rating_threshold = st.sidebar.slider(
    "Minimum Star Rating",
    min_value=0.0,
    max_value=5.0,
    value=DEFAULT_MIN_RATING_THRESHOLD,
    step=0.1
)


st.sidebar.markdown("---")
st.sidebar.subheader("Other Filters")

# Price Filter
min_price, max_price = st.sidebar.slider(
    "Price Range",
    min_value=float(df['price'].min()),
    max_value=float(df['price'].max()),
    value=(float(df['price'].min()), float(df['price'].max())),
    step=10.0
)

# Time Filter (in hours for user friendly input)
max_time_hours = int(df['total_minutes'].max() / 60) + 1 # Approx max hours
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
all_languages = df['language'].unique().tolist()
selected_languages = st.sidebar.multiselect(
    "Language",
    options=all_languages,
    default=all_languages # Select all by default
)

# Search Box
search_query = st.sidebar.text_input("Search (Title, Author, Narrator)").lower()

st.sidebar.markdown("---")
st.sidebar.subheader("Sorting")
sort_by = st.sidebar.selectbox(
    "Sort by",
    options=['Rating & Votes (Recommended)', 'Price (Low to High)', 'Price (High to Low)', 'Time (Shortest First)', 'Time (Longest First)', 'Votes (High to Low)', 'Rating (High to Low)']
)


# --- Apply Filters ---

# Apply rating and vote thresholds first
filtered_df = df[
    (df['rating'] >= min_rating_threshold) &
    (df['votes'] >= min_votes_threshold)
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Apply other filters
filtered_df = filtered_df[
    (filtered_df['price'] >= min_price) &
    (filtered_df['price'] <= max_price) &
    (filtered_df['total_minutes'] >= min_time_minutes_filter) &
    (filtered_df['total_minutes'] <= max_time_minutes_filter) &
    (filtered_df['language'].isin(selected_languages))
]

# Apply search filter
if search_query:
    filtered_df = filtered_df[
        filtered_df['name'].str.lower().str.contains(search_query) |
        filtered_df['author'].str.lower().str.contains(search_query) |
        filtered_df['narrator'].str.lower().str.contains(search_query)
    ]


# --- Apply Sorting ---

if sort_by == 'Rating & Votes (Recommended)':
    # Sort primarily by rating, secondarily by votes
    sorted_df = filtered_df.sort_values(by=['rating', 'votes'], ascending=[False, False])
elif sort_by == 'Price (Low to High)':
    sorted_df = filtered_df.sort_values(by='price', ascending=True)
elif sort_by == 'Price (High to Low)':
    sorted_df = filtered_df.sort_values(by='price', ascending=False)
elif sort_by == 'Time (Shortest First)':
    sorted_df = filtered_df.sort_values(by='total_minutes', ascending=True)
elif sort_by == 'Time (Longest First)':
    sorted_df = filtered_df.sort_values(by='total_minutes', ascending=False)
elif sort_by == 'Votes (High to Low)':
    sorted_df = filtered_df.sort_values(by='votes', ascending=False)
elif sort_by == 'Rating (High to Low)':
     # This is just sorting by rating, Votes & Rating is the recommended one
     sorted_df = filtered_df.sort_values(by='rating', ascending=False)


# --- Generate Audible Link Column ---

def create_audible_link(title):
    base_url = "https://www.audible.in/search?"
    # Encode the title for URL parameters
    encoded_title = urllib.parse.quote_plus(title)
    # Construct the URL matching the example structure
    # Using both 'keywords' and 'k' as seen in the example
    full_url = f"{base_url}keywords={encoded_title}&k={encoded_title}&i=eu-audible-in"
    # Return as a markdown link
    return f"[Audible Link]({full_url})"

sorted_df['Audible Link'] = sorted_df['name'].apply(create_audible_link)


# --- Display Results ---

st.write(f"Showing {len(sorted_df)} out of {len(df)} audiobooks")

# Select columns to display
display_columns = ['name', 'author', 'narrator', 'time', 'releasedate', 'language', 'rating', 'votes', 'price', 'Audible Link']

# Display the data table
st.dataframe(sorted_df[display_columns], use_container_width=True, hide_index=True)

# Optional: Display raw data (for debugging)
# st.sidebar.subheader("Raw Data (Debug)")
# if st.sidebar.checkbox("Show raw data"):
#     st.write(df)