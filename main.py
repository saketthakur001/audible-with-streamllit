import os
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import urllib.parse
import kagglehub

# Get the dataset path
dataset_path = kagglehub.dataset_download("snehangsude/audible-dataset")

# Find all files in the dataset folder
files = os.listdir(dataset_path)
print("Files in the dataset:", files)

# Assuming the dataset contains a CSV file
csv_file = [file for file in files if file.endswith(".csv")][0]  # Get the first CSV file
csv_path = os.path.join(dataset_path, csv_file)

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_path)

# Add Audible search link column
base_url = "https://www.audible.in/search"
data['audible_link'] = data['title'].apply(lambda title: f"{base_url}?{urllib.parse.urlencode({'keywords': title})}")

# Streamlit UI setup
st.title("Audible Dataset Explorer")

# Filters
st.sidebar.header("Filters")
min_price, max_price = st.sidebar.slider("Price Range", 0, int(data["price"].max()), (0, int(data["price"].max())))
min_votes = st.sidebar.slider("Minimum Votes", 0, int(data["votes"].max()), 0)
min_stars = st.sidebar.slider("Minimum Stars", 0.0, 5.0, 0.0, step=0.5)

# Apply filters
filtered_data = data[
    (data["price"] >= min_price) &
    (data["price"] <= max_price) &
    (data["votes"] >= min_votes) &
    (data["stars"] >= min_stars)
]

# Sorting
sort_by = st.sidebar.selectbox("Sort By", ["stars", "votes", "price"])
sort_order = st.sidebar.radio("Sort Order", ["Ascending", "Descending"])
filtered_data = filtered_data.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))

# Display filtered data in an AgGrid table
gb = GridOptionsBuilder.from_dataframe(filtered_data)
gb.configure_pagination(paginationPageSize=10)  # Pagination
gb.configure_selection("single", use_checkbox=True)  # Single row selection with checkbox
gb.configure_column("audible_link", cellRenderer="link", cellRendererParams={"label": "Search on Audible"})  # Audible link column
grid_options = gb.build()

st.write(f"Showing {len(filtered_data)} results:")
AgGrid(filtered_data, gridOptions=grid_options, enable_enterprise_modules=True)
