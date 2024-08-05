import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

def calculate_centroid(cluster, lat_col, lon_col):
    latitudes = cluster[lat_col]
    longitudes = cluster[lon_col]
    centroid_lat = latitudes.mean()
    centroid_lon = longitudes.mean()
    return centroid_lat, centroid_lon

def find_lat_lon_columns(df):
    lat_col = None
    lon_col = None
    for col in df.columns:
        if col.lower() in ['latitude', 'lat']:
            lat_col = col
        elif col.lower() in ['longitude', 'lon', 'long']:
            lon_col = col
    return lat_col, lon_col

def cluster_complaints_dbscan(df, eps_km, min_samples):
    lat_col, lon_col = find_lat_lon_columns(df)
    if lat_col is None or lon_col is None:
        raise ValueError("Latitude or Longitude columns not found in the dataset.")
    
    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    df = df.copy()
    
    # Drop rows with NaN values in latitude and longitude columns
    df.dropna(subset=[lat_col, lon_col], inplace=True)
    
    # Convert eps from km to radians
    kms_per_radian = 6371.0088
    eps = eps_km / kms_per_radian
    
    # Convert lat/lon to radians
    coords = np.radians(df[[lat_col, lon_col]].values)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    
    # Use .loc to set the 'cluster' column
    df.loc[:, 'cluster'] = dbscan.fit_predict(coords)
    
    unique_clusters = df['cluster'].unique()
    centroids = []
    for cluster_id in unique_clusters:
        if cluster_id != -1:  # -1 indicates noise points
            cluster_data = df[df['cluster'] == cluster_id]
            centroid_lat, centroid_lon = calculate_centroid(cluster_data, lat_col, lon_col)
            complaint_count = len(cluster_data)
            centroids.append((centroid_lat, centroid_lon, cluster_data['Issue Number'].tolist(), complaint_count))
    return centroids

st.title('Complaint Clustering App')

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write(df)

    eps = st.number_input('Maximum distance between complaints in kilometres', min_value=0.001, max_value=10.0, value=0.5, step=0.1, format="%.3f")
    min_samples = st.number_input('Minimum number of complaints to form a cluster', min_value=1, max_value=200, value=5, step=1)
    
    if st.button('Cluster Complaints'):
        try:
            centroids = cluster_complaints_dbscan(df, eps, min_samples)
            result_df = pd.DataFrame(centroids, columns=['centroid_latitude', 'centroid_longitude', 'Issue Numbers', 'Complaint Count'])
            st.write(result_df)

            # Save to Excel
        except ValueError as e:
            st.error(str(e))