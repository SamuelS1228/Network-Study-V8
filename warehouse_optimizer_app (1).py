import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import math
import random
from io import StringIO
import base64
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance

# Set page configuration
st.set_page_config(
    page_title="Warehouse Location Optimizer",
    page_icon="🏭",
    layout="wide"
)

# Title and description
st.title("Warehouse Location Optimizer")
st.markdown("""
This app helps you determine the optimal locations for warehouses based on store demand and transportation costs.
Upload your store data with locations and demand information to get started.
""")

# Continental US boundaries
CONTINENTAL_US = {
    "min_lat": 24.396308,  # Southern tip of Florida
    "max_lat": 49.384358,  # Northern border with Canada
    "min_lon": -124.848974,  # Western coast
    "max_lon": -66.885444   # Eastern coast
}

# Function to calculate distance between two points using Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 3956  # Radius of earth in miles
    return c * r

# Function to check if point is within continental US
def is_in_continental_us(lat, lon):
    return (CONTINENTAL_US["min_lat"] <= lat <= CONTINENTAL_US["max_lat"] and 
            CONTINENTAL_US["min_lon"] <= lon <= CONTINENTAL_US["max_lon"])

# Function to calculate transportation cost
def calculate_transportation_cost(distance, weight, rate):
    return distance * weight * rate

# Function to generate example data with consistent seed
def generate_example_data(num_stores=100, seed=42):
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random points within continental US
    data = []
    for i in range(num_stores):
        lat = random.uniform(CONTINENTAL_US["min_lat"], CONTINENTAL_US["max_lat"])
        lon = random.uniform(CONTINENTAL_US["min_lon"], CONTINENTAL_US["max_lon"])
        # Generate random yearly demand between 10,000 and 500,000 pounds
        yearly_demand = round(random.uniform(10000, 500000))
        data.append({"store_id": f"Store_{i+1}", "latitude": lat, "longitude": lon, "yearly_demand_lbs": yearly_demand})
    
    return pd.DataFrame(data)

# Function to download dataframe as CSV
def download_link(dataframe, filename, link_text):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to generate distinct colors for warehouses
def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate colors using HSV color space for better distinction
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB (simplified version)
        h = hue * 6
        c = 255
        x = 255 * (1 - abs(h % 2 - 1))
        
        if h < 1:
            rgb = [c, x, 0]
        elif h < 2:
            rgb = [x, c, 0]
        elif h < 3:
            rgb = [0, c, x]
        elif h < 4:
            rgb = [0, x, c]
        elif h < 5:
            rgb = [x, 0, c]
        else:
            rgb = [c, 0, x]
            
        colors.append(rgb)
    return colors

# Sidebar for uploading data and parameters
st.sidebar.header("Upload Store Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with store locations and demand data", type="csv")

# Option to use example data
use_example_data = st.sidebar.checkbox("Use example data instead", value=False)

# Sample data format explanation
with st.sidebar.expander("CSV Format Requirements"):
    st.write("""
    Your CSV file should include the following columns:
    - `store_id`: Unique identifier for each store
    - `latitude`: Store latitude
    - `longitude`: Store longitude
    - `yearly_demand_lbs`: Annual demand in pounds
    """)
    
    # Display sample data
    sample_df = pd.DataFrame({
        "store_id": ["Store_1", "Store_2", "Store_3"],
        "latitude": [40.7128, 34.0522, 41.8781],
        "longitude": [-74.0060, -118.2437, -87.6298],
        "yearly_demand_lbs": [250000, 175000, 320000]
    })
    
    st.dataframe(sample_df)
    st.markdown(download_link(sample_df, "sample_store_data.csv", "Download Sample CSV"), unsafe_allow_html=True)

# Optimization parameters
st.sidebar.header("Optimization Parameters")
num_warehouses = st.sidebar.slider("Number of Warehouses", min_value=1, max_value=20, value=3)
cost_per_pound_mile = st.sidebar.number_input("Transportation Cost Rate ($ per pound-mile)", min_value=0.0001, max_value=1.0, value=0.001, format="%.5f")
max_iterations = st.sidebar.slider("Max Optimization Iterations", min_value=10, max_value=100, value=50)

# Select optimization algorithm
st.sidebar.header("Optimization Method")
optimization_method = st.sidebar.radio(
    "Select optimization method:",
    ["KMeans-Weighted", "Enhanced Iterative", "Standard Iterative"],
    index=0  # Default to KMeans-Weighted
)

# Add an option to set seed for reproducibility
use_fixed_seed = st.sidebar.checkbox("Use fixed seed for reproducibility", value=True)
if use_fixed_seed:
    seed_value = st.sidebar.number_input("Seed value", min_value=1, max_value=10000, value=42)
else:
    seed_value = None

# Main app logic
if uploaded_file is not None:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)
    data_source = "uploaded"
elif use_example_data:
    # Generate example data with seed for reproducibility
    df = generate_example_data(seed=42 if use_fixed_seed else None)
    data_source = "example"
else:
    st.info("Please upload a CSV file or use example data to get started.")
    st.stop()

# Check if required columns exist
required_cols = ["store_id", "latitude", "longitude", "yearly_demand_lbs"]
if not all(col in df.columns for col in required_cols):
    st.error(f"The data must contain these columns: {', '.join(required_cols)}")
    st.stop()

# Display the data
st.subheader("Store Data")
st.dataframe(df)

# Function to normalize data for clustering
def normalize_locations(df):
    # Create a copy to avoid modifying the original dataframe
    locations = df[['latitude', 'longitude']].copy()
    
    # Min-max scaling for lat/lon to handle the Earth's curvature better
    for col in ['latitude', 'longitude']:
        min_val = locations[col].min()
        max_val = locations[col].max()
        locations[col] = (locations[col] - min_val) / (max_val - min_val)
    
    return locations

# KMeans-based warehouse location optimization
def kmeans_weighted_optimization(stores_df, n_warehouses, seed=None):
    # Set the seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Normalize latitude and longitude for better clustering
    locations = normalize_locations(stores_df)
    
    # Add demand weights
    weights = stores_df['yearly_demand_lbs'].values
    
    # Create a weighted dataset by repeating points based on their weights
    # Normalize weights first to avoid creating too many points
    norm_weights = np.ceil(weights / weights.min()).astype(int)
    
    # Cap the number of repetitions to keep computation manageable
    max_repetitions = 100
    if max(norm_weights) > max_repetitions:
        scaling_factor = max_repetitions / max(norm_weights)
        norm_weights = np.ceil(norm_weights * scaling_factor).astype(int)
    
    weighted_locations = np.vstack([np.tile(locations.iloc[i], (rep, 1)) 
                                   for i, rep in enumerate(norm_weights)])
    
    # Show progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Initializing K-Means clustering...")
    
    # KMeans clustering with multiple initializations to find the best solution
    best_kmeans = None
    best_score = -np.inf
    
    n_init = 10  # Number of times to run kmeans with different initializations
    for i in range(n_init):
        progress = int((i + 1) / n_init * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Running K-Means optimization: {i+1}/{n_init}")
        
        # Run KMeans
        kmeans = KMeans(n_clusters=n_warehouses, init='k-means++', n_init=1, random_state=seed+i if seed is not None else None)
        kmeans.fit(weighted_locations)
        
        # Evaluate the clustering quality
        if len(np.unique(kmeans.labels_)) == n_warehouses:  # Ensure we have the requested number of clusters
            try:
                score = silhouette_score(weighted_locations, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_kmeans = kmeans
            except:
                # If silhouette score fails (e.g., with only one sample in a cluster)
                pass
    
    if best_kmeans is None:
        # If no valid clustering was found, fall back to standard KMeans
        kmeans = KMeans(n_clusters=n_warehouses, random_state=seed)
        kmeans.fit(locations)
        best_kmeans = kmeans
    
    # Get the warehouse locations (cluster centers)
    # Denormalize to get actual latitude and longitude
    warehouse_locs = []
    for i, center in enumerate(best_kmeans.cluster_centers_):
        # Denormalize
        lat = center[0] * (stores_df['latitude'].max() - stores_df['latitude'].min()) + stores_df['latitude'].min()
        lon = center[1] * (stores_df['longitude'].max() - stores_df['longitude'].min()) + stores_df['longitude'].min()
        
        # Ensure the warehouse is within continental US
        if not is_in_continental_us(lat, lon):
            # If not, find the closest valid point
            valid_points = stores_df[['latitude', 'longitude']].values
            distances = np.sqrt((valid_points[:, 0] - lat)**2 + (valid_points[:, 1] - lon)**2)
            closest_idx = np.argmin(distances)
            lat = stores_df.iloc[closest_idx]['latitude']
            lon = stores_df.iloc[closest_idx]['longitude']
        
        warehouse_locs.append({
            "warehouse_id": f"WH_{i+1}",
            "latitude": lat,
            "longitude": lon
        })
    
    # Create DataFrame for warehouse locations
    warehouses_df = pd.DataFrame(warehouse_locs)
    
    # Assign each store to the closest warehouse
    assignments = []
    total_cost = 0
    
    for _, store in stores_df.iterrows():
        min_cost = float('inf')
        assigned_wh = None
        min_distance = 0
        
        for _, wh in warehouses_df.iterrows():
            distance = haversine(store["longitude"], store["latitude"], 
                                wh["longitude"], wh["latitude"])
            cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
            
            if cost < min_cost:
                min_cost = cost
                assigned_wh = wh["warehouse_id"]
                min_distance = distance
        
        assignments.append({
            "store_id": store["store_id"],
            "warehouse_id": assigned_wh,
            "distance_miles": min_distance,
            "transportation_cost": min_cost
        })
        
        total_cost += min_cost
    
    assignments_df = pd.DataFrame(assignments)
    
    progress_bar.progress(100)
    progress_text.text("Optimization completed successfully!")
    
    return warehouses_df, assignments_df, total_cost

# Enhanced iterative optimization with better initialization
def enhanced_iterative_optimization(stores_df, n_warehouses, max_iterations=50, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize warehouse locations using k-means++ strategy for better starting points
    # This helps ensure we start with a good initial distribution
    locations = stores_df[['latitude', 'longitude']].values
    
    # Initialize first warehouse at the point with highest demand
    highest_demand_idx = stores_df['yearly_demand_lbs'].argmax()
    centroids = [locations[highest_demand_idx]]
    
    # Initialize remaining warehouses with k-means++ logic
    for _ in range(1, n_warehouses):
        # Calculate distances from each point to nearest existing centroid
        dist_sq = np.array([min([np.sum((x-c)**2) for c in centroids]) for x in locations])
        
        # Select next centroid with probability proportional to dist_sq
        probs = dist_sq / dist_sq.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.searchsorted(cumprobs, r)
        centroids.append(locations[ind])
    
    # Create initial warehouse dataframe
    warehouses = []
    for i, centroid in enumerate(centroids):
        warehouses.append({
            "warehouse_id": f"WH_{i+1}",
            "latitude": centroid[0],
            "longitude": centroid[1]
        })
    
    warehouses_df = pd.DataFrame(warehouses)
    
    # Show initial progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Iterative optimization
    prev_cost = float('inf')
    for iteration in range(max_iterations):
        # Update progress
        progress = int((iteration + 1) / max_iterations * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Optimizing: Iteration {iteration + 1}/{max_iterations}")
        
        # Assign each store to closest warehouse
        assignments = []
        total_cost = 0
        
        for _, store in stores_df.iterrows():
            min_cost = float('inf')
            assigned_wh = None
            min_distance = 0
            
            for _, wh in warehouses_df.iterrows():
                distance = haversine(store["longitude"], store["latitude"], 
                                    wh["longitude"], wh["latitude"])
                cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
                
                if cost < min_cost:
                    min_cost = cost
                    assigned_wh = wh["warehouse_id"]
                    min_distance = distance
            
            assignments.append({
                "store_id": store["store_id"],
                "warehouse_id": assigned_wh,
                "distance_miles": min_distance,
                "transportation_cost": min_cost
            })
            
            total_cost += min_cost
        
        assignments_df = pd.DataFrame(assignments)
        
        # Check convergence
        if abs(prev_cost - total_cost) < 0.1:  # Tighter convergence criterion
            progress_bar.progress(100)
            progress_text.text(f"Optimization completed in {iteration + 1} iterations")
            break
        
        prev_cost = total_cost
        
        # Update warehouse locations to weighted centroid of assigned stores
        for _, wh in warehouses_df.iterrows():
            wh_id = wh["warehouse_id"]
            assigned_stores = assignments_df[assignments_df["warehouse_id"] == wh_id]
            
            if len(assigned_stores) > 0:
                # Get the actual store data
                store_indices = [stores_df[stores_df["store_id"] == store_id].index[0] 
                                for store_id in assigned_stores["store_id"]]
                assigned_stores_data = stores_df.iloc[store_indices]
                
                # Calculate weighted centroid based on demand
                total_demand = assigned_stores_data["yearly_demand_lbs"].sum()
                
                if total_demand > 0:
                    weighted_lat = (assigned_stores_data["latitude"] * assigned_stores_data["yearly_demand_lbs"]).sum() / total_demand
                    weighted_lon = (assigned_stores_data["longitude"] * assigned_stores_data["yearly_demand_lbs"]).sum() / total_demand
                    
                    # Ensure the warehouse is within continental US
                    if is_in_continental_us(weighted_lat, weighted_lon):
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "latitude"] = weighted_lat
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "longitude"] = weighted_lon
    
    # If we've reached max iterations without convergence
    if iteration == max_iterations - 1:
        progress_bar.progress(100)
        progress_text.text(f"Optimization completed after maximum {max_iterations} iterations")
    
    return warehouses_df, assignments_df, total_cost

# Standard iterative optimization from the original code
def standard_iterative_optimization(stores_df, n_warehouses, max_iterations=50, seed=None):
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Initialize random warehouse locations within continental US boundaries
    warehouses = []
    
    while len(warehouses) < n_warehouses:
        lat = random.uniform(CONTINENTAL_US["min_lat"], CONTINENTAL_US["max_lat"])
        lon = random.uniform(CONTINENTAL_US["min_lon"], CONTINENTAL_US["max_lon"])
        if is_in_continental_us(lat, lon):
            warehouses.append({
                "warehouse_id": f"WH_{len(warehouses)+1}",
                "latitude": lat,
                "longitude": lon
            })
    
    warehouses_df = pd.DataFrame(warehouses)
    
    # Show initial progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Iterative optimization
    prev_cost = float('inf')
    for iteration in range(max_iterations):
        # Update progress
        progress = int((iteration + 1) / max_iterations * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Optimizing: Iteration {iteration + 1}/{max_iterations}")
        
        # Assign each store to closest warehouse
        assignments = []
        total_cost = 0
        
        for _, store in stores_df.iterrows():
            min_cost = float('inf')
            assigned_wh = None
            min_distance = 0
            
            for _, wh in warehouses_df.iterrows():
                distance = haversine(store["longitude"], store["latitude"], 
                                    wh["longitude"], wh["latitude"])
                cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
                
                if cost < min_cost:
                    min_cost = cost
                    assigned_wh = wh["warehouse_id"]
                    min_distance = distance
            
            assignments.append({
                "store_id": store["store_id"],
                "warehouse_id": assigned_wh,
                "distance_miles": min_distance,
                "transportation_cost": min_cost
            })
            
            total_cost += min_cost
        
        assignments_df = pd.DataFrame(assignments)
        
        # Check convergence
        if abs(prev_cost - total_cost) < 1:
            progress_bar.progress(100)
            progress_text.text(f"Optimization completed in {iteration + 1} iterations")
            break
        
        prev_cost = total_cost
        
        # Update warehouse locations to center of assigned stores
        for _, wh in warehouses_df.iterrows():
            wh_id = wh["warehouse_id"]
            assigned_stores_indices = assignments_df.index[assignments_df["warehouse_id"] == wh_id].tolist()
            assigned_stores = stores_df.iloc[assigned_stores_indices]
            
            if len(assigned_stores) > 0:
                # Calculate weighted centroid based on demand
                total_demand = assigned_stores["yearly_demand_lbs"].sum()
                
                if total_demand > 0:
                    weighted_lat = (assigned_stores["latitude"] * assigned_stores["yearly_demand_lbs"]).sum() / total_demand
                    weighted_lon = (assigned_stores["longitude"] * assigned_stores["yearly_demand_lbs"]).sum() / total_demand
                    
                    # Ensure the warehouse is within continental US
                    if is_in_continental_us(weighted_lat, weighted_lon):
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "latitude"] = weighted_lat
                        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "longitude"] = weighted_lon
    
    # If we've reached max iterations without convergence
    if iteration == max_iterations - 1:
        progress_bar.progress(100)
        progress_text.text(f"Optimization completed after maximum {max_iterations} iterations")
    
    return warehouses_df, assignments_df, total_cost

# Run optimization
if st.button("Run Optimization"):
    # Use the selected optimization method
    if optimization_method == "KMeans-Weighted":
        optimized_warehouses, store_assignments, total_transportation_cost = kmeans_weighted_optimization(
            df, num_warehouses, seed=seed_value if use_fixed_seed else None)
    elif optimization_method == "Enhanced Iterative":
        optimized_warehouses, store_assignments, total_transportation_cost = enhanced_iterative_optimization(
            df, num_warehouses, max_iterations, seed=seed_value if use_fixed_seed else None)
    else:  # Standard Iterative
        optimized_warehouses, store_assignments, total_transportation_cost = standard_iterative_optimization(
            df, num_warehouses, max_iterations, seed=seed_value if use_fixed_seed else None)
    
    # Store results in session state
    st.session_state.optimized_warehouses = optimized_warehouses
    st.session_state.store_assignments = store_assignments
    st.session_state.total_transportation_cost = total_transportation_cost
    st.session_state.optimization_complete = True
    st.session_state.optimization_method = optimization_method
else:
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False

# Display results if optimization is complete
if st.session_state.optimization_complete:
    optimized_warehouses = st.session_state.optimized_warehouses
    store_assignments = st.session_state.store_assignments
    total_transportation_cost = st.session_state.total_transportation_cost
    used_optimization_method = st.session_state.optimization_method
    
    # Display metrics
    st.subheader("Optimization Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Warehouses", num_warehouses)
    
    with col2:
        st.metric("Total Transportation Cost", f"${total_transportation_cost:,.2f}")
    
    with col3:
        avg_cost_per_store = total_transportation_cost / len(df)
        st.metric("Avg. Cost per Store", f"${avg_cost_per_store:,.2f}")
    
    with col4:
        st.metric("Optimization Method", used_optimization_method)
    
    # Calculate additional metrics
    warehouse_metrics = store_assignments.groupby("warehouse_id").agg(
        num_stores=("store_id", "count"),
        total_cost=("transportation_cost", "sum"),
        avg_distance=("distance_miles", "mean"),
        max_distance=("distance_miles", "max"),
        total_demand=pd.NamedAgg(column="store_id", aggfunc=lambda x: df.loc[df['store_id'].isin(x), 'yearly_demand_lbs'].sum())
    ).reset_index()
    
    # Join with warehouse locations
    warehouse_metrics = warehouse_metrics.merge(optimized_warehouses, on="warehouse_id")
    
    # Generate colors for warehouses
    warehouse_colors = generate_colors(len(optimized_warehouses))
    warehouse_color_map = {wh: color for wh, color in zip(optimized_warehouses['warehouse_id'], warehouse_colors)}
    
    # Create a DataFrame that includes both warehouse and store info for visualization
    warehouse_data_for_map = optimized_warehouses.copy()
    warehouse_data_for_map["type"] = "warehouse"
    warehouse_data_for_map = warehouse_data_for_map.merge(
        warehouse_metrics[["warehouse_id", "num_stores", "total_cost", "total_demand"]], 
        on="warehouse_id"
    )
    
    # Add color for each warehouse
    for i, wh_id in enumerate(warehouse_data_for_map['warehouse_id']):
        warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_r'] = warehouse_colors[i][0]
        warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_g'] = warehouse_colors[i][1]
        warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_b'] = warehouse_colors[i][2]
    
    # Add store information for visualization
    store_data_for_map = df.copy()
    store_data_for_map["type"] = "store"
    
    # Merge with store assignments
    store_assignments_with_ids = store_assignments.copy()
    store_assignments_with_ids["store_idx"] = store_assignments_with_ids["store_id"].apply(
        lambda x: df.index[df["store_id"] == x].tolist()[0] if any(df["store_id"] == x) else None
    )
    
    store_data_for_map = store_data_for_map.merge(
        store_assignments[["store_id", "warehouse_id", "distance_miles", "transportation_cost"]], 
        on="store_id"
    )
    
    # Add color for each store based on its assigned warehouse
    for wh_id in warehouse_data_for_map['warehouse_id']:
        color = warehouse_color_map[wh_id]
        mask = store_data_for_map['warehouse_id'] == wh_id
        store_data_for_map.loc[mask, 'color_r'] = color[0]
        store_data_for_map.loc[mask, 'color_g'] = color[1]
        store_data_for_map.loc[mask, 'color_b'] = color[2]
    
    # Create a list of lines connecting stores to warehouses for the map
    lines = []
    for _, store in store_data_for_map.iterrows():
        warehouse = warehouse_data_for_map[warehouse_data_for_map["warehouse_id"] == store["warehouse_id"]].iloc[0]
        # Get the color from the warehouse
        color = [
            warehouse['color_r'],
            warehouse['color_g'],
            warehouse['color_b']
        ]
        
        lines.append({
            "start_lat": store["latitude"],
            "start_lon": store["longitude"],
            "end_lat": warehouse["latitude"],
            "end_lon": warehouse["longitude"],
            "store_id": store["store_id"],
            "warehouse_id": warehouse["warehouse_id"],
            "color_r": color[0],
            "color_g": color[1],
            "color_b": color[2]
        })
    
    lines_df = pd.DataFrame(lines)
    
    # Map showing stores and warehouses with enhanced warehouse representation
    st.subheader("Map Visualization")
    
    # Create layers for the map
    store_layer = pdk.Layer(
        "ScatterplotLayer",
        data=store_data_for_map,
        get_position=["longitude", "latitude"],
        get_radius=100,  # Radius for better visibility
        get_fill_color=["color_r", "color_g", "color_b", 200],  # Color based on warehouse assignment
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    # Line layer connecting stores to warehouses
    line_layer = pdk.Layer(
        "LineLayer",
        data=lines_df,
        get_source_position=["start_lon", "start_lat"],
        get_target_position=["end_lon", "end_lat"],
        get_color=["color_r", "color_g", "color_b", 150],  # Increased opacity, colors match warehouse
        get_width=2,  # Increased line width
        pickable=True,
    )
    
    # Enhanced warehouse representation with diamond shape and border
    warehouse_layer = pdk.Layer(
        "ScatterplotLayer",
        data=warehouse_data_for_map,
        get_position=["longitude", "latitude"],
        get_radius=1200,  # Very large radius for warehouses
        get_fill_color=["color_r", "color_g", "color_b", 250],  # Fill color from palette
        get_line_color=[0, 0, 0, 200],  # Black border
        get_line_width=10,  # Very thick border
        pickable=True,
        opacity=1.0,
        stroked=True,
        filled=True,
    )
    
    # Text layer for warehouse labels
    text_layer = pdk.Layer(
        "TextLayer",
        data=warehouse_data_for_map,
        get_position=["longitude", "latitude"],
        get_text="warehouse_id",
        get_size=18,
        get_color=[0, 0, 0],  # Black text
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
        pickable=True,
    )
    
    # Create the map with the enhanced layers
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=np.mean(df["latitude"]),
            longitude=np.mean(df["longitude"]),
            zoom=3,
            pitch=0,
        ),
        layers=[line_layer, store_layer, warehouse_layer, text_layer],
        tooltip={
            "html": "<b>ID:</b> {store_id or warehouse_id}<br><b>Type:</b> {type}<br><b>Demand:</b> {yearly_demand_lbs} lbs<br><b>Cost:</b> ${transportation_cost}",
            "style": {"background": "white", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        },
    ))
    
    # Add a legend to explain the visualization
    st.markdown("""
    ### Map Legend
    - **Large Circles with Black Borders**: Warehouses (optimized locations)
    - **Small Dots**: Stores (colored by their assigned warehouse)
    - **Lines**: Connections between stores and their assigned warehouses
    """)
    
    # Show detailed metrics
    st.subheader("Warehouse Details")
    
    # Add advanced metrics
    warehouse_metrics_display = warehouse_metrics.copy()
    warehouse_metrics_display["avg_cost_per_store"] = warehouse_metrics_display["total_cost"] / warehouse_metrics_display["num_stores"]
    warehouse_metrics_display["cost_per_demand"] = warehouse_metrics_display["total_cost"] / warehouse_metrics_display["total_demand"]
    
    # Format metrics for display
    warehouse_metrics_display["total_cost"] = warehouse_metrics_display["total_cost"].map("${:,.2f}".format)
    warehouse_metrics_display["avg_cost_per_store"] = warehouse_metrics_display["avg_cost_per_store"].map("${:,.2f}".format)
    warehouse_metrics_display["cost_per_demand"] = warehouse_metrics_display["cost_per_demand"].map("${:,.5f}".format)
    warehouse_metrics_display["total_demand"] = warehouse_metrics_display["total_demand"].map("{:,.0f} lbs".format)
    warehouse_metrics_display["avg_distance"] = warehouse_metrics_display["avg_distance"].map("{:,.1f} miles".format)
    warehouse_metrics_display["max_distance"] = warehouse_metrics_display["max_distance"].map("{:,.1f} miles".format)
    
    st.dataframe(warehouse_metrics_display)
    
    # Create expanded visualization section
    st.subheader("Performance Metrics")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stores per Warehouse")
        # Create a DataFrame for the bar chart
        chart_data1 = warehouse_metrics[["warehouse_id", "num_stores"]].set_index("warehouse_id")
        st.bar_chart(chart_data1)
    
    with col2:
        st.subheader("Cost per Warehouse")
        # Create a DataFrame for the bar chart
        chart_data2 = warehouse_metrics[["warehouse_id", "total_cost"]].set_index("warehouse_id")
        st.bar_chart(chart_data2)
    
    # Additional metrics - Demand distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demand per Warehouse")
        # Create a DataFrame for the bar chart
        chart_data3 = warehouse_metrics[["warehouse_id", "total_demand"]].set_index("warehouse_id")
        st.bar_chart(chart_data3)
    
    with col2:
        st.subheader("Average Distance per Warehouse")
        # Create a DataFrame for the bar chart
        chart_data4 = warehouse_metrics[["warehouse_id", "avg_distance"]].set_index("warehouse_id")
        st.bar_chart(chart_data4)
    
    # Warehouse efficiency comparison
    st.subheader("Warehouse Efficiency Comparison")
    
    # Calculate cost per demand unit for each warehouse
    efficiency_data = warehouse_metrics.copy()
    efficiency_data["cost_per_demand_unit"] = efficiency_data["total_cost"] / efficiency_data["total_demand"]
    
    # Sort by efficiency for the chart
    efficiency_data = efficiency_data.sort_values("cost_per_demand_unit")
    
    # Create a DataFrame for the bar chart
    chart_data5 = efficiency_data[["warehouse_id", "cost_per_demand_unit"]].set_index("warehouse_id")
    st.bar_chart(chart_data5)
    st.caption("Lower cost per demand unit indicates better efficiency")
    
    # Store details section
    st.subheader("Store Assignments")
    
    # Store details with merge to show warehouse data
    store_details = store_data_for_map.merge(
        optimized_warehouses[["warehouse_id", "latitude", "longitude"]], 
        on="warehouse_id",
        suffixes=("_store", "_warehouse")
    )
    
    # Add formatted distance and cost columns
    store_details["distance_miles_formatted"] = store_details["distance_miles"].map("{:,.1f} miles".format)
    store_details["transportation_cost_formatted"] = store_details["transportation_cost"].map("${:,.2f}".format)
    
    # Display interactive table
    st.dataframe(store_details)
    
    # Download results section
    st.subheader("Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(download_link(warehouse_metrics, "optimized_warehouses.csv", "Download Warehouse Data"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(download_link(store_details, "store_assignments.csv", "Download Store Assignments"), unsafe_allow_html=True)
    
    with col3:
        # Export full solution details
        full_solution = {
            "optimization_method": used_optimization_method,
            "num_warehouses": num_warehouses,
            "cost_per_pound_mile": cost_per_pound_mile,
            "total_transportation_cost": total_transportation_cost,
            "seed_value": seed_value if use_fixed_seed else "None"
        }
        full_solution_df = pd.DataFrame([full_solution])
        st.markdown(download_link(full_solution_df, "optimization_solution.csv", "Download Solution Details"), unsafe_allow_html=True)
    
    # Comparison with previous runs (if there are any saved)
    if 'previous_runs' not in st.session_state:
        st.session_state.previous_runs = []
    
    # Save current run
    current_run = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimization_method": used_optimization_method,
        "num_warehouses": num_warehouses,
        "total_cost": total_transportation_cost,
        "seed_value": seed_value if use_fixed_seed else "None"
    }
    
    # Add current run to previous runs if different from last run
    if len(st.session_state.previous_runs) == 0 or st.session_state.previous_runs[-1] != current_run:
        st.session_state.previous_runs.append(current_run)
    
    # Display previous runs if there are any
    if len(st.session_state.previous_runs) > 1:
        st.subheader("Comparison with Previous Runs")
        previous_runs_df = pd.DataFrame(st.session_state.previous_runs)
        st.dataframe(previous_runs_df)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Warehouse Location Optimizer - Powered by Streamlit")
    
    # Add information about the optimization methods
    with st.expander("About the Optimization Methods"):
        st.markdown("""
        ### Optimization Methods
        
        This app offers three different optimization methods:
        
        1. **KMeans-Weighted**: Uses K-means clustering with demand weighting to find optimal warehouse locations. This method provides the most consistent results and is the recommended approach for most scenarios. It leverages scikit-learn's implementation of K-means with multiple initializations to find the global optimum.
        
        2. **Enhanced Iterative**: An improved version of the iterative approach that uses k-means++ style initialization for better starting points. This helps avoid poor local optima and generally converges faster than the standard method.
        
        3. **Standard Iterative**: The original approach with random initialization. This can sometimes get stuck in local optima, leading to different results on different runs.
        
        For consistent results, use the "KMeans-Weighted" method with the "Use fixed seed" option enabled.
        """)
