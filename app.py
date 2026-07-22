import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import folium
from streamlit_folium import st_folium
from scipy.spatial.distance import cdist

# ==========================================
# 1. PAGE CONFIGURATION & LAYOUT
# ==========================================
st.set_page_config(page_title="Earthquake Severity Predictor", layout="wide")

# ==========================================
# 2. MODEL & CLUSTERING SETUP
# ==========================================
# Load the saved Random Forest model
@st.cache_resource
def load_model():
    try:
        return joblib.load('earthquake_classifier_model.pkl')
    except:
        return None

model = load_model()

# --- CENTROIDS ---
CENTROIDS = np.array([
    [32.721082, 80.055818],
    [10.180451, 93.506703],
    [35.651955, 71.125384],
    [24.100781, 94.409344],
    [21.850513, 72.073490],
    [31.257249, 89.696732]
])

# Define the acceptable bounding box for the Indian Region
# (Roughly Latitude 5.0 to 40.0, Longitude 65.0 to 100.0)
LAT_MIN, LAT_MAX = 5.0, 40.0
LON_MIN, LON_MAX = 65.0, 100.0

# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if 'lat' not in st.session_state:
    st.session_state.lat = 20.5937 
if 'lon' not in st.session_state:
    st.session_state.lon = 78.9629
if 'depth' not in st.session_state:
    st.session_state.depth = 50.0
if 'magmb' not in st.session_state:
    st.session_state.magmb = 4.5

# ==========================================
# 4. APP LAYOUT (25% Left, 75% Right)
# ==========================================
col1, col2 = st.columns([1, 3])

# ------------------------------------------
# RIGHT PANEL: INTERACTIVE MAP (Render first to catch clicks)
# ------------------------------------------
with col2:
    st.markdown("### 🗺️ Interactive Seismic Map")
    st.caption("Click anywhere on the map within the Indian region to set the coordinates.")
    
    # Create the base map
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=5)
    
    # Add a bounding box rectangle to show the user the allowed region
    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        color='#ff7800',
        fill=True,
        fill_opacity=0.05,
        fill_color='#ff7800',
        weight=1
    ).add_to(m)

    # Add a marker at the current selected location
    folium.Marker(
        [st.session_state.lat, st.session_state.lon], 
        tooltip="Selected Location",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Render the map and catch any clicks
    # height=700 ensures it takes up a good portion of the screen
    map_data = st_folium(m, width="100%", height=700)

    # If the user clicked the map, update the coordinates and refresh the UI
    if map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]
        
        # Only update if the click is within our defined Indian Region bounds
        if (LAT_MIN <= click_lat <= LAT_MAX) and (LON_MIN <= click_lon <= LON_MAX):
            if click_lat != st.session_state.lat or click_lon != st.session_state.lon:
                st.session_state.lat = click_lat
                st.session_state.lon = click_lon
                st.rerun()
        else:
            st.warning("⚠️ Please select a location within the Indian region bounding box.")

# ------------------------------------------
# LEFT PANEL: INPUTS & PREDICTION
# ------------------------------------------
with col1:
    st.markdown("### ⚙️ Sensor Inputs")
    
    # The Randomizer Button
    if st.button("🎲 Randomize Inputs", use_container_width=True):
        st.session_state.lat = round(random.uniform(LAT_MIN, LAT_MAX), 4)
        st.session_state.lon = round(random.uniform(LON_MIN, LON_MAX), 4)
        st.session_state.depth = round(random.uniform(5.0, 300.0), 1) # Realistic depth range
        st.session_state.magmb = round(random.uniform(3.5, 7.5), 1)   # Realistic mag range
        st.rerun() # Refresh the app to show new values

    st.markdown("---")
    
    # Input fields now linked directly to session_state
    lat_input = st.number_input("Latitude", min_value=LAT_MIN, max_value=LAT_MAX, value=float(st.session_state.lat), format="%.4f")
    lon_input = st.number_input("Longitude", min_value=LON_MIN, max_value=LON_MAX, value=float(st.session_state.lon), format="%.4f")
    # Sync manual typing back to the map
    if lat_input != st.session_state.lat or lon_input != st.session_state.lon:
        st.session_state.lat = lat_input
        st.session_state.lon = lon_input
        st.rerun()

    depth = st.number_input("Depth (KM)", min_value=0.0, max_value=800.0, value=float(st.session_state.depth), step=1.0)
    magmb = st.number_input("Body Wave Magnitude (MAGMB)", min_value=0.0, max_value=10.0, value=float(st.session_state.magmb), step=0.1)

    # Sync manual typing for depth and magmb
    st.session_state.depth = depth
    st.session_state.magmb = magmb

    st.markdown("---")
    
    if st.button("Predict Moment Magnitude", type="primary", use_container_width=True):
        if model is None:
            st.error("Model file not found! Please ensure 'earthquake_classifier_model.pkl' is in the same folder.")
        else:
            # 1. Calculate the closest K-Means Centroid (Seismic Zone ID)
            user_coords = np.array([[lat_input, lon_input]])
            distances = cdist(user_coords, CENTROIDS, metric='euclidean')
            closest_zone_id = np.argmin(distances)
            
            # 2. Prepare the feature dictionary
            feature_dict = {
                'DEPTH_KM': [depth],
                'MAGMB': [magmb],
                'Seismic_Zone_ID_0': [0], 'Seismic_Zone_ID_1': [0], 'Seismic_Zone_ID_2': [0], 
                'Seismic_Zone_ID_3': [0], 'Seismic_Zone_ID_4': [0], 'Seismic_Zone_ID_5': [0]
            }
            
            # Set the one-hot encoded zone to 1
            feature_dict[f'Seismic_Zone_ID_{closest_zone_id}'] = [1]
            
            # Convert to DataFrame
            input_df = pd.DataFrame(feature_dict)
            
            # 3. Make Prediction
            prediction = model.predict(input_df)[0]
            
            # 4. Display Results
            st.markdown("### Prediction Result:")
            
            if prediction == 0:
                st.success("**LIGHT** Earthquake (< 4.5 MW)")
            elif prediction == 1:
                st.warning("**MODERATE** Earthquake (4.5 - 5.4 MW)")
            elif prediction == 2:
                st.error("**STRONG** Earthquake (>= 5.5 MW)")
            
            st.caption(f"Calculated Seismic Zone ID: {closest_zone_id}")
