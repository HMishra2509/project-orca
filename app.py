import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Project ORCA", layout="wide")

st.title("🐋 Project ORCA")
st.caption("Oil-spill Recognition, Correlation & Attribution")

# ---------------- ENGINE 1 FUNCTIONS ----------------
def analyze_oil_spill(image_path, resolution_m_per_pixel=10):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    margin = 5
    img = img[margin:-margin, margin:-margin]
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spill_contours = [c for c in contours if cv2.contourArea(c) > 100]
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, spill_contours, -1, (0, 0, 255), 2)
    return img, cleaned, output, spill_contours

# ---------------- ENGINE 2 FUNCTIONS ----------------
def calculate_drift_position(start_lat, start_lon, current_speed_kmph, current_dir_deg,
                               wind_speed_kmph, wind_dir_deg, wind_drift_factor,
                               hours, direction="forward"):
    current_rad = math.radians(current_dir_deg)
    wind_rad = math.radians(wind_dir_deg)
    effective_speed_x = (current_speed_kmph * math.sin(current_rad)) + \
                         (wind_speed_kmph * wind_drift_factor * math.sin(wind_rad))
    effective_speed_y = (current_speed_kmph * math.cos(current_rad)) + \
                         (wind_speed_kmph * wind_drift_factor * math.cos(wind_rad))
    if direction == "backward":
        effective_speed_x *= -1
        effective_speed_y *= -1
    distance_x_km = effective_speed_x * hours
    distance_y_km = effective_speed_y * hours
    delta_lat = distance_y_km / 111.0
    delta_lon = distance_x_km / (111.0 * math.cos(math.radians(start_lat)))
    return round(start_lat + delta_lat, 5), round(start_lon + delta_lon, 5)

# ---------------- ENGINE 3 FUNCTIONS ----------------
def calculate_suspect_score(row):
    score = 0
    if row['distance_from_spill_km'] <= 5:
        score += 35
    elif row['distance_from_spill_km'] <= 10:
        score += 20
    elif row['distance_from_spill_km'] <= 20:
        score += 10
    if row['ais_signal_gap']:
        score += 30
    score += row['trajectory_match'] * 25
    if row['ship_type'] == "Oil Tanker":
        score += 10
    elif row['ship_type'] == "Cargo Ship":
        score += 5
    return round(score, 1)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Spill Parameters")
image_path = st.sidebar.text_input("Image path", "data/class_1_01780.jpg")
spill_lat = st.sidebar.number_input("Spill Latitude", value=15.2993, format="%.5f")
spill_lon = st.sidebar.number_input("Spill Longitude", value=74.1240, format="%.5f")
current_speed = st.sidebar.number_input("Current Speed (km/h)", value=1.5)
current_dir = st.sidebar.number_input("Current Direction (deg)", value=45)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", value=3.0)
wind_dir = st.sidebar.number_input("Wind Direction (deg)", value=60)

if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False

if st.sidebar.button("Run ORCA Pipeline"):
    st.session_state.run_pipeline = True

run_button = st.session_state.run_pipeline

# Sample ship data
ships_data = {
    "ship_name": ["MV Horizon", "SS Coral Queen", "MV Deccan Trader", "MT Sagar Prince", "MV Bay Runner"],
    "ship_type": ["Oil Tanker", "Passenger Ferry", "Cargo Ship", "Oil Tanker", "Cargo Ship"],
    "distance_from_spill_km": [2.1, 18.5, 6.4, 3.8, 25.0],
    "speed_knots": [4.2, 16.0, 12.5, 3.0, 14.2],
    "ais_signal_gap": [True, False, False, True, False],
    "trajectory_match": [0.85, 0.20, 0.55, 0.90, 0.10]
}
ships_df = pd.DataFrame(ships_data)

# ---------------- MAIN DASHBOARD ----------------
if run_button:
    col1, col2, col3 = st.columns(3)

    # ENGINE 1
    original, mask, highlighted, contours = analyze_oil_spill(image_path)
    with col1:
        st.subheader("Engine 1: Detection")
        st.image(original, caption="Original", width='stretch', clamp=True)
    with col2:
        st.image(mask, caption="Detected Regions", width='stretch', clamp=True)
    with col3:
        st.image(highlighted, caption="Spill Highlighted", width='stretch', clamp=True, channels="BGR")

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        area_km2 = (cv2.contourArea(largest) * 100) / 1_000_000
        st.success(f"Spill detected — Estimated Area: {round(area_km2,4)} km²")
    else:
        st.warning("No spill detected in this image.")

    # ENGINE 2
    st.subheader("Engine 2: Drift Analysis")
    origin_lat, origin_lon = calculate_drift_position(
        spill_lat, spill_lon, current_speed, current_dir, wind_speed, wind_dir, 0.03,
        hours=6, direction="backward"
    )
    future_lat, future_lon = calculate_drift_position(
        spill_lat, spill_lon, current_speed, current_dir, wind_speed, wind_dir, 0.03,
        hours=12, direction="forward"
    )

    m = folium.Map(location=[spill_lat, spill_lon], zoom_start=10)
    folium.Marker([spill_lat, spill_lon], popup="Detected Spill", icon=folium.Icon(color="red")).add_to(m)
    folium.Marker([origin_lat, origin_lon], popup="Estimated Origin", icon=folium.Icon(color="orange")).add_to(m)
    folium.Marker([future_lat, future_lon], popup="Predicted Future Position", icon=folium.Icon(color="blue")).add_to(m)
    folium.PolyLine(
        [[origin_lat, origin_lon], [spill_lat, spill_lon], [future_lat, future_lon]],
        color="gray", dash_array="5"
    ).add_to(m)

    st_folium(m, width=1200, height=400)

    st.write(f"**Estimated Origin (6h prior):** ({origin_lat}, {origin_lon})")
    st.write(f"**Predicted Position (12h ahead):** ({future_lat}, {future_lon})")

    # ENGINE 3
    st.subheader("Engine 3: Vessel Attribution")
    ships_df['suspect_score'] = ships_df.apply(calculate_suspect_score, axis=1)
    ranked = ships_df.sort_values(by='suspect_score', ascending=False).reset_index(drop=True)
    st.dataframe(
        ranked[['ship_name', 'ship_type', 'distance_from_spill_km', 'ais_signal_gap', 'suspect_score']],
        width='stretch'
    )

    top = ranked.iloc[0]
    st.error(f"🚨 Top Suspect Vessel: {top['ship_name']} (Score: {top['suspect_score']})")
else:
    st.info("Set parameters in the sidebar and click 'Run ORCA Pipeline' to begin.")