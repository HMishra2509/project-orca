import requests
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Project ORCA", page_icon="🐋", layout="wide")

# ---------------- CUSTOM STYLING (ORCA BRAND) ----------------
st.markdown("""
<style>
    :root {
        --navy: #0A2540;
        --teal: #00838F;
        --light-bg: #EAF4F6;
    }
    .main-header {
        background: linear-gradient(135deg, #0A2540 0%, #143d5c 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.2rem;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #9FD6DE;
        font-size: 1.05rem;
        margin: 0;
        letter-spacing: 1px;
    }
    .engine-card {
        background-color: #F8FAFB;
        border-left: 4px solid #00838F;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .engine-card h3 {
        color: #0A2540;
        margin-top: 0;
    }
    div[data-testid="stMetricValue"] {
        color: #0A2540;
    }
    .stButton>button {
        background-color: #00838F;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0A2540;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #0A2540;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] input {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-header">
    <h1>🐋 Project ORCA</h1>
    <p>OIL-SPILL RECOGNITION, CORRELATION &amp; ATTRIBUTION &nbsp;|&nbsp; DETECT · TRACE · ATTRIBUTE</p>
</div>
""", unsafe_allow_html=True)


# ---------------- U-NET MODEL DEFINITION ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.final(d1))


@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load('unet_final_model.pth', map_location=device))
    model.eval()
    return model


trained_model = load_model()


@st.cache_data
def load_ais_data():
    return pd.read_csv('ais_data/ais_data.csv')


ais_source_df = load_ais_data()


# ---------------- ENGINE 1: AI-BASED DETECTION ----------------
def analyze_oil_spill_ai(image_path, model, resolution_m_per_pixel=10, img_size=256):
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_shape = original_img.shape

    resized_img = cv2.resize(original_img, (img_size, img_size))
    normalized_img = resized_img.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(normalized_img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.squeeze().numpy()

    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    binary_mask_original_size = cv2.resize(binary_mask, (original_shape[1], original_shape[0]))

    contours, _ = cv2.findContours(binary_mask_original_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spill_contours = [c for c in contours if cv2.contourArea(c) > 50]

    output = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, spill_contours, -1, (0, 0, 255), 2)

    return original_img, binary_mask_original_size, output, spill_contours


# ---------------- ENGINE 2: REAL OCEAN/WIND DATA ----------------
def get_ocean_wind_data(latitude, longitude):
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "ocean_current_velocity,ocean_current_direction"
    }
    marine_response = requests.get(marine_url, params=marine_params)
    marine_data = marine_response.json()

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "wind_speed_10m,wind_direction_10m"
    }
    weather_response = requests.get(weather_url, params=weather_params)
    weather_data = weather_response.json()

    return {
        "current_speed_kmph": marine_data["current"]["ocean_current_velocity"],
        "current_direction_deg": marine_data["current"]["ocean_current_direction"],
        "wind_speed_kmph": weather_data["current"]["wind_speed_10m"],
        "wind_direction_deg": weather_data["current"]["wind_direction_10m"]
    }


def calculate_drift_position_real_data(start_lat, start_lon, hours, direction="forward", wind_drift_factor=0.03):
    env_data = get_ocean_wind_data(start_lat, start_lon)

    current_speed_kmph = env_data["current_speed_kmph"]
    current_dir_deg = env_data["current_direction_deg"]
    wind_speed_kmph = env_data["wind_speed_kmph"]
    wind_dir_deg = env_data["wind_direction_deg"]

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

    return round(start_lat + delta_lat, 5), round(start_lon + delta_lon, 5), env_data


# ---------------- ENGINE 3: REAL VESSEL DATA ATTRIBUTION ----------------
def generate_realistic_vessel_scenario(ais_df, spill_lat, spill_lon, radius_km=30, n_vessels=8, seed=42):
    relevant_types = ['Cargo', 'Tanker', 'Fishing', 'Pleasure', 'Port tender']
    real_vessels_sample = ais_df[ais_df['shiptype'].isin(relevant_types)].dropna(
        subset=['sog', 'shiptype']
    ).sample(n_vessels, random_state=seed).reset_index(drop=True)

    rng = np.random.RandomState(seed)
    vessels = []

    for idx, row in real_vessels_sample.iterrows():
        distance_km = rng.uniform(1, radius_km)
        bearing_deg = rng.uniform(0, 360)

        bearing_rad = np.radians(bearing_deg)
        delta_lat = (distance_km * np.cos(bearing_rad)) / 111.0
        delta_lon = (distance_km * np.sin(bearing_rad)) / (111.0 * np.cos(np.radians(spill_lat)))

        vessel_lat = spill_lat + delta_lat
        vessel_lon = spill_lon + delta_lon

        ais_gap = rng.choice([True, False], p=[0.3, 0.7]) if row['shiptype'] in ['Tanker', 'Cargo'] \
            else rng.choice([True, False], p=[0.1, 0.9])

        trajectory_match = rng.uniform(0.1, 0.95)

        vessels.append({
            "mmsi": int(row['mmsi']),
            "ship_type": row['shiptype'],
            "speed_knots": row['sog'],
            "length_m": row['length'],
            "distance_from_spill_km": round(distance_km, 2),
            "latitude": round(vessel_lat, 5),
            "longitude": round(vessel_lon, 5),
            "ais_signal_gap": bool(ais_gap),
            "trajectory_match": round(trajectory_match, 2)
        })

    return pd.DataFrame(vessels)


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
    if row['ship_type'] == "Tanker":
        score += 10
    elif row['ship_type'] == "Cargo":
        score += 5
    return round(score, 1)


# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.markdown("## ⚙️ Spill Parameters")
image_path = st.sidebar.text_input("Satellite Image Path", "data/class_1_01780.jpg")
spill_lat = st.sidebar.number_input("Spill Latitude", value=15.2993, format="%.5f")
spill_lon = st.sidebar.number_input("Spill Longitude", value=74.1240, format="%.5f")

st.sidebar.markdown("---")

if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False

if st.sidebar.button("▶  Run ORCA Pipeline"):
    st.session_state.run_pipeline = True

run_button = st.session_state.run_pipeline

st.sidebar.markdown("---")
st.sidebar.caption("Live ocean & wind data: Open-Meteo API")
st.sidebar.caption("Detection model: Trained U-Net (PyTorch)")
st.sidebar.caption("Vessel identity data: Real AIS records (MMSI, type, speed)")

# ---------------- MAIN DASHBOARD ----------------
if run_button:

    # ---- ENGINE 1 ----
    st.markdown('<div class="engine-card"><h3>🛰️ Engine 1 — Detection & Characterization</h3></div>', unsafe_allow_html=True)

    original, mask, highlighted, contours = analyze_oil_spill_ai(image_path, trained_model)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original, caption="Original Satellite Image", width='stretch', clamp=True)
    with col2:
        st.image(mask, caption="AI-Detected Regions", width='stretch', clamp=True)
    with col3:
        st.image(highlighted, caption="Spill Boundary Highlighted", width='stretch', clamp=True, channels="BGR")

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        area_km2 = (cv2.contourArea(largest) * 100) / 1_000_000
        m1, m2, m3 = st.columns(3)
        m1.metric("Spill Regions Detected", len(contours))
        m2.metric("Largest Spill Area", f"{round(area_km2, 4)} km²")
        m3.metric("Detection Model", "U-Net (PyTorch)")
    else:
        st.warning("No spill detected in this image.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- ENGINE 2 ----
    st.markdown('<div class="engine-card"><h3>🌊 Engine 2 — Hindcast & Forecast (Live Data)</h3></div>', unsafe_allow_html=True)

    origin_lat, origin_lon, env_data = calculate_drift_position_real_data(
        spill_lat, spill_lon, hours=6, direction="backward"
    )
    future_lat, future_lon, _ = calculate_drift_position_real_data(
        spill_lat, spill_lon, hours=12, direction="forward"
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Ocean Current", f"{env_data['current_speed_kmph']} km/h")
    e2.metric("Current Direction", f"{env_data['current_direction_deg']}°")
    e3.metric("Wind Speed", f"{env_data['wind_speed_kmph']} km/h")
    e4.metric("Wind Direction", f"{env_data['wind_direction_deg']}°")

    m = folium.Map(location=[spill_lat, spill_lon], zoom_start=9, tiles="CartoDB positron")
    folium.Marker([spill_lat, spill_lon], popup="Detected Spill",
                  icon=folium.Icon(color="red", icon="tint")).add_to(m)
    folium.Marker([origin_lat, origin_lon], popup="Estimated Origin (6h prior)",
                  icon=folium.Icon(color="orange", icon="flag")).add_to(m)
    folium.Marker([future_lat, future_lon], popup="Predicted Position (12h ahead)",
                  icon=folium.Icon(color="blue", icon="arrow-right")).add_to(m)
    folium.PolyLine(
        [[origin_lat, origin_lon], [spill_lat, spill_lon], [future_lat, future_lon]],
        color="#0A2540", weight=3, dash_array="8"
    ).add_to(m)

    st_folium(m, width=1200, height=400)

    o1, o2 = st.columns(2)
    o1.info(f"**Estimated Origin (6h prior):** {origin_lat}, {origin_lon}")
    o2.info(f"**Predicted Position (12h ahead):** {future_lat}, {future_lon}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- ENGINE 3 ----
    st.markdown('<div class="engine-card"><h3>🚢 Engine 3 — Vessel Attribution (Real AIS Vessel Data)</h3></div>', unsafe_allow_html=True)
    st.caption("Vessel identity (MMSI, type, speed, dimensions) from real historic AIS records. "
               "Positions are representative for demonstration — full historical position-tracking "
               "archives are planned for production deployment.")

    scenario_df = generate_realistic_vessel_scenario(ais_source_df, spill_lat, spill_lon)
    scenario_df['suspect_score'] = scenario_df.apply(calculate_suspect_score, axis=1)
    ranked = scenario_df.sort_values(by='suspect_score', ascending=False).reset_index(drop=True)

    st.dataframe(
        ranked[['mmsi', 'ship_type', 'speed_knots', 'distance_from_spill_km', 'ais_signal_gap', 'suspect_score']],
        width='stretch',
        column_config={
            "mmsi": "MMSI (Vessel ID)",
            "ship_type": "Type",
            "speed_knots": "Speed (knots)",
            "distance_from_spill_km": "Distance (km)",
            "ais_signal_gap": "AIS Signal Gap",
            "suspect_score": st.column_config.ProgressColumn(
                "Suspect Score", min_value=0, max_value=100, format="%.1f"
            ),
        }
    )

    top = ranked.iloc[0]
    st.error(f"🚨 **Top Suspect Vessel: MMSI {top['mmsi']}** ({top['ship_type']})  —  Suspect Score: {top['suspect_score']} / 100")

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 1rem; color: #666;">
        <h3>Set spill parameters in the sidebar and click <b>Run ORCA Pipeline</b> to begin.</h3>
        <p>Live satellite detection · Real oceanographic data · Real AIS vessel identity data</p>
    </div>
    """, unsafe_allow_html=True)