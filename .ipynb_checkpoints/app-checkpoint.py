import os
import requests
import streamlit as st
import cv2
import time
import glob
import random
import numpy as np
import pandas as pd
import math
import altair as alt
import torch
import base64
import torch.nn as nn
import folium
import asyncio
import websockets
import json
from streamlit_folium import st_folium
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
os.makedirs("uploaded_images", exist_ok=True)

supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY")
)
def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
st.set_page_config(page_title="Project ORCA", page_icon="🐋", layout="wide")

# ---------------- CUSTOM STYLING (ORCA BRAND) ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* header ko hide mat karo, bas transparent/borderless bana do */
header {
    background: transparent !important;
    box-shadow: none !important;
}

/* Sidebar toggle button ko force visible + styled rakho */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    color: var(--signal) !important;
    z-index: 999999 !important;
}

    :root {
        --void: #05070A;
        --panel: #0D131A;
        --panel-alt: #121A23;
        --signal: #00FF9C;
        --amber: #FFB020;
        --grid: rgba(0, 255, 156, 0.06);
        --text: #DCEDE7;
        --text-dim: #6B8A82;
    }

    .stApp {
        background-color: var(--void);
        background-image:
            linear-gradient(var(--grid) 1px, transparent 1px),
            linear-gradient(90deg, var(--grid) 1px, transparent 1px);
        background-size: 32px 32px;
    }

    * { font-family: 'Inter', sans-serif; }
    .mono, div[data-testid="stMetricValue"], .stDataFrame, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    

    /* ---- HEADER ---- */
    .main-header {
        background: linear-gradient(160deg, #0D131A 0%, #0A2540 100%);
        border: 1px solid rgba(0,255,156,0.25);
        padding: 2rem 2.2rem;
        border-radius: 6px;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: "";
        position: absolute; top: 0; left: -100%;
        width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,255,156,0.06), transparent);
        animation: sweep 5s ease-in-out infinite;
    }
    @keyframes sweep {
        0% { left: -60%; } 100% { left: 120%; }
    }
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: white;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-family: 'JetBrains Mono', monospace;
        color: var(--signal);
        font-size: 0.85rem;
        letter-spacing: 2px;
        margin: 0;
    }
    .live-dot {
        display: inline-block; width: 8px; height: 8px;
        background: var(--signal); border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px var(--signal);
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
    }

    /* ---- ENGINE CARDS (targeting frame) ---- */
    .engine-card {
        background: var(--panel);
        border: 1px solid rgba(0,255,156,0.15);
        border-radius: 4px;
        padding: 1.1rem 1.6rem;
        margin-bottom: 1.1rem;
        position: relative;
    }
    .engine-card::before, .engine-card::after {
        content: ""; position: absolute; width: 14px; height: 14px;
        border-color: var(--signal); border-style: solid; border-width: 0;
    }
    .engine-card::before {
        top: -1px; left: -1px;
        border-top-width: 2px; border-left-width: 2px;
    }
    .engine-card::after {
        bottom: -1px; right: -1px;
        border-bottom-width: 2px; border-right-width: 2px;
    }
    .engine-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
        margin: 0;
        font-weight: 600;
        font-size: 1.15rem;
    }

    /* ---- METRICS ---- */
    div[data-testid="stMetric"] {
        background: var(--panel-alt);
        border: 1px solid rgba(0,255,156,0.12);
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
    }
    div[data-testid="stMetricValue"] {
        color: var(--signal) !important;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(0,255,156,0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-dim) !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 1px;
    }

    /* ---- BODY TEXT / MARKDOWN ---- */
    .stMarkdown, .stCaption, p, span, label {
        color: var(--text) !important;
    }

    /* ---- BUTTONS ---- */
    .stButton>button {
        background: transparent;
        color: var(--signal);
        border: 1px solid var(--signal);
        border-radius: 4px;
        padding: 0.6rem 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: rgba(0,255,156,0.1);
        box-shadow: 0 0 16px rgba(0,255,156,0.3);
    }

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {
        background: #080B10;
        border-right: 1px solid rgba(0,255,156,0.15);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select {
        background-color: var(--panel-alt) !important;
        color: var(--signal) !important;
        border: 1px solid rgba(0,255,156,0.2) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-family: 'Space Grotesk', sans-serif;
        border-bottom: 1px solid var(--signal);
        padding-bottom: 0.5rem;
    }

    /* ---- TABS ---- */
    button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 1px;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: var(--signal) !important;
    }

    /* ---- TABLE ---- */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(0,255,156,0.15);
        border-radius: 4px;
    }

    /* ---- ALERTS ---- */
    div[data-testid="stAlert"] {
        border-radius: 4px;
        border-left: 3px solid var(--amber);
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)
# ---------------- BOOT SEQUENCE (full-screen image, plays once per session) ----------------
if "booted" not in st.session_state:
    st.session_state.booted = False

if not st.session_state.booted:
    boot_placeholder = st.empty()
    boot_img_b64 = get_image_base64("assets/loading_ship.png")
    BOOT_DURATION = 2.2  # seconds — shortened for repeated demo runs

    boot_placeholder.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ visibility: hidden; }}

        @keyframes fillBar {{
            from {{ width: 0%; }}
            to {{ width: 100%; }}
        }}
        #boot-bar-fill {{
            animation: fillBar {BOOT_DURATION}s linear forwards;
        }}
        #boot-msg::after {{
            content: "INITIALIZING ORCA SYSTEM...";
            animation: msgSwap {BOOT_DURATION}s steps(1) forwards;
        }}
        @keyframes msgSwap {{
            0%   {{ content: "INITIALIZING ORCA SYSTEM..."; }}
            15%  {{ content: "CONNECTING TO DISASTER RESPONSE NETWORK..."; }}
            30%  {{ content: "LOADING AI DETECTION MODEL (U-NET)..."; }}
            50%  {{ content: "SYNCING LIVE OCEAN & WIND DATA FEED..."; }}
            68%  {{ content: "LOADING AIS VESSEL REGISTRY..."; }}
            85%  {{ content: "PREPARING RESPONSE RECOMMENDATIONS..."; }}
            97%  {{ content: "SYSTEM READY."; }}
        }}
    </style>
    <div style="position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:99999;
                background: linear-gradient(rgba(5,7,10,0.45), rgba(5,7,10,0.9)),
                            url('data:image/jpeg;base64,{boot_img_b64}');
                background-size:cover; background-position:center;
                display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
                padding-bottom:9vh;">
        <div style="font-family:'Space Grotesk', sans-serif; color:white; font-size:2rem; font-weight:700;
                    letter-spacing:3px; margin-bottom:0.8rem; text-shadow:0 0 20px rgba(0,0,0,0.8);">
            PROJECT ORCA
        </div>
        <div id="boot-msg" style="font-family:'JetBrains Mono', monospace; color:#00FF9C; font-size:0.9rem;
                    letter-spacing:2px; margin-bottom:1.2rem; text-shadow:0 0 10px rgba(0,255,156,0.5);">
        </div>
        <div style="width:380px; max-width:80%; background:rgba(255,255,255,0.08);
                    border:1px solid rgba(0,255,156,0.4); border-radius:20px; overflow:hidden; height:12px;">
            <div id="boot-bar-fill" style="height:100%; background:linear-gradient(90deg, #00FF9C, #007A85);
                        box-shadow:0 0 12px #00FF9C; width:0%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(BOOT_DURATION)
    st.session_state.booted = True
    boot_placeholder.empty()
    st.rerun()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"ORC-{random.randint(1000,9999)}-{random.randint(100,999)}"

import datetime
sync_time = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
# ---------------- HEADER ----------------
header_html = (
    f'<div class="main-header" style="display:flex; justify-content:space-between; align-items:center; position:relative;">'
    f'<div style="position:absolute; top:-1px; left:-1px; width:18px; height:18px; border-top:2px solid #00FF9C; border-left:2px solid #00FF9C;"></div>'
    f'<div style="position:absolute; bottom:-1px; right:-1px; width:18px; height:18px; border-bottom:2px solid #00FF9C; border-right:2px solid #00FF9C;"></div>'
    f'<div>'
    f'<h1>&#128011; Project ORCA</h1>'
    f'<p><span class="live-dot"></span>MARINE DISASTER RESPONSE SYSTEM &nbsp;//&nbsp; DETECT &middot; TRACE &middot; ATTRIBUTE &nbsp;//&nbsp; NTRO SIH26143</p>'
    f'</div>'
    f'<div style="text-align:right; font-family:\'JetBrains Mono\', monospace; font-size:0.7rem; color:#6B8A82; line-height:1.8;">'
    f'<div>SYSTEM STATUS: <span style="color:#00FF9C; font-weight:700;">ONLINE</span></div>'
    f'<div>SESSION: <span style="color:#00FF9C; font-weight:700;">{st.session_state.session_id}</span></div>'
    f'<div>LAST SYNC: <span style="color:#00FF9C; font-weight:700;">{sync_time}</span></div>'
    f'<div>MODE: <span style="color:#FFB020; font-weight:700;">DISASTER RESPONSE</span></div>'
    f'</div>'
    f'</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

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

def get_vessel_image_path(ship_type):
    mapping = {
        "Tanker": "ship_images/tanker.jpg",
        "Cargo": "ship_images/cargo.jpg",
        "Fishing": "ship_images/fishing.jpg",
        "Pleasure": "ship_images/pleasure.jpg",
        "Port tender": "ship_images/port_tender.jpg"
    }
    return mapping.get(ship_type, "ship_images/cargo.jpg")


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

def characterize_spill_shape_and_age(contours):
    if len(contours) == 0:
        return None

    num_fragments = len(contours)
    largest = max(contours, key=cv2.contourArea)

    # ---- SHAPE: aspect ratio via minimum-area bounding rectangle ----
    rect = cv2.minAreaRect(largest)
    rect_w, rect_h = rect[1]
    aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 1.0

    if aspect_ratio >= 3.0:
        shape_label = "Elongated / Linear"
        shape_reasoning = "Consistent with a moving-source discharge — e.g. a vessel leaking while underway, leaving a trailing streak."
    elif aspect_ratio >= 1.5:
        shape_label = "Moderately Elongated"
        shape_reasoning = "Suggests partial drift influence acting on an initially more compact release."
    else:
        shape_label = "Compact / Circular"
        shape_reasoning = "Consistent with a point-source release — e.g. a static leak or stationary discharge event."

    # ---- SHAPE COMPLEXITY (isoperimetric-style index) ----
    perimeter = cv2.arcLength(largest, True)
    area = cv2.contourArea(largest)
    complexity_index = (perimeter ** 2) / (4 * math.pi * area) if area > 0 else 1.0

    # ---- AGE / WEATHERING HEURISTIC ----
    if num_fragments <= 1 and complexity_index < 2.5:
        age_label = "Fresh (likely < 3 hours)"
    elif num_fragments <= 3 and complexity_index < 4.5:
        age_label = "Recent (likely 3–12 hours)"
    else:
        age_label = "Weathered (likely 12+ hours)"

    return {
        "aspect_ratio": round(aspect_ratio, 2),
        "shape_label": shape_label,
        "shape_reasoning": shape_reasoning,
        "complexity_index": round(complexity_index, 2),
        "num_fragments": num_fragments,
        "age_label": age_label
    }
# ---------------- ENGINE 2: REAL OCEAN/WIND DATA ----------------
@st.cache_data(ttl=300)
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

# ---------------- IMPACT ZONE REFERENCE DATA (Indian coast) ----------------
INDIA_PORTS = {
    "Mumbai Port / JNPT": (18.95, 72.95),
    "Kandla Port (Gujarat)": (23.03, 70.22),
    "Paradip Port (Odisha)": (20.27, 86.67),
    "Visakhapatnam Port": (17.68, 83.28),
    "Chennai Port": (13.10, 80.30),
    "Kochi Port": (9.97, 76.24),
    "New Mangalore Port": (12.92, 74.80),
    "Tuticorin Port": (8.76, 78.21),
    "Haldia Port (West Bengal)": (22.02, 88.06),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_port(spill_lat, spill_lon):
    nearest_name, nearest_dist = None, float("inf")
    for name, (plat, plon) in INDIA_PORTS.items():
        dist = haversine_km(spill_lat, spill_lon, plat, plon)
        if dist < nearest_dist:
            nearest_name, nearest_dist = name, dist
    return nearest_name, round(nearest_dist, 1)


# ---------------- ENGINE 3: REAL VESSEL DATA ATTRIBUTION ----------------
def generate_realistic_vessel_scenario(ais_df, spill_lat, spill_lon, radius_km=30, n_vessels=8, seed=None):
    if seed is None:
        seed = int((spill_lat * 1000 + spill_lon * 1000) % 100000)
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


def generate_port_overview_vessels(port_lat, port_lon):
    """Small, deterministic vessel set shown around a non-active port for
    the national overview map — not part of any attribution/scoring."""
    return generate_realistic_vessel_scenario(ais_source_df, port_lat, port_lon, radius_km=15, n_vessels=4)


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


# ---------------- DATABASE SAVE ----------------
def get_risk_tier(score):
    if score >= 70:
        return "🔴 HIGH RISK", "red"
    elif score >= 40:
        return "🟡 MEDIUM RISK", "orange"
    else:
        return "🟢 LOW RISK", "green"


def check_repeat_offender(mmsi):
    try:
        result = supabase.table("vessels").select("id").eq("ship_name", f"MMSI {mmsi}").execute()
        return len(result.data)
    except Exception:
        return 0
def play_radar_scan():
    radar_placeholder = st.empty()
    scan_messages = [
        "SCANNING VESSEL REGISTRY...",
        "CROSS-REFERENCING AIS DATABASE...",
        "CALCULATING SUSPECT SCORES...",
        "LOCKING TARGET..."
    ]
    for msg in scan_messages:
        radar_placeholder.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; padding:2.5rem 0;">
            <div style="position:relative; width:140px; height:140px; border-radius:50%;
                        border:1px solid rgba(0,255,156,0.25); display:flex; align-items:center; justify-content:center;">
                <div style="position:absolute; width:100%; height:100%; border-radius:50%;
                            background:conic-gradient(rgba(0,255,156,0.6), transparent 40%);
                            animation: radar-spin 1.2s linear infinite;"></div>
                <div style="position:absolute; width:70%; height:70%; border-radius:50%; border:1px solid rgba(0,255,156,0.15);"></div>
                <div style="position:absolute; width:40%; height:40%; border-radius:50%; border:1px solid rgba(0,255,156,0.15);"></div>
                <div style="position:relative; width:8px; height:8px; background:#00FF9C; border-radius:50%; box-shadow:0 0 10px #00FF9C;"></div>
            </div>
            <div style="font-family:'JetBrains Mono', monospace; color:#00FF9C; font-size:0.85rem; letter-spacing:2px; margin-top:1.2rem;">
                {msg}
            </div>
        </div>
        <style>
            @keyframes radar-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        </style>
        """, unsafe_allow_html=True)
        time.sleep(0.6)

    radar_placeholder.empty()
def render_mission_tracker(placeholder, current_stage):
    stages = ["SCAN", "DETECT", "TRACE", "ATTRIBUTE", "COMPLETE"]
    stage_index = stages.index(current_stage)

    items_html = ""
    for i, stage in enumerate(stages):
        if i < stage_index:
            color, icon, opacity = "#00FF9C", "&#10003;", "1"
        elif i == stage_index:
            color, icon, opacity = "#00FF9C", "&#9679;", "1"
        else:
            color, icon, opacity = "#3A4A52", "&#9675;", "0.5"

        glow = "text-shadow: 0 0 10px #00FF9C;" if i <= stage_index else ""
        items_html += (
            f'<div style="display:flex; flex-direction:column; align-items:center; opacity:{opacity};">'
            f'<div style="font-size:1.5rem; color:{color}; {glow}">{icon}</div>'
            f'<div style="font-family:\'JetBrains Mono\', monospace; font-size:0.7rem; color:{color}; letter-spacing:1px; margin-top:0.3rem;">{stage}</div>'
            f'</div>'
        )
        if i < len(stages) - 1:
            line_color = "#00FF9C" if i < stage_index else "#3A4A52"
            items_html += f'<div style="flex:1; height:2px; background:{line_color}; margin-top:0.75rem;"></div>'

    html = (
        f'<div style="display:flex; align-items:flex-start; justify-content:space-between; '
        f'background:#0D131A; border:1px solid rgba(0,255,156,0.15); border-radius:8px; '
        f'padding:1.2rem 2rem; margin-bottom:1.5rem;">'
        f'{items_html}'
        f'</div>'
    )
    placeholder.markdown(html, unsafe_allow_html=True)
        
def save_pipeline_results(spill_lat, spill_lon, area_km2, source_image, ranked_vessels_df):
    try:
        spill_insert = supabase.table("spills").insert({
            "latitude": spill_lat,
            "longitude": spill_lon,
            "area_sq_km": area_km2,
            "source_image": source_image
        }).execute()

        spill_id = spill_insert.data[0]["id"]

        for _, row in ranked_vessels_df.iterrows():
            vessel_insert = supabase.table("vessels").insert({
                "ship_name": f"MMSI {row['mmsi']}",
                "ship_type": row['ship_type'],
                "latitude": row['latitude'],
                "longitude": row['longitude'],
                "speed_knots": row['speed_knots'],
                "ais_signal_gap": bool(row['ais_signal_gap'])
            }).execute()

            vessel_id = vessel_insert.data[0]["id"]

            supabase.table("attributions").insert({
                "spill_id": spill_id,
                "vessel_id": vessel_id,
                "distance_km": row['distance_from_spill_km'],
                "trajectory_match": row['trajectory_match'],
                "suspect_score": row['suspect_score']
            }).execute()

        return True, spill_id

    except Exception as e:
        return False, str(e)


# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.markdown("## ⚙️ Spill Parameters")
all_images = glob.glob("data/*.jpg") + glob.glob("data/*.png")

if "current_image" not in st.session_state:
    st.session_state.current_image = random.choice(all_images) if all_images else None

if st.sidebar.button("🔀 New Random Image"):
    st.session_state.current_image = random.choice(all_images)

uploaded_file = st.sidebar.file_uploader(
    "Or upload your own satellite image (optional)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    saved_path = f"uploaded_images/{uploaded_file.name}"
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    image_path = saved_path
else:
    image_path = st.session_state.current_image

st.sidebar.caption(f"Current image: `{image_path}`")
india_locations = dict(INDIA_PORTS)
india_locations["Custom / Manual entry"] = None

selected_location = st.sidebar.selectbox("Spill Location Preset (Major Indian Ports)", list(india_locations.keys()))

if india_locations[selected_location] is not None:
    default_lat, default_lon = india_locations[selected_location]
else:
    default_lat, default_lon = 15.2993, 74.1240

spill_lat = st.sidebar.number_input("Spill Latitude", value=default_lat, format="%.5f")
spill_lon = st.sidebar.number_input("Spill Longitude", value=default_lon, format="%.5f")

st.sidebar.markdown("---")
map_style = st.sidebar.radio("Map Style", ["Standard (OpenStreetMap)", "Satellite (Esri)"], horizontal=True)

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

    # ==================================================================
    # ALL CALCULATIONS HAPPEN FIRST (before any rendering) so the top
    # summary strip has everything it needs immediately.
    # ==================================================================

    # ---- ENGINE 1 CALC ----
    original, mask, highlighted, contours = analyze_oil_spill_ai(image_path, trained_model)
    area_km2 = None
    characterization = None
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        area_km2 = (cv2.contourArea(largest) * 100) / 1_000_000
        characterization = characterize_spill_shape_and_age(contours)

    # ---- ENGINE 3 CALC (vessel ranking) ----
    scenario_df = generate_realistic_vessel_scenario(ais_source_df, spill_lat, spill_lon)
    scenario_df['suspect_score'] = scenario_df.apply(calculate_suspect_score, axis=1)
    ranked = scenario_df.sort_values(by='suspect_score', ascending=False).reset_index(drop=True)
    ranked['risk_label'] = ranked['suspect_score'].apply(lambda s: get_risk_tier(s)[0])
    ranked['risk_color'] = ranked['suspect_score'].apply(lambda s: get_risk_tier(s)[1])
    ranked['prior_incidents'] = ranked['mmsi'].apply(check_repeat_offender)
    top = ranked.iloc[0]

    # ---- ENGINE 2 CALC (ocean/wind + drift trajectory) ----
    env_data = get_ocean_wind_data(spill_lat, spill_lon)

    hindcast_hours = [12, 9, 6, 3]
    forecast_hours = [3, 6, 9, 12]
    trajectory_points = []

    for h in hindcast_hours:
        lat, lon, _ = calculate_drift_position_real_data(spill_lat, spill_lon, hours=h, direction="backward")
        trajectory_points.append((lat, lon, f"-{h}h"))
    trajectory_points.append((spill_lat, spill_lon, "Detected Spill (T+0)"))
    for h in forecast_hours:
        lat, lon, _ = calculate_drift_position_real_data(spill_lat, spill_lon, hours=h, direction="forward")
        trajectory_points.append((lat, lon, f"+{h}h"))

    origin_lat, origin_lon = trajectory_points[0][0], trajectory_points[0][1]
    future_lat, future_lon = trajectory_points[-1][0], trajectory_points[-1][1]

    # ---- IMPACT ZONE CALC ----
    nearest_port_name, nearest_port_dist_km = find_nearest_port(spill_lat, spill_lon)
    # Impact radius approximated as the furthest drift point from the spill origin
    impact_radius_km = max(
        haversine_km(spill_lat, spill_lon, lat, lon) for lat, lon, _ in trajectory_points
    )

    # ---- URGENCY CALC ----
    spill_detected = len(contours) > 0
    urgency = "IMMEDIATE" if (spill_detected and (area_km2 > 0.5 or top['suspect_score'] >= 70)) else \
              "PRIORITY" if (spill_detected and area_km2 > 0.1) else "MONITOR"
    urgency_color = {"IMMEDIATE": "#EF4444", "PRIORITY": "#FFB020", "MONITOR": "#00FF9C"}[urgency]

    # ==================================================================
    # TOP SUMMARY STRIP — always visible, no scrolling needed
    # ==================================================================
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Spill Detected", "Yes" if spill_detected else "No")
    s2.metric("Spill Area", f"{round(area_km2, 4)} km²" if area_km2 else "—")
    s3.metric("Top Suspect Score", f"{top['suspect_score']}/100" if spill_detected else "N/A")
    s4.markdown(f"""
    <div style="background:#0D131A; border:1px solid {urgency_color}; border-radius:4px; padding:0.9rem 1.1rem; text-align:center;">
        <div style="color:#6B8A82; font-family:'JetBrains Mono', monospace; text-transform:uppercase; font-size:0.7rem; letter-spacing:1px;">Urgency</div>
        <div style="color:{urgency_color}; font-weight:700; font-size:1.4rem; text-shadow:0 0 12px {urgency_color}66;">{urgency}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================================================================
    # TABS — each engine gets its own section, no more one long scroll
    # ==================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["🛰️ Detection", "🌊 Drift Analysis", "🚢 Attribution", "🚨 Response"])

    # ---------------- TAB 1: DETECTION ----------------
    with tab1:
        st.markdown('<div class="engine-card"><h3>🛰️ Engine 1 — Detection & Characterization</h3></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="Original Satellite Image", width='stretch', clamp=True)
        with col2:
            st.image(mask, caption="AI-Detected Regions", width='stretch', clamp=True)
        with col3:
            st.image(highlighted, caption="Spill Boundary Highlighted", width='stretch', clamp=True, channels="BGR")

        if len(contours) > 0:
            m1, m2, m3 = st.columns(3)
            m1.metric("Spill Regions Detected", len(contours))
            m2.metric("Largest Spill Area", f"{round(area_km2, 4)} km²")
            m3.metric("Detection Model", "U-Net (PyTorch)")

            c1, c2 = st.columns(2)
            c1.metric("Estimated Shape", characterization["shape_label"])
            c2.metric("Estimated Age", characterization["age_label"])

            with st.expander("ℹ️ How shape & age were estimated"):
                st.caption(f"**Aspect ratio:** {characterization['aspect_ratio']} — {characterization['shape_reasoning']}")
                st.caption(f"**Shape complexity index:** {characterization['complexity_index']} (1.0 = perfect circle; higher = more irregular/fragmented)")
                st.caption(f"**Fragments detected:** {characterization['num_fragments']}")
                st.caption("Age is a heuristic estimate based on spill fragmentation and boundary irregularity, not a precise physical measurement.")
        else:
            st.warning("No spill detected in this image.")

    # ---------------- TAB 2: DRIFT ANALYSIS ----------------
    with tab2:
        st.markdown('<div class="engine-card"><h3>🌊 Engine 2 — Hindcast & Forecast (Live Data)</h3></div>', unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Ocean Current", f"{env_data['current_speed_kmph']} km/h")
        e2.metric("Current Direction", f"{env_data['current_direction_deg']}°")
        e3.metric("Wind Speed", f"{env_data['wind_speed_kmph']} km/h")
        e4.metric("Wind Direction", f"{env_data['wind_direction_deg']}°")

        st.markdown("<br>", unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        i1.metric("Nearest Port / Facility", nearest_port_name)
        i2.metric("Distance to Port", f"{nearest_port_dist_km} km")
        i3.metric("Impact Zone Radius", f"{round(impact_radius_km, 1)} km")

        if map_style == "Satellite (Esri)":
            m = folium.Map(location=[spill_lat, spill_lon], zoom_start=9, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri, Maxar, Earthstar Geographics")
        else:
            m = folium.Map(location=[spill_lat, spill_lon], zoom_start=9, tiles="OpenStreetMap")

        for lat, lon, label in trajectory_points:
            if label == "Detected Spill (T+0)":
                folium.Marker([lat, lon], popup=label, icon=folium.Icon(color="darkred", icon="tint")).add_to(m)
            elif label.startswith("-"):
                folium.CircleMarker([lat, lon], radius=5, popup=f"Hindcast: {label}", color="#FFB020", fill=True, fill_opacity=0.9).add_to(m)
            else:
                folium.CircleMarker([lat, lon], radius=5, popup=f"Forecast: {label}", color="#0074D9", fill=True, fill_opacity=0.9).add_to(m)

        folium.PolyLine([[p[0], p[1]] for p in trajectory_points], color="#0A2540", weight=3, dash_array="6").add_to(m)

        # ---- Impact zone circle (approximate drift-affected radius) ----
        folium.Circle(
            location=[spill_lat, spill_lon],
            radius=impact_radius_km * 1000,
            color="#FFB020",
            weight=1.5,
            fill=True,
            fill_color="#FFB020",
            fill_opacity=0.08,
            popup=f"Approx. impact zone: {round(impact_radius_km, 1)} km radius"
        ).add_to(m)

        # ---- Nearest port marker ----
        nearest_port_coords = INDIA_PORTS[nearest_port_name]
        folium.Marker(
            list(nearest_port_coords),
            popup=f"{nearest_port_name} — {nearest_port_dist_km} km from spill",
            icon=folium.Icon(color="blue", icon="anchor", prefix="fa")
        ).add_to(m)

        for _, v in ranked.iterrows():
            folium.Marker(
                [v['latitude'], v['longitude']],
                popup=f"MMSI {v['mmsi']} — {v['ship_type']} — {v['risk_label']} ({v['suspect_score']}/100)",
                icon=folium.Icon(color=v['risk_color'], icon="ship", prefix="fa")
            ).add_to(m)

        # ---- NATIONAL PORT OVERVIEW: all 9 ports + their own live vessel traffic ----
        active_port_name = selected_location if selected_location in INDIA_PORTS else nearest_port_name

        for port_name, (plat, plon) in INDIA_PORTS.items():
            if port_name == active_port_name:
                # Highlight the active/selected region distinctly from the rest
                folium.Marker(
                    [plat, plon],
                    popup=f"⭐ ACTIVE REGION: {port_name}",
                    icon=folium.Icon(color="red", icon="star", prefix="fa")
                ).add_to(m)
                folium.Circle(
                    location=[plat, plon],
                    radius=25000,
                    color="#EF4444",
                    weight=2,
                    fill=False,
                    dash_array="4",
                    popup=f"Active monitoring region: {port_name}"
                ).add_to(m)
            else:
                # Background national traffic — not attributed to the active spill
                folium.Marker(
                    [plat, plon],
                    popup=f"{port_name} (reference port)",
                    icon=folium.Icon(color="lightgray", icon="anchor", prefix="fa")
                ).add_to(m)

                overview_vessels = generate_port_overview_vessels(plat, plon)
                for _, ov in overview_vessels.iterrows():
                    folium.CircleMarker(
                        [ov['latitude'], ov['longitude']],
                        radius=4,
                        color="#3A4A52",
                        fill=True,
                        fill_color="#6B8A82",
                        fill_opacity=0.7,
                        popup=f"MMSI {ov['mmsi']} — {ov['ship_type']} (background traffic, {port_name})"
                    ).add_to(m)

        st_folium(m, width=1200, height=550, returned_objects=[])

        st.caption("🔴 Red star = active monitored region &nbsp;|&nbsp; ⚓ Grey anchors = other national ports (reference traffic, not attributed)")

        o1, o2 = st.columns(2)
        o1.info(f"**Estimated Origin (12h prior):** {origin_lat}, {origin_lon}")
        o2.info(f"**Predicted Position (12h ahead):** {future_lat}, {future_lon}")

    # ---------------- TAB 3: ATTRIBUTION ----------------
    with tab3:
        if not spill_detected:
            st.markdown('<div class="engine-card"><h3>🚢 Engine 3 — Vessel Attribution (Real AIS Vessel Data)</h3></div>', unsafe_allow_html=True)
            st.info("No spill was detected in this image, so there is nothing to attribute. Vessel attribution only runs against a confirmed spill detection from Engine 1.")
            st.session_state.show_engine3_details = False

        if "show_engine3_details" not in st.session_state:
            st.session_state.show_engine3_details = False

        if spill_detected and not st.session_state.show_engine3_details:
            st.markdown('<div class="engine-card"><h3>🚢 Engine 3 — Vessel Attribution (Real AIS Vessel Data)</h3></div>', unsafe_allow_html=True)
            st.caption("Vessel identity (MMSI, type, speed, dimensions) from real historic AIS records. "
                       "Positions are representative for demonstration.")
            play_radar_scan()

            img_base64 = get_image_base64(get_vessel_image_path(top['ship_type']))

            top_card_html = (
                f'<div style="background:#0D131A; border:1px solid rgba(0,255,156,0.2); border-radius:8px; '
                f'padding:1.2rem; margin-bottom:1rem; position:relative; display:flex; gap:1.5rem; align-items:center; flex-wrap:wrap;">'
                f'<div style="position:absolute; top:-1px; left:-1px; width:16px; height:16px; border-top:2px solid #00FF9C; border-left:2px solid #00FF9C;"></div>'
                f'<div style="position:absolute; bottom:-1px; right:-1px; width:16px; height:16px; border-bottom:2px solid #00FF9C; border-right:2px solid #00FF9C;"></div>'
                f'<div style="flex-shrink:0; width:180px;">'
                f'<img src="data:image/jpeg;base64,{img_base64}" style="width:100%; height:130px; object-fit:cover; border-radius:6px; border:1px solid rgba(0,255,156,0.15);">'
                f'<div style="color:#3A4A52; font-size:0.6rem; font-family:\'JetBrains Mono\', monospace; margin-top:0.4rem; text-align:center;">TYPE ILLUSTRATION — NOT ACTUAL VESSEL PHOTO</div>'
                f'</div>'
                f'<div style="flex:1; min-width:200px;">'
                f'<div style="font-family:\'JetBrains Mono\', monospace; color:#FFB020; font-size:0.75rem; letter-spacing:2px; margin-bottom:0.5rem;">'
                f'&#128680; TOP SUSPECT VESSEL'
                f'</div>'
                f'<div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:1rem;">'
                f'<div>'
                f'<div style="font-family:\'Space Grotesk\', sans-serif; color:#DCEDE7; font-size:1.6rem; font-weight:700;">MMSI {top["mmsi"]}</div>'
                f'<div style="color:#6B8A82; font-family:\'JetBrains Mono\', monospace; font-size:0.85rem; margin-top:0.2rem;">{top["ship_type"]} &nbsp;&bull;&nbsp; {top["risk_label"]}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-family:\'JetBrains Mono\', monospace; color:#00FF9C; font-size:2.2rem; font-weight:700; text-shadow:0 0 12px rgba(0,255,156,0.4);">{top["suspect_score"]}/100</div>'
                f'<div style="color:#6B8A82; font-size:0.7rem; font-family:\'JetBrains Mono\', monospace;">SUSPECT SCORE</div>'
                f'</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(top_card_html, unsafe_allow_html=True)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Distance from Spill", f"{top['distance_from_spill_km']} km")
            d2.metric("Speed", f"{top['speed_knots']} knots")
            d3.metric("AIS Signal Gap", "Yes" if top['ais_signal_gap'] else "No")
            d4.metric("Prior Incidents", int(top['prior_incidents']))

            if top['prior_incidents'] > 0:
                st.warning(f"⚠️ Repeat Offender Alert: MMSI {top['mmsi']} has appeared as a suspect vessel in {top['prior_incidents']} prior recorded incident(s).")

            if st.button("📊 View Full Analysis"):
                st.session_state.show_engine3_details = True
                st.rerun()

        elif spill_detected:
            if st.button("⬅ Back"):
                st.session_state.show_engine3_details = False
                st.rerun()

            st.markdown('<div class="engine-card"><h3>📊 Full Vessel Analysis</h3></div>', unsafe_allow_html=True)

            distance_pts = 35 if top['distance_from_spill_km'] <= 5 else (20 if top['distance_from_spill_km'] <= 10 else (10 if top['distance_from_spill_km'] <= 20 else 0))
            gap_pts = 30 if top['ais_signal_gap'] else 0
            traj_pts = round(top['trajectory_match'] * 25, 1)
            type_pts = 10 if top['ship_type'] == "Tanker" else (5 if top['ship_type'] == "Cargo" else 0)

            st.markdown("###### Why the top suspect scored this way")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Proximity", f"+{distance_pts}")
            b2.metric("AIS Signal Gap", f"+{gap_pts}")
            b3.metric("Trajectory Match", f"+{traj_pts}")
            b4.metric("Vessel Type", f"+{type_pts}")

            st.markdown("###### All vessels by suspect score")
            chart_data = ranked[['mmsi', 'suspect_score', 'risk_label']].copy()
            chart_data['mmsi'] = chart_data['mmsi'].astype(str)

            color_scale = alt.Scale(
                domain=["🔴 HIGH RISK", "🟡 MEDIUM RISK", "🟢 LOW RISK"],
                range=["#EF4444", "#F59E0B", "#22C55E"]
            )

            chart = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('suspect_score:Q', title='Suspect Score', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('mmsi:N', sort='-x', title='Vessel (MMSI)'),
                color=alt.Color('risk_label:N', scale=color_scale, legend=alt.Legend(title="Risk Tier")),
                tooltip=['mmsi', 'suspect_score', 'risk_label']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("###### Full vessel data table")
            st.dataframe(
                ranked[['mmsi', 'ship_type', 'speed_knots', 'distance_from_spill_km', 'ais_signal_gap', 'suspect_score', 'risk_label', 'prior_incidents']],
                width='stretch',
                column_config={
                    "mmsi": "MMSI (Vessel ID)",
                    "ship_type": "Type",
                    "speed_knots": "Speed (knots)",
                    "distance_from_spill_km": "Distance (km)",
                    "ais_signal_gap": "AIS Signal Gap",
                    "suspect_score": st.column_config.ProgressColumn("Suspect Score", min_value=0, max_value=100, format="%.1f"),
                    "risk_label": "Risk Tier",
                    "prior_incidents": "Prior Incidents Flagged",
                }
            )

            if st.button("⬅ Back", key="bottom_back_button"):
                st.session_state.show_engine3_details = False
                st.rerun()

    # ---------------- TAB 4: RESPONSE ----------------
    with tab4:
        st.markdown('<div class="engine-card"><h3>🚨 Recommended Response Actions</h3></div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Response Urgency", urgency)
        r2.metric("Est. Response Window", "< 3 hrs" if urgency == "IMMEDIATE" else "< 12 hrs" if urgency == "PRIORITY" else "Routine")
        r3.metric("Recommended Authority", "Indian Coast Guard + State Pollution Control Board" if spill_detected else "None — monitoring only")

        if spill_detected:
            st.info(f"📋 **Suggested Action:** Alert nearest Coast Guard station of spill at ({spill_lat}, {spill_lon}). "
                    f"Flag MMSI {top['mmsi']} for boarding/inspection based on {top['suspect_score']}/100 suspect score. "
                    f"Deploy containment resources along projected {forecast_hours[-1]}h drift path.")
        else:
            st.success("✅ No active spill detected. No response action required — system remains in passive monitoring mode.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- DATABASE SAVE ----
    save_status_placeholder = st.empty()
    if spill_detected:
        with st.spinner("Saving results to database..."):
            success, result = save_pipeline_results(
                spill_lat, spill_lon,
                round(area_km2, 4),
                image_path,
                ranked
            )
        if success:
            save_status_placeholder.success(f"✅ Results saved to database (Spill ID: {result})")
        else:
            save_status_placeholder.warning(f"⚠️ Database save failed: {result}")
        time.sleep(2)
        save_status_placeholder.empty()
    else:
        save_status_placeholder.info("ℹ️ No spill detected — nothing saved to database for this run.")
        time.sleep(1.5)
        save_status_placeholder.empty()

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 1rem; color: #666;">
        <h3>Set spill parameters in the sidebar and click <b>Run ORCA Pipeline</b> to begin.</h3>
        <p>Rapid spill detection · Origin & drift forecasting · Vessel accountability — built for faster marine disaster response</p>
    </div>
    """, unsafe_allow_html=True)