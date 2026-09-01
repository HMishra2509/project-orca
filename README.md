# 🐋 Project ORCA

**O**il-spill **R**ecognition, **C**orrelation & **A**ttribution

An AI-driven system that detects oil spills from satellite imagery, traces their origin and drift using live oceanographic data, and attributes them to the most probable responsible vessel through a transparent, explainable scoring model.

**Detect · Trace · Attribute**

---

## 📋 Problem Statement

| | |
|---|---|
| **Problem Statement ID** | SIH26143 |
| **Organization** | National Technical Research Organisation (NTRO) |
| **Category** | Software |
| **Theme** | Disaster Management |
| **Team** | Orcinus |

Oil spills cause severe, often irreversible damage to marine ecosystems and coastal economies. Identifying the responsible vessel is critical for both environmental response and legal accountability — but manual detection and attribution is slow and rarely provides a transparent basis for action. ORCA solves this end-to-end.

---

## ✨ Features

- 🛰️ **AI-powered spill detection** — a trained U-Net segmentation model identifies spill boundaries from satellite (SAR) imagery, plus area, shape, and age characterization
- 🌊 **Live drift prediction** — real ocean current and wind data drives a 9-point hindcast/forecast trajectory
- 🚢 **Explainable vessel attribution** — real AIS vessel data scored through a fully transparent, auditable formula, with risk tiers and repeat-offender detection
- 🗺️ **Interactive dashboard** — live map, vessel markers, standard/satellite map toggle, manual or random test image input
- 🇮🇳 **Built for Indian coastal waters** — preset locations (Gulf of Kutch, Paradip Port, Mumbai High offshore field), with optional real live AIS integration for Indian shipping lanes
- 💾 **Persistent history** — every run is saved to a database, enabling genuine pattern detection across incidents over time

---

## 🏗️ Architecture

```
Satellite Image
      │
      ▼
[ ENGINE 1: Detection ]  →  U-Net (PyTorch) → spill area, shape, age
      │
      ▼
[ ENGINE 2: Drift ]  →  Live ocean/wind data → hindcast origin + forecast path
      │
      ▼
[ ENGINE 3: Attribution ]  →  Real AIS data → explainable suspect scoring
      │
      ▼
[ Database: Supabase ]  →  [ Dashboard: Streamlit ]
```

Each engine is independently built and connected only through shared coordinates — meaning any single data source (e.g., swapping in a live satellite feed or a paid AIS subscription) can be upgraded without redesigning the rest of the system.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Detection Model | PyTorch, U-Net (~31M parameters), OpenCV |
| Drift Calculation | Open-Meteo API (live ocean + wind data) |
| Vessel Attribution | Real AIS vessel dataset, VesselAPI (optional live mode) |
| Database | Supabase (PostgreSQL + PostGIS) |
| Dashboard | Streamlit, Folium, Altair |
| Training Infrastructure | Google Colab (GPU) |

---

## 📂 Project Structure

```
orca-project/
├── app.py                    # Main Streamlit dashboard application
├── unet_final_model.pth      # Trained U-Net model weights
├── requirements.txt          # Python dependencies
├── data/                     # Sample satellite test images
├── ais_data/
│   └── ais_data.csv          # Real historic AIS vessel dataset
├── ship_images/              # Vessel type reference photos
├── uploaded_images/          # User-uploaded images (created at runtime)
└── .env                      # API keys and secrets (not committed)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A free [Supabase](https://supabase.com) project (for the database)
- A free [Open-Meteo](https://open-meteo.com) account (no key required for basic use)

### Installation

```bash
# Clone the repository
git clone https://github.com/HMishra2509/project-orca.git
cd project-orca

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
VESSELAPI_KEY=your_vesselapi_key_here   # optional, for real live AIS mode
```

### Database Setup

In your Supabase project, create three tables: `spills`, `vessels`, and `attributions`, linked via `spill_id` and `vessel_id` foreign keys.

### Run the App

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🎯 How It Works

1. **Select or upload a satellite image**, and choose a spill location (preset Indian coastal points or manual coordinates)
2. Click **Run ORCA Pipeline**
3. **Engine 1** detects the spill and reports its area, shape, and estimated age
4. **Engine 2** calculates a 9-point drift trajectory using live ocean and wind data
5. **Engine 3** ranks nearby vessels by an explainable suspect score, shown as a compact top-suspect card with an option to view the full analysis
6. Results are automatically saved to the database

---

## ⚠️ Known Limitations & Honesty Notes

We believe in being transparent about what's real vs. representative in this build:

- **Satellite imagery** is sourced from a pre-existing labeled dataset, not a live satellite feed
- **Vessel positions** (in default mode) are real vessel identities placed at realistic representative positions, since live AIS position-tracking typically requires a paid enterprise subscription. An optional **real live AIS mode** is available for Indian coastal waters via VesselAPI where data availability allows.
- **Drift prediction** assumes broadly stable ocean/wind conditions across the prediction window rather than updating per-hour

These are documented design decisions made under real data-access constraints, not oversights — and the modular architecture means each can be upgraded independently as better data access becomes available.

---

## 🔭 Future Scope

- Live satellite feed integration (e.g., ISRO or Sentinel data)
- Full production-grade live AIS position tracking
- Per-hour forecasted ocean/wind conditions
- Coastline-aware drift path clipping
- Partnership with agencies (ISRO, INCOIS, Indian Coast Guard) for production deployment

---

## 👥 Team Orcinus

| Member | Focus Area |
|---|---|
| ML/AI Engineering & System Architecture | Model development, pipeline integration |
| Data Science / Predictive Modeling | Drift physics, scoring algorithm |
| Data Engineering | Dataset sourcing and validation |
| Backend & Database Engineering | Supabase schema, data persistence |
| DevOps & QA Engineering | Testing, deployment reliability |
| Frontend & UX Engineering | Dashboard design and experience |

---

## 📄 License

This project was built for Smart India Hackathon 2026 (SIH26143).

---

**Detect · Trace · Attribute** 🐋
