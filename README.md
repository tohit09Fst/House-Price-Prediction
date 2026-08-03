# 🏠 House Price Prediction — Bangladesh

An end-to-end **Machine Learning web app** to predict residential property prices across 8 major cities in Bangladesh. Built with **scikit-learn** (Random Forest Regressor) + **Streamlit** with a clean, responsive UI. R² ≈ 0.61.

---

## ✨ Features

- 🔮 **Instant Price Prediction** — tweak 18 real-world property features and get a price in Bangladeshi Taka (৳), Lakhs, and Crore)
- 📍 **8 Cities Supported** — Dhaka, Chattogram, Barishal, Khulna, Mymensingh, Rajshahi, Rangpur, Sylhet
- 🏘️ **5 Property Types** — Apartment, Condo, Duplex, House, Townhouse
- 📊 **5 Condition Grades — New → Excellent → Good → Fair → Needs Renovation
- 🔄 **Bulletproof Model Loader** — auto retrains from parquet data if the pickle has a scikit-learn version mismatch (no more `AttributeError: _RemainderColsList` on Streamlit Cloud)
- 💻 **Clean Streamlit UI** — two-column form, checkbox amenities, live property age, Lakh/Crore breakdown, loading spinner, error handling, metrics cards

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── 🏠 app.py                          # Streamlit web UI + fallback retrain logic
├── 🤖 best_house_price_model.pkl     # Trained Random Forest pipeline (~45 MB)
├── 📊 train-00000-of-00001.parquet     # Training dataset (5000 rows × 20 cols)
├── 📋 requirements.txt              # EXACT pinned dependency versions
├── 🧪 test_predictions.py            # 5 end-to-end prediction test cases
├── 🏋️ train_model.py                 # One-click script to retrain the model
└── 📖 README.md                       # This file
```

---

## 🧠 Model & Architecture

### Pipeline
| Step | Component | Details |
|---|---|---|
| 1 | **ColumnTransformer** | One-Hot Encodes 4 categorical columns (city, property_type, condition, furnishing) + passthrough for 14 numeric |
| 2 | **RandomForestRegressor** | 300 trees, `min_samples_leaf=2`, `random_state=42`, `n_jobs=-1` |

### Performance
| Metric | Value |
|---|---|
| **Train/Test Split** | 80/20 (stratified random split) |
| **R² Score** | **0.6115** |
| **MAE** | ৳ 3,947,420 |
| **RMSE** | ৳ 5,838,641 |
| **3-Fold CV R²** | 0.5948 (± 0.0195) |
| **Model Size** | ~45 MB |

### Feature Correlation with Price (top drivers)
| Feature | Correlation |
|---|---|
| 📐 `area_sqft` | +0.47 |
| 🛏️ `bedrooms` | +0.31 |
| 🚿 `bathrooms` | +0.26 |
| 🌳 `has_garden` | +0.14 |
| 🚗 `parking_spaces` | +0.09 |
| 🏫 `near_school` | +0.08 |
| 🏥 `near_hospital` | +0.08 |
| 📏 `distance_to_city_center_km` | **-0.19** |
| 🏢 `floors` | **-0.16** |
| 🔴 `crime_rate_index` | **-0.13** |

---

## 📊 Dataset

**File:** `train-00000-of-00001.parquet`
- **Rows:** 5,000 property records
- **Columns:** 20 (18 features + `id` + `price_bdt`)

| Column | Type | Range / Values |
|---|---|---|
| `city` | categorical | Dhaka, Chattogram, Barishal, Khulna, Mymensingh, Rajshahi, Rangpur, Sylhet |
| `property_type` | categorical | Apartment, Condo, Duplex, House, Townhouse |
| `area_sqft` | int | 450 – 4,855 |
| `bedrooms` | int | 1 – 8 |
| `bathrooms` | int | 1 – 6 |
| `floors` | int | 1 – 25 (total floors in building) |
| `floor_number` | int | 0 – 25 (0 = ground floor) |
| `year_built` | int | 1985 – 2026 |
| `age_years` | int | 0 – 41 (derived: 2026 − year_built) |
| `condition` | categorical | New, Excellent, Good, Fair, Needs Renovation |
| `furnishing` | categorical | Unfurnished, Semi-Furnished, Fully Furnished |
| `parking_spaces` | int | 0 – 3 |
| `has_garden` | binary | 0 / 1 |
| `has_pool` | binary | 0 / 1 |
| `distance_to_city_center_km` | float | 0.3 – 65.55 |
| `near_school` | binary | 0 / 1 |
| `near_hospital` | binary | 0 / 1 |
| `crime_rate_index` | float | 5.0 – 100.0 |
| **`price_bdt`** | int | **৳ 641,000 – ৳ 95,179,000 (target) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit 1.39.1 |
| **ML Framework** | scikit-learn 1.6.1 (RandomForestRegressor, ColumnTransformer, Pipeline) |
| **Data** | pandas 2.2.1 + pyarrow 22.0.0 (Parquet) |
| **Serialization** | joblib 1.3.2 |
| **Numerics** | NumPy 1.26.4 |
| **Deployment** | Streamlit Cloud / any Streamlit-compatible host |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python **3.10 – 3.12
- pip / conda

### Step 1: Clone & Install Dependencies
```bash
cd House-Price-Prediction

# Option A — venv + pip
python3 -m venv venv
source venv/bin/activate         # macOS/Linux
# venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Option B — conda
conda create -n houseprice python=3.12 -y
conda activate houseprice
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
streamlit run app.py
```
Opens in your browser at **http://localhost:8501**

---

## 🧪 Running Tests

```bash
# 5 end-to-end prediction scenarios (Dhaka Apt → Sylhet Townhouse):
python test_predictions.py

# Retrain the model from scratch (rewrites best_house_price_model.pkl):
python train_model.py
```

Expected result:
```
✅ Test 1: Dhaka Apartment (Mid-range)   → ৳ 19,389,334.65
✅ Test 2: Chattogram Duplex (Luxury)  → ৳ 48,453,771.00
✅ Test 3: Khulna House (Budget)     → ৳ 4,273,080.68
✅ Test 4: Barishal New Condo        → ৳ 6,205,111.55
✅ Test 5: Sylhet Townhouse          → ৳ 13,823,626.04
🎉 All 5 test cases passed successfully!
```

---

## 🚀 Deployment (Streamlit Cloud)

1. Push this folder to a **public/private GitHub repo.
2. Go to <https://share.streamlit.io> → **New app** → pick the repo → set **Main file path** = `app.py` → **Deploy!**.
3. ✅ **requirements.txt pins **exact versions** → Cloud will install sklearn 1.6.1 so the pickle loads directly.
4. 🔧 **If Cloud still hits `AttributeError: _RemainderColsList?** → no problem: `load_model()` **auto retrains LIVE from `train-00000-of-00001.parquet on the first visitor (~40s) → saves a fresh pickle → instant for every visitor after that.

### Deployment Troubleshooting Checklist
- [ ] `requirements.txt` committed with **exact pins** (`==` not `>=`)
- [ ] `train-00000-of-00001.parquet` committed (fallback data)
- [ ] `best_house_price_model.pkl` committed (primary model, <100MB → within Streamlit Cloud limit OK)
- [ ] Python 3.12 in Advanced settings (matches requirements are for)

---

## ⚠️ Why `AttributeError Fix (The Scikit-Learn Pickle Problem)

**Error seen on Streamlit Cloud:**
```
AttributeError: Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ...>
```

**Root cause:** scikit-learn's `ColumnTransformer` uses private helper classes (`_RemainderColsList`) are not part of public API → when pickled in one version → fail to unpickle in another version, even slightly different.

**This project solves it 3 ways:**

1. **Pin **`scikit-learn==1.6.1 in `requirements.txt` so Cloud matches the pickle builder's version.
2. **Fallback retrain:** if joblib.load ANY reason → `app.fit on parquet data live in 30-60s auto and saves a new pickle.
3. **Architecture encoded in `_build_pipeline()` inside `app.py` always work no need for a matching pickle at all.

---

## 🔮 Using the Web UI

### Left Column
🏙️ City → 🏘️ Property Type → 📐 Area (sqft) → 🛏️ Bedrooms → 🚿 Bathrooms → 🏢 Total Floors → 🔢 Floor Number

### Right Column
📅 Year Built → ⚙️ Condition → 🪑 Furnishing → 🚗 Parking Spaces → 📏 Distance to City Center → 🔴 Crime Rate Index → 👀 Live **Property Age**

### 4 Amenity Checkboxes 🌳 Garden 🏊 Pool 🏫 School 🏥 Hospital

### Prediction Output 💰 🔮 Predict House Price → displays:
- **Primary:** ৳ 18,337,614.60 → -In:** ৳ 183.38 L → **Crore:** ৳ 1.83 Cr

---

## 📜 License

Educational / Demo project. Dataset is synthetic for demo purposes — prices are predictions do not reflect real market.

---

## 👨‍💻 Author

Built with Streamlit · Random Forest · 🏠 Happy Home prediction! 🇧🇩
