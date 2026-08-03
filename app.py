import streamlit as st
import pandas as pd
import joblib
import os
import sklearn

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

MODEL_PATH = "best_house_price_model.pkl"
DATA_PATH = "train-00000-of-00001.parquet"
TARGET_COL = "price_bdt"
DROP_COLS = ["id"]
CAT_COLS = ["city", "property_type", "condition", "furnishing"]


def _build_pipeline():
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough"
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])


def _retrain_and_save():
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in [TARGET_COL] + DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


@st.cache_resource(show_spinner="Loading / training model...")
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            if not os.path.exists(DATA_PATH):
                st.error(
                    f"Neither model file ({MODEL_PATH}) nor training data "
                    f"({DATA_PATH}) found. Cannot proceed."
                )
                return None
            st.warning("Model file missing — retraining from data (30-60s)...")
            return _retrain_and_save()

        return joblib.load(MODEL_PATH)

    except (AttributeError, ModuleNotFoundError, ValueError, Exception) as e:
        st.warning(
            f"Couldn't load saved model (sklearn v{sklearn.__version__}): "
            f"{type(e).__name__}. Retraining from data now...",
            icon="🔧"
        )
        if not os.path.exists(DATA_PATH):
            st.error(f"Training data ({DATA_PATH}) also missing. Cannot proceed.")
            return None
        try:
            return _retrain_and_save()
        except Exception as e2:
            st.error(f"Retrain failed: {type(e2).__name__}: {e2}")
            return None


model = load_model()

st.title("🏠 House Price Prediction")
st.write("Predict house prices in Bangladesh using Machine Learning")

st.markdown("---")

st.subheader("📍 Property Details")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "City",
        ["Dhaka", "Chattogram", "Barishal", "Khulna", "Mymensingh", "Rajshahi", "Rangpur", "Sylhet"]
    )

    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "Condo", "Duplex", "House", "Townhouse"]
    )

    area_sqft = st.number_input(
        "Area (sqft)",
        min_value=450,
        max_value=5000,
        value=1500,
        step=50
    )

    bedrooms = st.slider(
        "Bedrooms",
        1,
        8,
        3
    )

    bathrooms = st.slider(
        "Bathrooms",
        1,
        6,
        2
    )

    floors = st.slider(
        "Total Floors in Building",
        1,
        25,
        10
    )

    floor_number = st.slider(
        "Floor Number (0 = Ground Floor)",
        0,
        25,
        5
    )

with col2:
    year_built = st.number_input(
        "Year Built",
        min_value=1985,
        max_value=2026,
        value=2018
    )

    age_years = 2026 - year_built
    st.info(f"Property Age: **{age_years} years**")

    condition = st.selectbox(
        "Condition",
        ["New", "Excellent", "Good", "Fair", "Needs Renovation"]
    )

    furnishing = st.selectbox(
        "Furnishing",
        ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
    )

    parking_spaces = st.slider(
        "Parking Spaces",
        0,
        3,
        1
    )

    distance_to_city_center_km = st.number_input(
        "Distance to City Center (km)",
        min_value=0.3,
        max_value=70.0,
        value=5.0,
        step=0.5
    )

    crime_rate_index = st.slider(
        "Crime Rate Index (5-100)",
        5.0,
        100.0,
        45.0,
        step=0.5
    )

st.markdown("---")
st.subheader("✅ Amenities")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    has_garden = st.checkbox("🌳 Garden", value=False)

with col_b:
    has_pool = st.checkbox("🏊 Swimming Pool", value=False)

with col_c:
    near_school = st.checkbox("🏫 Near School", value=True)

with col_d:
    near_hospital = st.checkbox("🏥 Near Hospital", value=True)

st.markdown("---")

has_garden = int(has_garden)
has_pool = int(has_pool)
near_school = int(near_school)
near_hospital = int(near_hospital)

if st.button("🔮 Predict House Price", type="primary", use_container_width=True):

    if model is None:
        st.error("Model could not be loaded. Please check the model file.")
    else:
        input_data = pd.DataFrame({
            "city": [city],
            "property_type": [property_type],
            "area_sqft": [area_sqft],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "floors": [floors],
            "floor_number": [floor_number],
            "year_built": [year_built],
            "age_years": [age_years],
            "condition": [condition],
            "furnishing": [furnishing],
            "parking_spaces": [parking_spaces],
            "has_garden": [has_garden],
            "has_pool": [has_pool],
            "distance_to_city_center_km": [distance_to_city_center_km],
            "near_school": [near_school],
            "near_hospital": [near_hospital],
            "crime_rate_index": [crime_rate_index]
        })

        try:
            with st.spinner("Calculating prediction..."):
                prediction = model.predict(input_data)[0]

            st.markdown("### 💰 Predicted House Price")
            st.success(f"৳ {prediction:,.2f}")

            price_lakhs = prediction / 100000
            price_crore = prediction / 10000000

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("In Lakhs", f"৳ {price_lakhs:,.2f} L")
            with col_p2:
                st.metric("In Crore", f"৳ {price_crore:,.2f} Cr")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption("🏠 House Price Prediction System | Model: Random Forest Regressor")
