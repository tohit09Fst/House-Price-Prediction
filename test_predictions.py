import pandas as pd
import joblib
import sys

model = joblib.load("best_house_price_model.pkl")

print("Running prediction tests...\n")

test_cases = [
    {
        "name": "Dhaka Apartment (Mid-range)",
        "data": {
            "city": "Dhaka",
            "property_type": "Apartment",
            "area_sqft": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
            "floors": 10,
            "floor_number": 5,
            "year_built": 2018,
            "age_years": 8,
            "condition": "Good",
            "furnishing": "Semi-Furnished",
            "parking_spaces": 1,
            "has_garden": 0,
            "has_pool": 0,
            "distance_to_city_center_km": 5.0,
            "near_school": 1,
            "near_hospital": 1,
            "crime_rate_index": 45.0
        }
    },
    {
        "name": "Chattogram Duplex (Luxury)",
        "data": {
            "city": "Chattogram",
            "property_type": "Duplex",
            "area_sqft": 3200,
            "bedrooms": 5,
            "bathrooms": 4,
            "floors": 8,
            "floor_number": 7,
            "year_built": 2022,
            "age_years": 4,
            "condition": "Excellent",
            "furnishing": "Fully Furnished",
            "parking_spaces": 2,
            "has_garden": 1,
            "has_pool": 1,
            "distance_to_city_center_km": 3.0,
            "near_school": 1,
            "near_hospital": 1,
            "crime_rate_index": 25.0
        }
    },
    {
        "name": "Khulna House (Budget)",
        "data": {
            "city": "Khulna",
            "property_type": "House",
            "area_sqft": 900,
            "bedrooms": 2,
            "bathrooms": 1,
            "floors": 1,
            "floor_number": 0,
            "year_built": 2000,
            "age_years": 26,
            "condition": "Fair",
            "furnishing": "Unfurnished",
            "parking_spaces": 0,
            "has_garden": 0,
            "has_pool": 0,
            "distance_to_city_center_km": 12.0,
            "near_school": 0,
            "near_hospital": 0,
            "crime_rate_index": 60.0
        }
    },
    {
        "name": "Barishal New Condo",
        "data": {
            "city": "Barishal",
            "property_type": "Condo",
            "area_sqft": 1200,
            "bedrooms": 3,
            "bathrooms": 2,
            "floors": 15,
            "floor_number": 10,
            "year_built": 2026,
            "age_years": 0,
            "condition": "New",
            "furnishing": "Unfurnished",
            "parking_spaces": 1,
            "has_garden": 0,
            "has_pool": 0,
            "distance_to_city_center_km": 8.0,
            "near_school": 1,
            "near_hospital": 0,
            "crime_rate_index": 40.0
        }
    },
    {
        "name": "Sylhet Townhouse (Fair Condition)",
        "data": {
            "city": "Sylhet",
            "property_type": "Townhouse",
            "area_sqft": 2000,
            "bedrooms": 4,
            "bathrooms": 3,
            "floors": 3,
            "floor_number": 2,
            "year_built": 1995,
            "age_years": 31,
            "condition": "Needs Renovation",
            "furnishing": "Semi-Furnished",
            "parking_spaces": 2,
            "has_garden": 1,
            "has_pool": 0,
            "distance_to_city_center_km": 10.0,
            "near_school": 1,
            "near_hospital": 1,
            "crime_rate_index": 55.0
        }
    }
]

all_passed = True
for i, tc in enumerate(test_cases):
    try:
        df = pd.DataFrame([tc["data"]])
        pred = model.predict(df)[0]
        print(f"✅ Test {i+1}: {tc['name']}")
        print(f"   Predicted Price: ৳ {pred:,.2f} ({pred/100000:,.2f} L)\n")
    except Exception as e:
        print(f"❌ Test {i+1}: {tc['name']} FAILED")
        print(f"   Error: {str(e)}\n")
        all_passed = False

if all_passed:
    print("🎉 All 5 test cases passed successfully!")
    sys.exit(0)
else:
    print("⚠️  Some test cases failed!")
    sys.exit(1)
