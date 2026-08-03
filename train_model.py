import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

DATA_PATH = "train-00000-of-00001.parquet"
MODEL_PATH = "best_house_price_model.pkl"

print("📊 Loading training data...")
df = pd.read_parquet(DATA_PATH)
print(f"   Data shape: {df.shape}")

TARGET_COL = "price_bdt"
DROP_COLS = ["id"]

feature_cols = [c for c in df.columns if c not in [TARGET_COL] + DROP_COLS]
print(f"   Features ({len(feature_cols)}): {feature_cols}")

cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]
print(f"   Categorical: {cat_cols}")
print(f"   Numerical: {num_cols}")

X = df[feature_cols]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n   Train: {X_train.shape}, Test: {X_test.shape}")

print("\n🔧 Building pipeline (ColumnTransformer + RandomForestRegressor)...")
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

print("🚀 Training model...")
pipeline.fit(X_train, y_train)

print("\n📈 Evaluating...")
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"   MAE:  ৳ {mae:,.2f}")
print(f"   RMSE: ৳ {rmse:,.2f}")
print(f"   R²:   {r2:.4f}")

print("\n🔍 Cross-validation (3-fold)...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
print(f"   CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print(f"\n💾 Saving model to {MODEL_PATH} ...")
joblib.dump(pipeline, MODEL_PATH)
size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"   Model size: {size_mb:.2f} MB")

print("\n✅ Quick smoke test on 5 samples...")
sample = X_test.head(5)
preds = pipeline.predict(sample)
actuals = y_test.head(5).values
for i in range(5):
    pct_err = abs(preds[i] - actuals[i]) / actuals[i] * 100
    print(f"   #{i+1}: Predicted ৳ {preds[i]:,.0f} | Actual ৳ {actuals[i]:,.0f} | Error {pct_err:.1f}%")

print("\n🎉 Model retraining complete!")
