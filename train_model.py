from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent

def create_dataset(n=600, seed=42):
    rng = np.random.default_rng(seed)
    area = rng.integers(500, 3001, n)
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 41, n)
    noise = rng.normal(0, 4, n)
    price = 12 + 0.045 * area + 5 * bedrooms - 0.35 * age + noise
    return pd.DataFrame({
        "area_sqft": area,
        "bedrooms": bedrooms,
        "age_years": age,
        "price_lakhs": price
    })

def main():
    print("1. Creating synthetic house-price dataset...")
    data = create_dataset()
    X = data[["area_sqft", "bedrooms", "age_years"]]
    y = data["price_lakhs"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

    print("2. Training Linear Regression pipeline...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)

    print("3. Evaluating...")
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    report = (
        f"Test-set metrics\nMAE: {mae:.3f} lakh\n"
        f"RMSE: {rmse:.3f} lakh\nR2: {r2:.3f}\n"
    )
    print(report)

    print("4. Saving model and metrics...")
    joblib.dump(model, HERE / "model.joblib")
    (HERE / "metrics.txt").write_text(report, encoding="utf-8")
    print("Saved model.joblib and metrics.txt")
    print("Training complete. Run: streamlit run app.py")

if __name__ == "__main__":
    main()
