import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load dataset
data = pd.read_csv("instagram_data.csv")

print("\nDataset loaded successfully!")
print("Shape:", data.shape)

# Clean column names
data.columns = data.columns.str.strip()

print("\nColumns:")
print(data.columns.tolist())


# Features from your actual dataset
features = [
    "Likes",
    "Comments",
    "Shares",
    "Saves",
    "Profile Visits",
    "Follows"
]

# Target
target = "Impressions"


# Check columns
missing_columns = []

for column in features + [target]:

    if column not in data.columns:
        missing_columns.append(column)


if missing_columns:

    print("\n❌ Missing columns:")
    print(missing_columns)

    print("\nYour actual columns are:")
    print(data.columns.tolist())

    exit()


# Select X and y
X = data[features]
y = data[target]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Polynomial features
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)


X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)


# Linear Regression
model = LinearRegression()

model.fit(
    X_train_poly,
    y_train
)


# Prediction
y_pred = model.predict(X_test_poly)


# R2 Score
score = r2_score(
    y_test,
    y_pred
)


print("\n================================")
print("MODEL TRAINED SUCCESSFULLY")
print("================================")

print("R2 Score:", round(score, 4))


# Save model
with open("model.pkl", "wb") as file:

    pickle.dump(
        model,
        file
    )


# Save polynomial transformer
with open("poly.pkl", "wb") as file:

    pickle.dump(
        poly,
        file
    )


print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print("✅ model.pkl")
print("✅ poly.pkl")