# Step 1: Import libraries
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor


# Step 2: Page configuration
st.set_page_config(
    page_title="KNN Regression App",
    page_icon="📘",
    layout="centered"
)


# Step 3: Title
st.title("📘 KNN Regression Student Score Prediction")

st.write(
    "Predict Math Score using K-Nearest Neighbors Regression"
)


# Step 4: Load dataset
df = pd.read_csv("StudentsPerformance (1).csv")


# Step 5: Encode categorical data
label_encoder = LabelEncoder()

df["gender"] = label_encoder.fit_transform(df["gender"])

df["race/ethnicity"] = label_encoder.fit_transform(
    df["race/ethnicity"]
)

df["parental level of education"] = label_encoder.fit_transform(
    df["parental level of education"]
)

df["lunch"] = label_encoder.fit_transform(df["lunch"])

df["test preparation course"] = label_encoder.fit_transform(
    df["test preparation course"]
)


# Step 6: Define X and y
X = df.drop("math score", axis=1)

y = df["math score"]


# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 8: Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# Step 9: Create model
knn_regressor = KNeighborsRegressor(
    n_neighbors=5,
    metric='minkowski',
    p=2
)


# Step 10: Train model
knn_regressor.fit(X_train_scaled, y_train)


# Step 11: User input
st.subheader("Enter Student Details")


gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

if gender == "Female":
    gender_value = 0
else:
    gender_value = 1


race = st.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

race_mapping = {
    "group A": 0,
    "group B": 1,
    "group C": 2,
    "group D": 3,
    "group E": 4
}

race_value = race_mapping[race]


education = st.selectbox(
    "Parental Education",
    [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school"
    ]
)

education_mapping = {
    "associate's degree": 0,
    "bachelor's degree": 1,
    "high school": 2,
    "master's degree": 3,
    "some college": 4,
    "some high school": 5
}

education_value = education_mapping[education]


lunch = st.selectbox(
    "Lunch Type",
    ["free/reduced", "standard"]
)

if lunch == "free/reduced":
    lunch_value = 0
else:
    lunch_value = 1


test_prep = st.selectbox(
    "Test Preparation",
    ["none", "completed"]
)

if test_prep == "none":
    test_prep_value = 0
else:
    test_prep_value = 1


reading_score = st.slider(
    "Reading Score",
    0,
    100,
    50
)

writing_score = st.slider(
    "Writing Score",
    0,
    100,
    50
)


# Step 12: Predict button
if st.button("Predict Math Score"):

    new_data = [[
        gender_value,
        race_value,
        education_value,
        lunch_value,
        test_prep_value,
        reading_score,
        writing_score
    ]]

    new_data_scaled = scaler.transform(new_data)

    prediction = knn_regressor.predict(new_data_scaled)

    st.success(
        f"Predicted Math Score: {prediction[0]:.2f}"
    )