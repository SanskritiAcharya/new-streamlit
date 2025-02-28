import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Data
@st.cache_data
def load_data():
    file_path = "0reduced_retail_sales_data.csv"
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# Encode Product Category
if "Product_Category" in df.columns:
    df = pd.get_dummies(df, columns=["Product_Category"], drop_first=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Sales Prediction"])

# Home Page
if page == "Home":
    st.title("Sales Forecasting and Optimization for Retail Business")
    st.write("This app predicts sales based on various features using Linear Regression.")
    st.write("Navigate to different sections using the sidebar.")

# Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    st.subheader("Sales Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Sales_Amount"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Sales Prediction Page
elif page == "Sales Prediction":
    st.title("Sales Prediction Using Linear Regression")
    
    # Feature Selection
    feature_cols = ["Quantity_Sold", "Price_per_Unit", "Discount"] + [col for col in df.columns if "Product_Category" in col]
    target_col = "Sales_Amount"
    X = df[feature_cols]
    y = df[target_col]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model Performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    st.subheader("Model Performance")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    
    # User Input for Prediction
    st.subheader("Make a Prediction")
    quantity = st.number_input("Quantity Sold", min_value=1, value=10)
    price = st.number_input("Price per Unit", min_value=0.1, value=20.0)
    discount = st.slider("Discount", min_value=0.0, max_value=1.0, value=0.1)
    
    product_categories = [col.replace("Product_Category_", "") for col in df.columns if "Product_Category" in col]
    selected_category = st.selectbox("Select Product Category", product_categories)
    
    category_data = {f"Product_Category_{cat}": 1 if cat == selected_category else 0 for cat in product_categories}
    
    if st.button("Predict Sales"):
        input_data = [[quantity, price, discount] + list(category_data.values())]
        prediction = model.predict(input_data)[0]
        st.write(f"### Predicted Sales Amount: ${prediction:.2f}")
        
        # Dynamic Visualization
        st.subheader("Dynamic Prediction Visualization")
        fig, ax = plt.subplots()
        ax.plot(range(len(y_test)), y_test.values, label="Actual", marker="o", linestyle="dotted")
        ax.plot(range(len(y_test)), y_pred, label="Predicted", marker="x", linestyle="dashed")
        ax.axhline(prediction, color='red', linestyle='-', label="Your Prediction")
        ax.set_xlabel("Test Samples")
        ax.set_ylabel("Sales Amount")
        ax.legend()
        st.pyplot(fig)