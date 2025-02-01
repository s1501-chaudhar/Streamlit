import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.title("House Price Prediction App")
st.write("Upload a dataset to train a model and predict house prices.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Display dataset information
    st.write("### Dataset Information")
    st.write(f"Shape: {data.shape}")
    st.write("Columns:")
    st.write(data.columns.tolist())

    # Check for missing values
    st.write("### Missing Values")
    st.write(data.isnull().sum())

    # Feature and target selection
    st.write("### Feature Selection")
    target_column = st.selectbox("Select the target column (Price)", data.columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in data.columns if col != target_column])
    
    if target_column and feature_columns:
        # Split the data
        X = data[feature_columns]
        y = data[target_column]
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a model
        st.write("### Model Training")
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"R-squared Score: {r2:.2f}")

        # Feature importance
        st.write("### Feature Importance")
        feature_importances = pd.DataFrame(
            {"Feature": X.columns, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False)
        st.dataframe(feature_importances)

        # Visualization
        st.write("### Actual vs Predicted Prices")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Predict on new data
        st.write("### Predict House Price for New Data")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.text_input(f"Enter value for {col}", "0")
        
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.write(f"Predicted House Price: {prediction:.2f}")

else:
    st.write("Please upload a dataset to proceed.")

# Footer
st.write("Developed by Shardul")
