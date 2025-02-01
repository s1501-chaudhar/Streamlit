import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.title("Customer Churn Prediction App")
st.write("Upload a dataset to train a model and predict customer churn.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Display basic dataset information
    st.write("### Dataset Information")
    st.write(f"Shape: {data.shape}")
    st.write("Columns:")
    st.write(data.columns.tolist())

    # Check for missing values
    st.write("### Missing Values")
    st.write(data.isnull().sum())

    # Feature selection
    st.write("### Feature Selection")
    target_column = st.selectbox("Select the target column (Churn Indicator)", data.columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in data.columns if col != target_column])
    
    if target_column and feature_columns:
        # Split the data
        X = data[feature_columns]
        y = data[target_column]
        
        # Convert categorical variables to numeric (if needed)
        X = pd.get_dummies(X, drop_first=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a model
        st.write("### Model Training")
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)

        # Model evaluation
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        # Confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Classification report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Predict on new customer data
        st.write("### Predict on New Data")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.text_input(f"Enter value for {col}", "0")

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = classifier.predict(input_df)[0]
            prediction_proba = classifier.predict_proba(input_df)
            st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
            st.write(f"Prediction Probability: {prediction_proba}")

else:
    st.write("Please upload a dataset to proceed.")

# Footer
st.write("Developed by Shardul")
