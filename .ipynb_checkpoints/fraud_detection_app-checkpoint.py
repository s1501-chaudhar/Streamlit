import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# App Title
st.title("Credit Card Fraud Detection App")
st.write("Identify fraudulent transactions using machine learning.")

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type="csv")

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Display data information
    st.write("### Dataset Information")
    st.write(data.describe())
    
    # Check for the 'Class' column (assumed to represent fraud)
    if 'Class' not in data.columns:
        st.error("Dataset must contain a 'Class' column to indicate fraud (1 for fraud, 0 for non-fraud).")
    else:
        # Fraud and non-fraud distribution
        st.write("### Fraudulent vs. Non-Fraudulent Transactions")
        fraud_counts = data['Class'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=fraud_counts.index, y=fraud_counts.values, ax=ax, palette="Set2")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Fraud (0)', 'Fraud (1)'])
        ax.set_title("Transaction Class Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Feature Selection
        st.write("### Select Features for Model")
        features = st.multiselect("Select features to use for fraud detection", data.columns.tolist(), default=data.columns[:-1])

        if len(features) < 2:
            st.warning("Select at least two features for the model.")
        else:
            # Train-Test Split
            X = data[features]
            y = data['Class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Model Training
            st.write("### Training a Random Forest Classifier")
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            st.write(f"**Accuracy**: {accuracy:.2f}")
            st.write(f"**Precision**: {precision:.2f}")
            st.write(f"**Recall**: {recall:.2f}")

            # Confusion Matrix
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            # Fraud Prediction on New Data
            st.write("### Test Fraud Detection on New Transactions")
            new_data = st.file_uploader("Upload a new CSV file for fraud prediction", type="csv")

            if new_data:
                test_data = pd.read_csv(new_data)
                st.write("Uploaded Data Preview")
                st.dataframe(test_data.head())

                if set(features).issubset(test_data.columns):
                    predictions = model.predict(test_data[features])
                    test_data['Fraud Prediction'] = predictions
                    st.write("### Prediction Results")
                    st.dataframe(test_data)

                    # Download predictions
                    csv = test_data.to_csv(index=False)
                    st.download_button("Download Predictions", data=csv, file_name="fraud_predictions.csv", mime="text/csv")
                else:
                    st.error("The uploaded dataset must contain the selected features.")

else:
    st.write("Please upload a dataset to begin.")

# Footer
st.write("Developed by Shardul")
