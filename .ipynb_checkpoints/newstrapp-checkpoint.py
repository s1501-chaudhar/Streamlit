import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Data Science Project with Streamlit")
st.write("This app allows you to upload a dataset, explore it, and visualize key statistics.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Display dataset
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Dataset summary
    st.write("### Dataset Information")
    st.write(f"Shape of the dataset: {data.shape}")
    st.write("Columns in the dataset:")
    st.write(data.columns.tolist())
    st.write("Data types:")
    st.write(data.dtypes)

    # Missing values
    st.write("### Missing Values")
    st.write(data.isnull().sum())

    # Descriptive statistics
    st.write("### Descriptive Statistics")
    st.write(data.describe())

    # Select columns for visualization
    st.write("### Data Visualization")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_columns) > 0:
        x_axis = st.selectbox("Select a column for the x-axis", numeric_columns)
        y_axis = st.selectbox("Select a column for the y-axis", numeric_columns)
        plot_type = st.radio("Choose the type of plot", ["Scatter Plot", "Box Plot", "Histogram"])
        
        # Generate plots
        if plot_type == "Scatter Plot":
            st.write(f"Scatter Plot: {x_axis} vs {y_axis}")
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)
        elif plot_type == "Box Plot":
            st.write(f"Box Plot of {x_axis}")
            fig, ax = plt.subplots()
            sns.boxplot(data=data, x=x_axis, ax=ax)
            st.pyplot(fig)
        elif plot_type == "Histogram":
            st.write(f"Histogram of {x_axis}")
            fig, ax = plt.subplots()
            data[x_axis].plot(kind='hist', ax=ax, bins=20)
            st.pyplot(fig)
    else:
        st.write("No numeric columns available for visualization.")

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    if len(numeric_columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns to display a correlation heatmap.")
else:
    st.write("Please upload a dataset to begin!")

# Footer
st.write("Developed by Shardul")