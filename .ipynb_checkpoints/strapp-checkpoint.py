import pandas as pd
import streamlit as st
import numpy as np

# Create a title for your app
st.title('My first Streamlit app')

# display a simple dataframe
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

# display the dataframe
st.write("Here's our first dataframe:")
st.write(df)

# display a line chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)