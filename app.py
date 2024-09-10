import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Import joblib directly

# Load the pre-trained model (assuming it's a scikit-learn model)
model_path = r"customer_segmentation_model.joblib"
model = joblib.load(model_path)

# Set background image
def add_bg_from_local(image_path):
    bg = Image.open(image_path)
    st.image(bg, use_column_width=True)

# Example background image
bg_image_path = r"backg.jpeg"  # Update this to your background image path
add_bg_from_local(bg_image_path)

# App title and description
st.title('Customer Segmentation Prediction')
st.write('Enter the customer details below to predict the cluster they belong to.')
education_levels = {
    1: 'Undergraduate',
    2: 'Graduate',
    3: 'Postgraduate',
    4: 'Doctorate'
}
defaulted_options = {
    None: 'Select Default Status',
    0: '0',
    1: '1'
}
# Input fields (Default to None to allow empty inputs)
age = st.number_input('Age', min_value=18, max_value=100, value=None)
edu = st.selectbox('Education Level', options=[None] + list(education_levels.keys()),
                   format_func=lambda x: f"{x}: {education_levels[x]}" if x is not None else "Select Education Level")
years_employed = st.number_input('Years Employed', min_value=0, max_value=50, value=None)
income = st.number_input('Income ($)', min_value=0.0, value=None)
card_debt = st.number_input('Card Debt ($)', min_value=0.0, value=None)
other_debt = st.number_input('Other Debt ($)', min_value=0.0, value=None)
defaulted = st.selectbox('Defaulted', options=list(defaulted_options.keys()),
                         format_func=lambda x: defaulted_options[x])
debt_income_ratio = st.number_input('Debt to Income Ratio (%)', min_value=0.0, max_value=100.0, value=None)

# Ensure all fields are filled
if st.button('Predict Cluster'):
    if None in [age, years_employed, income, card_debt, other_debt, debt_income_ratio]:
        st.error("Please fill out all fields to make a prediction.")
    else:
        input_data = np.array([[age, edu, years_employed, income, card_debt, other_debt, defaulted, debt_income_ratio]])

        # Make prediction using the loaded model
        cluster_prediction = model.predict(input_data)

        # Display the result
        st.write(f"The customer belongs to cluster: **{cluster_prediction[0]}**")

        # Prepare data for saving and visualization
        user_data = {
            'Age': age,
            'Edu': edu,
            'Years Employed': years_employed,
            'Income': income,
            'Card Debt': card_debt,
            'Other Debt': other_debt,
            'Defaulted': defaulted,
            'DebtIncomeRatio': debt_income_ratio,
            'Predicted Cluster': cluster_prediction[0]
        }

        user_df = pd.DataFrame([user_data])

        # Save the prediction to a CSV file
        csv_path = 'user_prediction.csv'
        if os.path.isfile(csv_path):
            user_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            user_df.to_csv(csv_path, mode='w', header=True, index=False)

        st.success(f"Prediction saved to {csv_path}")

        # Visualize the user input data and predicted cluster
        st.write("---")
        st.write("## User Data Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=user_df.columns[:-1], y=user_df.iloc[0, :-1].values, ax=ax)
        plt.title(f'User Data with Predicted Cluster: {cluster_prediction[0]}')
        plt.xlabel('Features')
        plt.ylabel('Values')
        st.pyplot(fig)

# Footer
st.write("---")
st.write("Developed by [Your Name](https://your-website.com)")