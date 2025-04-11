import streamlit as st
import pandas as pd
import joblib

st.title("Fraud Detection App - Testing")
# Load the trained model
try:
    model = joblib.load('fraud_model.pkl')
except FileNotFoundError:
    st.error("Error: 'fraud_model.pkl' not found. Make sure the model file is in the same directory as this script.")
    st.stop()

st.title('Credit Card Fraud Detection')
st.subheader('Enter the transaction details below to get a prediction.')

# Explanation of the input features (optional but helpful)
st.markdown("### Feature Descriptions:")
st.markdown("""
This interface allows you to input transaction details and get a fraud prediction. The features correspond to anonymized variables V1 to V28, the transaction Amount, and the Time elapsed since the first transaction.
""")

# Create input fields for each feature
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                 'Amount']
features = {}
for feature in feature_names:
    features[feature] = st.number_input(f"{feature}:", step=0.001)

if st.button('Predict'):
    # Create a DataFrame from the input features
    input_data = pd.DataFrame([features])

    try:
        # Make the prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100 # Probability of being fraudulent

        st.subheader("Prediction Result:")
        if prediction == 0:
            st.success(f'This transaction is likely NOT fraudulent. (Probability of Fraud: {probability:.2f}%)')
        else:
            st.error(f'This transaction is likely FRAUDULENT! (Probability of Fraud: {probability:.2f}%)')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.sidebar.header("About")
st.sidebar.info(
    "This is a simple credit card fraud detection interface built with Streamlit. "
    "It uses a pre-trained Random Forest model to predict whether a transaction is fraudulent or not."
)