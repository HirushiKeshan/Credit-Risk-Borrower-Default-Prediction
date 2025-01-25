import streamlit as st
import pickle
import pandas as pd

# Load the model
model_name = "XGBClassifier.pkl"  # Update this path
try:
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found: {model_name}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define expected feature names based on your model's training data
expected_features = [
    "NUMBER_OF_INSTALLMENTS", "SANCTION_AMT", "TOT_OUTSTD_BAL", "OVER_DUE_AMT", 
    "AMOUNT_OF_INSTALLMENT", "NO_OF_DAYS_PAST_DUE", "currency_code_TZS", 
    "currency_code_USD", "REPAYMENT_FREQUENCY_AnnualInstalments360Days", 
    "REPAYMENT_FREQUENCY_AtTheFinalDayOfThePeriodOfContract", 
    "REPAYMENT_FREQUENCY_FortnightlyInstalments15Days", 
    "REPAYMENT_FREQUENCY_IrregularInstalments", 
    "REPAYMENT_FREQUENCY_MonthlyInstalments30Days", 
    "REPAYMENT_FREQUENCY_SixMonthInstalments180Days", 
    "INSTALMENT_LOAN_TYPE_BusinessLoan", "INSTALMENT_LOAN_TYPE_ConsumerLoan", 
    "INSTALMENT_LOAN_TYPE_CreditCard", "INSTALMENT_LOAN_TYPE_LeasingFinancial", 
    "INSTALMENT_LOAN_TYPE_LineOfCreditOnCurrentAccount", 
    "INSTALMENT_LOAN_TYPE_MortgageLoan", "INSTALMENT_LOAN_TYPE_OtherInstalmentOperation", 
    "INSTALMENT_LOAN_TYPE_Overdraft", "loan_status_Existing", 
    "loan_status_TerminatedAccordingTheContract", 
    "loan_status_TerminatedInAdvanceCorrectly", 
    "loan_status_TerminatedInAdvanceIncorrectly", "YEAR_REPORTED", 
    "MONTH_REPORTED", "DAY_REPORTED", "DAY_OF_WEEK_REPORTED", 
    "ACCOUNT_AGE", "LAST_PAYMENT_AGE", "LOAN_DURATION"
]

# Function to collect user input
def user_input_features():
    NUMBER_OF_INSTALLMENTS = st.number_input("Number of Installments", min_value=1, value=1)
    SANCTION_AMT = st.number_input("Sanction Amount", min_value=0.0, value=0.0)
    TOT_OUTSTD_BAL = st.number_input("Total Outstanding Balance", min_value=0.0, value=0.0)
    OVER_DUE_AMT = st.number_input("Overdue Amount", min_value=0.0, value=0.0)
    AMOUNT_OF_INSTALLMENT = st.number_input("Amount of Installment", min_value=0.0, value=0.0)
    NO_OF_DAYS_PAST_DUE = st.number_input("Number of Days Past Due", min_value=0, value=0)
    
    # Categorical inputs
    REPAYMENT_FREQUENCY = st.selectbox("Repayment Frequency", [
        "MonthlyInstalments30Days", "AnnualInstalments360Days", 
        "FortnightlyInstalments15Days", "IrregularInstalments", 
        "SixMonthInstalments180Days", "AtTheFinalDayOfThePeriodOfContract"
    ])
    INSTALMENT_LOAN_TYPE = st.selectbox("Instalment Loan Type", [
        "BusinessLoan", "ConsumerLoan", "CreditCard", "LeasingFinancial", 
        "LineOfCreditOnCurrentAccount", "MortgageLoan", 
        "OtherInstalmentOperation", "Overdraft"
    ])
    loan_status = st.selectbox("Loan Status", [
        "Existing", "TerminatedAccordingTheContract", 
        "TerminatedInAdvanceCorrectly", "TerminatedInAdvanceIncorrectly"
    ])
    currency_code = st.selectbox("Currency Code", ["TZS", "USD"])
    
    # Additional variables
    YEAR_REPORTED = st.number_input("Year Reported", min_value=2000, max_value=2024, value=2024)
    MONTH_REPORTED = st.number_input("Month Reported", min_value=1, max_value=12, value=10)
    DAY_REPORTED = st.number_input("Day Reported", min_value=1, max_value=31, value=1)
    DAY_OF_WEEK_REPORTED = st.number_input("Day of Week Reported", min_value=1, max_value=7, value=1)
    ACCOUNT_AGE = st.number_input("Account Age (Days)", min_value=0, value=0)
    LAST_PAYMENT_AGE = st.number_input("Last Payment Age (Days)", min_value=0, value=0)
    LOAN_DURATION = st.number_input("Loan Duration (Days)", min_value=0, value=0)

    input_data = {
        "NUMBER_OF_INSTALLMENTS": NUMBER_OF_INSTALLMENTS,
        "SANCTION_AMT": SANCTION_AMT,
        "TOT_OUTSTD_BAL": TOT_OUTSTD_BAL,
        "OVER_DUE_AMT": OVER_DUE_AMT,
        "AMOUNT_OF_INSTALLMENT": AMOUNT_OF_INSTALLMENT,
        "NO_OF_DAYS_PAST_DUE": NO_OF_DAYS_PAST_DUE,
        "REPAYMENT_FREQUENCY_" + REPAYMENT_FREQUENCY: 1,
        "INSTALMENT_LOAN_TYPE_" + INSTALMENT_LOAN_TYPE: 1,
        "loan_status_" + loan_status: 1,
        "currency_code_" + currency_code: 1,
        "YEAR_REPORTED": YEAR_REPORTED,
        "MONTH_REPORTED": MONTH_REPORTED,
        "DAY_REPORTED": DAY_REPORTED,
        "DAY_OF_WEEK_REPORTED": DAY_OF_WEEK_REPORTED,
        "ACCOUNT_AGE": ACCOUNT_AGE,
        "LAST_PAYMENT_AGE": LAST_PAYMENT_AGE,
        "LOAN_DURATION": LOAN_DURATION
    }
    
    return pd.DataFrame([input_data])

# Set up the Streamlit app layout
st.title("Borrower Default Prediction")
st.write("Enter the borrower details below:")

# Get user input
input_df = user_input_features()

# Ensure input DataFrame has the same column order as expected by the model
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[expected_features]  # Ensure correct column order

# Display user input for debugging
st.write("User Input:", input_df)

# Button for prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]  # Probability of default
        st.write("Prediction: ", "Yes" if prediction[0] == 1 else "No")
        st.write("Probability of Default: {:.2f}%".format(prediction_proba[0] * 100))
    except Exception as e:
        st.error(f"Error during prediction: {e}")