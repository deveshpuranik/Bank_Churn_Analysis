import streamlit as st
import pandas as pd
import numpy as np
import joblib # Used for loading models/scalers
import altair as alt # Required for advanced charts

# --- Configuration ---
# UPDATED PATHS TO /Users/welcome/python_stream/
FILE_PATH = "/Users/welcome/Bank_Churn_GitHub/Bank Customer Churn Prediction.csv"
MODEL_PATH = "/Users/welcome/Bank_Churn_GitHub/Trained_model.sav"
# Preprocessor is set to look for preprocessor.pkl
PREPROCESSOR_PATH = "/Users/welcome/Bank_Churn_GitHub/preprocessor.pkl" 

st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
# @st.cache_data is used to speed up the app by caching the data load
@st.cache_data
def load_data(path):
    """Loads the customer churn data and drops the ID column."""
    try:
        # FIX: Added encoding='latin1' to handle potential non-UTF-8 characters
        data = pd.read_csv(path, encoding='latin1')
        
        # Standardize column names to lower_case for easier access
        data.columns = [col.lower() for col in data.columns]
        
        # FIX: Explicitly drop 'customer_id' as the trained model does not use it.
        if 'customer_id' in data.columns:
            data = data.drop(columns=['customer_id'])
            
        return data
    except FileNotFoundError:
        st.error(f"Error: File not found at {path}. Please ensure the file exists at this exact path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}. Check file format or encoding.")
        return None

# --- Model Loading ---
@st.cache_resource
def load_model_and_preprocessor():
    """
    Loads the trained model and preprocessor using joblib.
    Includes fallback to Mock objects if files are missing.
    """
    model = None
    preprocessor = None
    
    # 1. Load the Model
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"Successfully loaded model: {MODEL_PATH}")
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}. Using a MOCK model.")
    except Exception as e:
        st.error(f"Error loading model {MODEL_PATH}: {e}. Using a MOCK model.")

    # 2. Load the Preprocessor (if separate)
    if PREPROCESSOR_PATH and PREPROCESSOR_PATH != MODEL_PATH:
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            st.success(f"Successfully loaded preprocessor: {PREPROCESSOR_PATH}")
        except FileNotFoundError:
            st.warning(f"Preprocessor file not found: {PREPROCESSOR_PATH}. Using MOCK preprocessor logic.")
        except Exception as e:
            st.warning(f"Error loading preprocessor {PREPROCESSOR_PATH}: {e}. Using MOCK preprocessor logic.")

    # 3. Fallback to Mock Objects if loading failed
    if model is None or preprocessor is None:
        st.warning("One or more essential files failed to load. Falling back to Mock Model/Preprocessor.")
        
        class MockModel:
            """Dummy model for fallback."""
            def predict_proba(self, X):
                # Mock logic for demonstration: higher risk if older and higher balance
                age = X[0, 3] 
                balance = X[0, 5]
                risk_score = (age / 100) * (balance / 200000)
                prob_churn = np.clip(0.1 + risk_score, 0.05, 0.95)
                return np.array([[1 - prob_churn, prob_churn]])

        class MockPreprocessor:
            """Dummy preprocessor to match the expected feature structure."""
            def transform(self, df):
                country_map = {'France': 0, 'Spain': 1, 'Germany': 2}
                gender_map = {'Female': 0, 'Male': 1}
                
                X_raw = df.copy() 
                X_raw['country_encoded'] = X_raw['Country'].map(country_map).fillna(0)
                X_raw['gender_encoded'] = X_raw['Gender'].map(gender_map).fillna(0)
                
                # These 10 features must match the training set features and order
                X_transformed = X_raw[[
                    'CreditScore', 'country_encoded', 'gender_encoded', 
                    'Age', 'Tenure', 'Balance', 'ProductsNumber', 
                    'CreditCard', 'ActiveMember', 'EstimatedSalary'
                ]].values
                
                return X_transformed

        if model is None:
            model = MockModel()
        if preprocessor is None:
            preprocessor = MockPreprocessor()
            
    return model, preprocessor

# --- Prediction Function ---
def predict_churn(model, preprocessor, input_data):
    """
    Performs the full prediction workflow: Preprocess -> Predict.
    """
    # 1. Create a DataFrame from the input data (single row)
    input_df = pd.DataFrame([input_data])
    
    try:
        # 2. Preprocess the input data using the loaded preprocessor
        X_processed = preprocessor.transform(input_df)
        
        # 3. Predict probability
        proba = model.predict_proba(X_processed)[0] # [P(0), P(1)]
        
        # 4. Determine prediction class
        prediction = 1 if proba[1] > 0.5 else 0
        
        return prediction, proba[1] # Return class and churn probability
    
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("Check if your model/preprocessor successfully loaded and if the feature names/order are correct.")
        return -1, 0.0

# --- Load Data and Model on App Startup ---
df = load_data(FILE_PATH)
model, preprocessor = load_model_and_preprocessor()

# --- MAIN APP LAYOUT ---

st.title("🏦 Bank Customer Churn Prediction Dashboard")
st.markdown("Use this tool to analyze the dataset and predict the likelihood of a customer churning.")

if df is None:
    st.stop()

# --- SIDEBAR (User Input for Prediction) ---
with st.sidebar:
    st.header("👤 Customer Details for Prediction")
    
    # Numerical Inputs
    credit_score_input = st.slider("Credit Score", min_value=300, max_value=850, value=650, step=10)
    age_input = st.slider("Age", min_value=18, max_value=100, value=35, step=1)
    tenure_input = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
    balance_input = st.number_input("Balance ($)", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
    estimated_salary_input = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)
    products_number_input = st.slider("Number of Products", min_value=1, max_value=4, value=1, step=1)

    st.markdown("---")
    
    # Categorical/Binary Inputs
    try:
        country_options = df['country'].unique()
        gender_options = df['gender'].unique()
    except KeyError:
        country_options = ['France', 'Spain', 'Germany']
        gender_options = ['Male', 'Female']
        
    country_input = st.selectbox("Country", country_options)
    gender_input = st.selectbox("Gender", gender_options)
        
    credit_card_check = st.selectbox("Has Credit Card?", ["Yes", "No"]) == "Yes"
    active_member_check = st.selectbox("Is Active Member?", ["Yes", "No"]) == "Yes"
    
    # Convert bools to integers (0 or 1)
    credit_card_val = 1 if credit_card_check else 0
    active_member_val = 1 if active_member_check else 0
    
    # Prediction Button
    predict_button = st.button("Predict Churn Risk", type="primary")

# --- MAIN PAGE TABS ---
tab1, tab2 = st.tabs(["📊 Data Exploration", "🤖 Churn Prediction"])

with tab1:
    st.header("Dataset Overview")
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

    if 'churn' in df.columns:
        st.subheader("Churn Distribution")
        churn_counts = df['churn'].value_counts()
        churn_df = pd.DataFrame({
            'Churn': ['No Churn (0)', 'Churn (1)'], 
            'Count': churn_counts.values
        })
        st.bar_chart(churn_df.set_index('Churn'))
    
    st.subheader("Feature Distributions")
    
    if 'age' in df.columns:
        st.markdown("#### Age Distribution (Histogram)")
        
        # Using Altair to create a proper histogram with bins
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('age', bin=alt.Bin(maxbins=20), title='Age Bins'),
            y=alt.Y('count()', title='Number of Customers'),
            tooltip=['age', 'count()']
        ).properties(
            height=300
        ).interactive() 
        
        st.altair_chart(chart, use_container_width=True)
    
    if 'country' in df.columns:
        st.markdown("#### Country Distribution")
        st.bar_chart(df['country'].value_counts())

with tab2:
    st.header("Individual Customer Prediction")
    
    if predict_button:
        # 1. Assemble Input Dictionary
        # Note: Keys are capitalized to match how the input is received in the mock preprocessor
        input_data = {
            'CreditScore': credit_score_input,
            'Country': country_input,
            'Gender': gender_input,
            'Age': age_input,
            'Tenure': tenure_input,
            'Balance': balance_input,
            'ProductsNumber': products_number_input,
            'CreditCard': credit_card_val,
            'ActiveMember': active_member_val,
            'EstimatedSalary': estimated_salary_input,
        }

        # 2. Get Prediction
        final_prediction, churn_probability = predict_churn(model, preprocessor, input_data)
        
        # 3. Display Results
        
        st.subheader("Input Summary")
        st.dataframe(pd.DataFrame(input_data, index=[0]))
        
        st.subheader("Prediction Result")
        
        col_pred, col_prob = st.columns(2)
        
        # Display Prediction Class
        with col_pred:
            if final_prediction == 1:
                st.error(f"**PREDICTION: LIKELY TO CHURN**")
            elif final_prediction == 0:
                st.success(f"**PREDICTION: UNLIKELY TO CHURN**")
            else:
                st.warning("**PREDICTION FAILED** - Check Console for Details")

        # Display Probability
        with col_prob:
            churn_percentage = churn_probability * 100
            st.metric(label="Churn Probability", value=f"{churn_percentage:.2f}%")
            
        # Customize progress bar color based on risk
        st.markdown(f"""
        <style>
            .stProgress > div > div > div > div {{
                background-color: {'#e53e3e' if churn_probability > 0.5 else '#38a169'};
            }}
        </style>
        """, unsafe_allow_html=True)
            
        # Display Probability Progress Bar
        st.progress(churn_probability)
        
        # Additional Advice
        if churn_probability > 0.75:
            st.toast("High Risk! Immediate intervention is recommended.", icon="🔥")
        elif churn_probability > 0.5:
            st.toast("Moderate Risk. Monitor customer activity closely.", icon="⚠️")
        else:
            st.toast("Low Risk. Customer seems stable.", icon="✅")

# --- Footer/Model Info ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Model File: **{MODEL_PATH}**")
st.sidebar.caption("Remember to keep your data and model files in the specified paths.")
