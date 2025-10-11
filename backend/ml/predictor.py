import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Define the directory where the ML models will be stored
ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ML_DIR, 'student_model.pkl')
SCALER_PATH = os.path.join(ML_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(ML_DIR, 'feature_columns.joblib')

# =====================================================================
# PART 1: MODEL TRAINING AND PREPROCESSING (Run this part once)
# =====================================================================

def preprocess_training_data(df):
    """
    Cleans and prepares the initial training data.
    Extracted directly from your app.py.
    """
    df_clean = df.copy()

    # Handle missing values with strategy based on missing percentage
    numeric_cols = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            missing_pct = df_clean[col].isnull().sum() / len(df_clean) * 100
            if missing_pct < 5:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # Handle categorical columns
    categorical_cols = ['Gender', 'ExtracurricularActivities', 'ParentalSupport']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown').astype(str)

    # Convert Online Classes Taken to numeric
    if 'Online Classes Taken' in df_clean.columns:
        df_clean['Online Classes Taken'] = df_clean['Online Classes Taken'].map({
            'True': 1, 'False': 0, True: 1, False: 0, 'Yes': 1, 'No': 0
        }).fillna(0)

    # Create target variable
    if 'PreviousGrade' in df_clean.columns:
        threshold = 70
        df_clean['Pass'] = (df_clean['PreviousGrade'] >= threshold).astype(int)
    else:
        raise ValueError("Cannot create target variable - PreviousGrade column not found")

    # Drop unnecessary columns
    columns_to_drop = ['StudentID', 'Name', 'FinalGrade', 'Study Hours', 'Attendance (%)']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])

    # Encode categorical variables
    for col in categorical_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])

    # Select final features and drop rows with any remaining NaN values
    required_features = ['Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 
                         'ExtracurricularActivities', 'ParentalSupport', 'Online Classes Taken', 'Pass']
    available_features = [col for col in required_features if col in df_clean.columns]
    df_final = df_clean[available_features].dropna()
    
    return df_final

def train_and_save_model(csv_path="../assets/student_performance_updated_1000.csv"):
    """
    Loads data, preprocesses it, trains the model and scaler,
    and saves them to disk.
    """
    print("Starting model training...")
    df = pd.read_csv(csv_path)
    
    # 1. Preprocess the data
    df_processed = preprocess_training_data(df)
    
    X = df_processed.drop(columns=['Pass'])
    y = df_processed['Pass']
    
    # 2. Train the Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    
    # 3. Train the Logistic Regression Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y)
    
    # 4. Save the model, scaler, and feature columns
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH) # Save the column order
    
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Scaler saved to {SCALER_PATH}")
    print(f"✅ Feature list saved to {FEATURES_PATH}")

# =====================================================================
# PART 2: PREDICTOR CLASS (Use this in your Django app)
# =====================================================================

class Predictor:
    def __init__(self):
        """Loads the pre-trained model, scaler, and feature list."""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_columns = joblib.load(FEATURES_PATH)
        except FileNotFoundError:
            raise FileNotFoundError("Model or scaler not found. Please run train_and_save_model() first.")

    def clean_input_data(self, input_data):
        """
        Cleans and standardizes a single record of input data.
        'input_data' should be a dictionary.
        """
        # Create a DataFrame from the dictionary
        cleaned_df = pd.DataFrame([input_data])
        
        # This part is adapted from the 'clean_uploaded_data' function
        # Direct Mapping for Categorical and Boolean columns
        if 'ExtracurricularActivities' in cleaned_df.columns:
            cleaned_df['ExtracurricularActivities'] = cleaned_df['ExtracurricularActivities'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
        if 'ParentalSupport' in cleaned_df.columns:
            cleaned_df['ParentalSupport'] = cleaned_df['ParentalSupport'].astype(str).str.strip().str.lower().map({'high': 2, 'medium': 1, 'low': 0}).fillna(1).astype(int)
        if 'Gender' in cleaned_df.columns:
            cleaned_df['Gender'] = cleaned_df['Gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0, 'other': 2}).fillna(2).astype(int)
        
        # Convert numeric and clip to valid ranges
        numeric_cols = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade']
        for col in numeric_cols:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        if 'AttendanceRate' in cleaned_df.columns:
            cleaned_df['AttendanceRate'] = cleaned_df['AttendanceRate'].clip(0, 100)
        if 'StudyHoursPerWeek' in cleaned_df.columns:
            cleaned_df['StudyHoursPerWeek'] = cleaned_df['StudyHoursPerWeek'].clip(0, 80)
        if 'PreviousGrade' in cleaned_df.columns:
            cleaned_df['PreviousGrade'] = cleaned_df['PreviousGrade'].clip(0, 100)
        
        # Handle 'Online Classes Taken' which might be numeric or boolean
        if 'Online Classes Taken' in cleaned_df.columns:
            cleaned_df['Online Classes Taken'] = (pd.to_numeric(cleaned_df['Online Classes Taken'], errors='coerce').fillna(0) > 0).astype(int)

        # Ensure all required columns are present and in the correct order
        for col in self.feature_columns:
            if col not in cleaned_df.columns:
                cleaned_df[col] = 0 # Or a more sensible default
        
        return cleaned_df[self.feature_columns]

    def predict(self, input_data):
        """
        Takes a dictionary of student survey data, cleans it,
        and returns a prediction.
        """
        # 1. Clean the input data
        cleaned_df = self.clean_input_data(input_data)
        
        # 2. Scale the data
        scaled_data = self.scaler.transform(cleaned_df)
        
        # 3. Make prediction
        prediction_val = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        result = 'PASS' if prediction_val == 1 else 'FAIL'
        confidence = probabilities.max()
        
        return result, confidence

# =====================================================================
# HOW TO USE THIS FILE
# =====================================================================

if __name__ == '__main__':
    # This block will only run when you execute `python predictor.py` directly
    # from your terminal. It will not run when imported by Django.

    # STEP 1: Train and save the model (only needs to be done once)
    print("Running the training script...")
    train_and_save_model("../assets/student_performance_updated_1000.csv")
    print("\n" + "="*50 + "\n")

    # STEP 2: Test the predictor class
    print("Testing the Predictor class with sample data...")
    
    # Sample data for a student likely to pass
    passing_student_data = {
        'Gender': 'Female',
        'AttendanceRate': 95,
        'StudyHoursPerWeek': 15,
        'PreviousGrade': 88,
        'ExtracurricularActivities': 'yes',
        'ParentalSupport': 'high',
        'Online Classes Taken': 1
    }

    # Sample data for a student likely to fail
    failing_student_data = {
        'Gender': 'Male',
        'AttendanceRate': 70,
        'StudyHoursPerWeek': 5,
        'PreviousGrade': 55,
        'ExtracurricularActivities': 'no',
        'ParentalSupport': 'low',
        'Online Classes Taken': 0
    }
    
    # Create an instance of the predictor
    predictor = Predictor()
    
    # Make a prediction
    result, confidence = predictor.predict(passing_student_data)
    print(f"Prediction for passing student: {result} (Confidence: {confidence:.2%})")

    result, confidence = predictor.predict(failing_student_data)
    print(f"Prediction for failing student: {result} (Confidence: {confidence:.2%})")