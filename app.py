from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Initialize Flask app
app = Flask(__name__)

# Global variables
xgb_model = None
label_encoder = None
cat_imputer = None
cat_encoder = None
X_train = None
cat_cols = None
num_cols = None
college_country_map = {}

def parse_numeric_value(value):
    """Convert numeric values from various formats"""
    if pd.isna(value):
        return np.nan
    value = str(value).strip().replace('INR', '').replace('$', '').replace(',', '')
    if 'L' in value.upper():
        try:
            return float(value.replace('L', '').strip()) * 100000
        except ValueError:
            return np.nan
    value = value.replace('%', '')
    try:
        return float(value)
    except ValueError:
        return np.nan

def preprocess_data(data):
    global cat_cols, num_cols
    
    # Drop rows with missing College
    data = data.dropna(subset=['College'])
    
    # Convert specified columns
    columns_to_convert = ['Plustwo', 'Fees per year', 'IELTS(9)', 'TOEFL(120)', 'PTE(90)']
    for col in columns_to_convert:
        if col in data.columns:
            data[col] = data[col].apply(parse_numeric_value)
    
    # Identify numeric and categorical columns
    num_cols = [col for col in columns_to_convert if col in data.columns]
    cat_cols = ['Country', 'Course', 'Internship&placement', 'Partime job', 'Stayback', 'Higher studies possible']
    cat_cols = [col for col in cat_cols if col in data.columns]
    
    # Filter colleges with enough samples
    y_counts = data['College'].value_counts()
    valid_classes = y_counts[y_counts > 5].index
    data = data[data['College'].isin(valid_classes)]
    
    return data

def encode_and_resample(X, y):
    global cat_imputer, cat_encoder, label_encoder

    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    cat_data = cat_imputer.fit_transform(X[cat_cols])
    X_encoded = cat_encoder.fit_transform(cat_data)
    X_encoded_df = pd.DataFrame(X_encoded, columns=cat_encoder.get_feature_names_out(cat_cols))

    X_final = pd.concat([X[num_cols].reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_final, y_encoded)
    except ValueError:
        print("⚠️ SMOTE resampling failed. Using original data.")
        X_resampled, y_resampled = X_final, y_encoded

    return X_resampled, y_resampled

def initialize_model():
    global xgb_model, X_train, cat_imputer, cat_encoder, label_encoder, college_country_map

    try:
        # Load and preprocess data
        data = pd.read_csv("data.csv")
        data = preprocess_data(data)

        # Store country mapping
        college_country_map = dict(zip(data['College'], data['Country']))

        X, y = data[num_cols + cat_cols], data['College']
        X_resampled, y_resampled = encode_and_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model Initialized! Accuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        xgb_model = None

# Initialize model at startup
initialize_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if xgb_model is None or cat_imputer is None or cat_encoder is None:
            return jsonify({"error": "Model is not initialized. Try again later."}), 500

        # Parse input JSON
        data = request.get_json()

        user_country = data.get('Country', '')

        # Validate required fields
        required_fields = ['Country', 'Course', 'IELTS', 'Plustwo', 'Fees']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400


        user_input_df = pd.DataFrame([{  
            'Country': data['Country'],
            'Course': data['Course'],
            'IELTS(9)': float(data['IELTS']),
            'Plustwo': float(data['Plustwo']),
            'TOEFL(120)': float(data.get('TOEFL', 0)),
            'PTE(90)': float(data.get('PTE', 0)),
            'Fees per year': float(data['Fees']),
            'Internship&placement': data.get('Internship', 'No'),
            'Partime job': data.get('Partime', 'No'),
            'Stayback': data.get('Stayback', 'No'),
            'Higher studies possible': data.get('HigherStudies', 'No')
        }])

        # Preprocess input
        user_input_df[cat_cols] = cat_imputer.transform(user_input_df[cat_cols])
        user_encoded = cat_encoder.transform(user_input_df[cat_cols])
        user_encoded_df = pd.DataFrame(user_encoded, columns=cat_encoder.get_feature_names_out(cat_cols))
        user_final = pd.concat([user_input_df[num_cols].reset_index(drop=True), user_encoded_df], axis=1)
        user_final = user_final.reindex(columns=X_train.columns, fill_value=0)

        # Predict
        probabilities = xgb_model.predict_proba(user_final)[0]
        college_names = label_encoder.inverse_transform(np.arange(len(probabilities)))
        college_probs = list(zip(college_names, probabilities))

        # 🎯 Country priority filtering
        filtered = [(college, prob) for college, prob in college_probs
                    if college_country_map.get(college) == user_country]

        # Fallback to global top 5 if no country-specific match
        if not filtered:
            filtered = college_probs

        top_5 = sorted(filtered, key=lambda x: x[1], reverse=True)[:3]
        top_5_response = [{"college": c, "confidence": float(round(p * 100, 2))} for c, p in top_5]

        return jsonify({'top_5_colleges': top_5_response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=10000)
