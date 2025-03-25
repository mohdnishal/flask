from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

data = pd.read_csv("data.csv")

data = data.dropna(subset=['College'])

if 'Plustwo' in data.columns:
    data['Plustwo'] = data['Plustwo'].str.rstrip('%').astype(float)
if 'Fees per year' in data.columns:
    data['Fees per year'] = data['Fees per year'].str.replace(',', '').str.extract(r'([0-9]+\.?[0-9]*)')[0].astype(float)
if 'Duration of course' in data.columns:
    data['Duration of course'] = data['Duration of course'].str.extract(r'([0-9]+\.?[0-9]*)')[0].astype(float)

num_cols = ['IELTS(9)', 'TOEFL(120)', 'PTE(90)', 'Plustwo', 'Fees per year', 'Duration of course']
cat_cols = ['Country', 'Course', 'Internship&placement', 'Partime job', 'Stayback', 'Higher studies possible']
num_cols = [col for col in num_cols if col in data.columns]
cat_cols = [col for col in cat_cols if col in data.columns]

X = data[num_cols + cat_cols]
y = data['College']

y_counts = y.value_counts()
valid_classes = y_counts[y_counts > 1].index
data = data[data['College'].isin(valid_classes)]

X = data[num_cols + cat_cols]
y = data['College']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=8, 
        random_state=42
    ))
])

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['Country', 'Course', 'IELTS', 'Plustwo', 'Duration', 'Fees']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400

        user_data = pd.DataFrame([{  
            'Country': data['Country'],
            'Course': data['Course'],
            'IELTS(9)': float(data['IELTS']),
            'Plustwo': float(data['Plustwo']),
            'TOEFL(120)': float(data.get('TOEFL', 0)),
            'PTE(90)': float(data.get('PTE', 0)),
            'Fees per year': float(data['Fees']),
            'Duration of course': float(data['Duration']),
            'Internship&placement': data.get('Internship', 'No'),
            'Partime job': data.get('Partime', 'No'),
            'Stayback': data.get('Stayback', 'No'),
            'Higher studies possible': data.get('HigherStudies', 'No')
        }])

        # Get probabilities for each college
        probabilities = xgb_model.predict_proba(user_data)[0]

        # Get top 5 indices based on highest probabilities
        top_5_indices = np.argsort(probabilities)[::-1][:5]
        
        # Get corresponding college names and their scores, ensuring float conversion
        top_5_colleges = [
            {"college": label_encoder.inverse_transform([idx])[0], "confidence": float(round(probabilities[idx] * 100, 2))}
            for idx in top_5_indices
        ]

        return jsonify({'top_5_colleges': top_5_colleges})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=10000)
