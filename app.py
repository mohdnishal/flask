from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load all files
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
num_imputer = joblib.load('num_imputer.pkl')
cat_imputer = joblib.load('cat_imputer.pkl')
cat_encoder = joblib.load('cat_encoder.pkl')
X_columns = joblib.load('columns.pkl')
college_country_map = joblib.load('college_country_map.pkl')

cat_cols = ['Country', 'Course', 'Internship&placement', 'Partime job', 'Stayback', 'Higher studies possible']
num_cols = ['Plustwo', 'Fees per year', 'IELTS(9)', 'TOEFL(120)', 'PTE(90)']

def parse_numeric(value):
    if pd.isna(value):
        return np.nan
    value = str(value).replace('INR', '').replace('$', '').replace(',', '').strip()
    if 'L' in value.upper():
        return float(value.upper().replace('L', '').strip()) * 1e5
    if '%' in value:
        value = value.replace('%', '')
    try:
        return float(value)
    except:
        return np.nan

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)

        if not data or "Country" not in data:
            return jsonify({"error": "Missing 'Country' in request"}), 400

        user_country = data.get('Country')

        user_df = pd.DataFrame([{
            'Country': data['Country'],
            'Course': data['Course'],
            'IELTS(9)': parse_numeric(data['IELTS']),
            'Plustwo': parse_numeric(data['Plustwo']),
            'TOEFL(120)': parse_numeric(data.get('TOEFL', 0)),
            'PTE(90)': parse_numeric(data.get('PTE', 0)),
            'Fees per year': parse_numeric(data['Fees']),
            'Internship&placement': data.get('Internship', 'No'),
            'Partime job': data.get('Partime', 'No'),
            'Stayback': data.get('Stayback', 'No'),
            'Higher studies possible': data.get('HigherStudies', 'No')
        }])

        user_df[num_cols] = num_imputer.transform(user_df[num_cols])
        cat_transformed = cat_imputer.transform(user_df[cat_cols])
        cat_encoded = cat_encoder.transform(cat_transformed)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_encoder.get_feature_names_out(cat_cols))

        final_input = pd.concat([user_df[num_cols].reset_index(drop=True), cat_df], axis=1)
        final_input = final_input.reindex(columns=X_columns, fill_value=0)

        probs = model.predict_proba(final_input)[0]
        colleges = label_encoder.inverse_transform(np.arange(len(probs)))
        ranked = sorted(zip(colleges, probs), key=lambda x: x[1], reverse=True)

        filtered = [r for r in ranked if college_country_map.get(r[0]) == user_country]
        if not filtered:
            filtered = ranked
        top_3 = [{"college": str(c), "confidence": float(round(p * 100, 2))} for c, p in filtered[:3]]

        return jsonify({"top_5_colleges": top_3})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)
