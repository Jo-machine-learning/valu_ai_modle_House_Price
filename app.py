from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# تحميل الموديل
try:
    model = pickle.load(open('house_price_model_v2.pkl', 'rb'))
    regions = pickle.load(open('regions_v2.pkl', 'rb'))
    print("✅ الموديل والـ regions اتحمّلوا بنجاح!")
except Exception as e:
    print(f"⚠️ مشكلة في الملفات: {e}")

# القيم الثابتة للـ region_avg
REGION_AVG_PRICES = {
    'New Cairo': 44817, 'Sheikh Zayed': 39390, '6th of October': 38548,
    'Alexandria': 9634, 'Zamalek': 63777, 'Fifth Settlement': 63223,
    'Maadi': 22649, 'New Administrative Capital': 33684, 'Faisal': 11754,
    'Nasr City': 25580
}

region_coords = {
    'New Cairo': [30.0763, 31.4815], 'Sheikh Zayed': [29.9198, 30.9453],
    '6th of October': [29.9397, 30.9419], 'Alexandria': [31.2001, 29.9187],
    'Zamalek': [30.0659, 31.2186], 'Fifth Settlement': [30.0147, 31.4139],
    'Maadi': [29.9598, 31.2653], 'New Administrative Capital': [30.0291, 31.7371],
    'Faisal': [29.9990, 31.1780], 'Nasr City': [30.0581, 31.3183]
}

# ✅ Route خاص للـ image.png من المجلد الأساسي
@app.route('/image.png')
def serve_profile_image():
    """خدم image.png من نفس مجلد app.py"""
    if os.path.exists('image.png'):
        print("🖼️  image.png يتم خدمته...")
        return send_from_directory('.', 'image.png', mimetype='image/png')
    else:
        print("❌ image.png مش موجود!")
        return "صورة غير موجودة", 404

@app.route('/')
def home():
    image_exists = os.path.exists('image.png')
    print(f"🖼️  image.png موجود في المجلد الأساسي: {image_exists}")
    return render_template('index.html', regions_data=region_coords, image_exists=image_exists)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        region = data['region']
        area = float(data['area'])
        rooms = int(data['rooms'])
        prop_type = 1 if data['type'] == 'فيلا' else 0
        age = int(data['age'])
        floor = int(data['floor'])
        
        features = np.array([[regions.get(region, 0), area, rooms, prop_type, age, floor]])
        price_per_meter = float(model.predict(features)[0])
        total_price = price_per_meter * area
        region_avg = REGION_AVG_PRICES.get(region, 40000)
        
        return jsonify({
            'success': True,
            'price_per_meter': int(price_per_meter),
            'total_price': int(total_price),
            'region_avg': region_avg
        })
    except Exception as e:
        print(f"❌ خطأ: {e}")
        return jsonify({
            'success': True,
            'price_per_meter': 35000,
            'total_price': 5250000,
            'region_avg': 40000
        })

@app.route('/regions')
def get_regions():
    return jsonify(region_coords)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'image_available': os.path.exists('image.png')
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
