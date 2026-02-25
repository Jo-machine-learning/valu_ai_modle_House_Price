import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

np.random.seed(42)
n_samples = 500  # dataset أكبر وأدق

regions = ['New Cairo', 'Sheikh Zayed', '6th of October', 'Alexandria', 'Zamalek', 
           'Fifth Settlement', 'Maadi', 'New Administrative Capital', 'Faisal', 'Nasr City']
prices = [45000, 40000, 41000, 10000, 70000, 70000, 25000, 35000, 12000, 30000]  # تحديث 2026

data = {
    'region': np.random.choice(regions, n_samples),
    'area': np.random.uniform(50, 400, n_samples),
    'rooms': np.random.randint(1, 7, n_samples),
    'type': np.random.choice([0,1], n_samples, p=[0.7, 0.3]),  # 0: شقة، 1: فيلا
    'age': np.random.randint(0, 35, n_samples),
    'floor': np.random.randint(1, 15, n_samples),  # عامل إضافي: الطابق
}

df = pd.DataFrame(data)

# إضافة سعر المتر الأساسي مع تباين
df['price_per_meter'] = df['region'].map({r: p for r, p in zip(regions, prices)})
df['price_per_meter'] += np.random.normal(0, 5000, n_samples)

# تعديل السعر حسب العوامل (أكثر واقعية)
df['price_per_meter'] *= (1 + 0.05 * df['rooms'] - 0.02 * df['age'] + 0.1 * df['type'] + 0.01 * df['floor'])
df['price_per_meter'] = np.clip(df['price_per_meter'], 5000, 150000)

# ترميز المناطق
df['region_encoded'] = pd.Categorical(df['region']).codes

# المتغيرات والنتيجة
X = df[['region_encoded', 'area', 'rooms', 'type', 'age', 'floor']]
y = df['price_per_meter']

# نموذج أقوى
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)

# حفظ الملفات
with open('house_price_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)
region_codes = {name: i for i, name in enumerate(regions)}
with open('regions_v2.pkl', 'wb') as f:
    pickle.dump(region_codes, f)

print("✅ تم تدريب النموذج بنجاح!")
print("\n📊 متوسطات الأسعار حسب المنطقة:")
print(df.groupby('region')['price_per_meter'].mean().round(0))
print(f"\n🔥 دقة النموذج (R²): {model.score(X, y):.3f}")
print(f"📈 حجم الـDataset: {len(df)} عينة")
print("\n💾 الملفات جاهزة: house_price_model_v2.pkl & regions_v2.pkl")
