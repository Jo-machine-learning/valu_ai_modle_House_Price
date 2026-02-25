import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np
from PIL import Image, ImageTk  # pip install pillow

# تحميل الموديل
with open('house_price_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)
with open('regions_v2.pkl', 'rb') as f:
    region_codes = pickle.load(f)

regions = sorted(region_codes.keys())

def predict_price():
    try:
        region = region_combo.get()
        area = float(area_entry.get())
        rooms = int(rooms_entry.get())
        prop_type = 0 if type_var.get() == "شقة" else 1
        age = int(age_entry.get())
        floor = int(floor_entry.get())
        
        features = np.array([[region_codes[region], area, rooms, prop_type, age, floor]])
        price_per_meter = model.predict(features)[0]
        total_price = price_per_meter * area
        
        result_label.config(text=f"سعر المتر: {price_per_meter:,.0f} جنيه\nالسعر الإجمالي: {total_price:,.0f} جنيه")
    except ValueError:
        messagebox.showerror("خطأ", "أدخل أرقام صحيحة!")

# الواجهة الرئيسية
root = tk.Tk()
root.title("🏠 نظام توقع سعر العقار في مصر 2026")
root.geometry("500x700")
root.configure(bg="#f0f8ff")  # أزرق فاتح
root.resizable(False, False)

# عنوان أنيق
title_label = tk.Label(root, text="توقع سعر بيتك في مصر", font=("Arial", 20, "bold"), 
                       bg="#f0f8ff", fg="#1e3a8a")
title_label.pack(pady=20)

# صورة العمارات (ضع image.jpg في المجلد)
try:
    img = Image.open("image.png")
    img = img.resize((450, 200), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=photo, bg="#f0f8ff")
    img_label.image = photo  # الحفاظ عليها
    img_label.pack(pady=10)
except:
    tk.Label(root, text="ضع صورة image.jpg (عمارات)", bg="#f0f8ff", fg="red").pack(pady=10)

# إطار الإدخال
frame = tk.Frame(root, bg="#e0f2fe", relief="ridge", bd=2)
frame.pack(pady=20, padx=20, fill="x")

ttk.Label(frame, text="المنطقة:", font=("Arial", 12)).pack(pady=10)
region_combo = ttk.Combobox(frame, values=regions, state="readonly", width=35, font=("Arial", 11))
region_combo.set(regions[0])
region_combo.pack(pady=5)

ttk.Label(frame, text="المساحة (م²):", font=("Arial", 12)).pack(pady=5)
area_entry = ttk.Entry(frame, width=35, font=("Arial", 11))
area_entry.pack(pady=5)

ttk.Label(frame, text="عدد الغرف:", font=("Arial", 12)).pack(pady=5)
rooms_entry = ttk.Entry(frame, width=35, font=("Arial", 11))
rooms_entry.pack(pady=5)

ttk.Label(frame, text="نوع العقار:", font=("Arial", 12)).pack(pady=5)
type_var = tk.StringVar(value="شقة")
type_frame = tk.Frame(frame, bg="#e0f2fe")
type_frame.pack(pady=5)
ttk.Radiobutton(type_frame, text="شقة", variable=type_var, value="شقة", width=10).pack(side="left", padx=10)
ttk.Radiobutton(type_frame, text="فيلا", variable=type_var, value="فيلا", width=10).pack(side="left")

ttk.Label(frame, text="عمر البناء (سنوات):", font=("Arial", 12)).pack(pady=5)
age_entry = ttk.Entry(frame, width=35, font=("Arial", 11))
age_entry.pack(pady=5)

ttk.Label(frame, text="الطابق:", font=("Arial", 12)).pack(pady=5)  # إضافة جديدة
floor_entry = ttk.Entry(frame, width=35, font=("Arial", 11))
floor_entry.pack(pady=5)

# زر التوقع ملون
predict_btn = tk.Button(root, text="🔮 توقع السعر الآن", command=predict_price, 
                       bg="#1e40af", fg="white", font=("Arial", 14, "bold"),
                       relief="raised", bd=3, pady=10, cursor="hand2")
predict_btn.pack(pady=30)

# نتيجة ملونة
result_label = tk.Label(root, text="أدخل البيانات واضغط التوقع!", font=("Arial", 14, "bold"), 
                        bg="#f0f8ff", fg="#059669")
result_label.pack(pady=20)

# تذييل
footer = tk.Label(root, text="بناءً على بيانات 2026 | Perplexity AI", font=("Arial", 10), 
                  bg="#f0f8ff", fg="#6b7280")
footer.pack(side="bottom", pady=10)

root.mainloop()
