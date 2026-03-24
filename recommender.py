import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import requests
import urllib.parse
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. Load Model
model = keras.models.load_model('skincare_transfer_leaning_v2_26jan.keras')

# 2. Setup Database
groups = {
    'Non-Inflammatory': ['Blackheads', 'Whiteheads'],
    'Inflammatory': ['Cyst', 'Papules', 'Pustules']
}
class_names = ['Blackheads', 'Cyst', 'Papules', 'Pustules', 'Whiteheads']

recommendations = {
    'Non-Inflammatory': {
        'Diagnosis': 'Non-Inflammatory Congested Acne',
        'Description': 'Mainly clogged pores (blackheads/whiteheads).',
        'Primary_Treatments': ['Salicylic Acid (BHA)', 'Adapalene'],
        'Morning_Routine': 'Use a gentle BHA cleanser and SPF 30+.',
        'Evening_Routine': 'Double cleanse. Apply Adapalene gel.',
        'Supportive_Ingredients': ['Niacinamide', 'Hyaluronic Acid'],
        'Professional_Advice': 'Retinoids can cause purging; start slow.',
        'Expert_Tip': 'Always apply sunscreen (SPF 30+)!',
        'Red_Flags': 'Consult a pro for deep painful lumps.'
    },
    'Inflammatory': {
        'Diagnosis': 'Active Inflammatory Breakout',
        'Description': 'Red, swollen bumps with bacteria.',
        'Primary_Treatments': ['Benzoyl Peroxide', 'Azelaic Acid'],
        'Morning_Routine': 'Use a soap-free cleanser and Ceramides.',
        'Evening_Routine': 'Spot treat with Benzoyl Peroxide.',
        'Supportive_Ingredients': ['Ceramides', 'Centella Asiatica'],
        'Professional_Advice': 'Benzoyl Peroxide can bleach fabrics.',
        'Expert_Tip': 'Change your pillowcase every 2 days.',
        'Red_Flags': 'Do not squeeze! Seek a doctor for cystic acne.'
    }
}

def get_live_uv_data():
    try:
        geo_data = requests.get('http://ip-api.com/json/').json()
        city = geo_data.get('city', 'Unknown')
        lat, lon = geo_data.get('lat', 0), geo_data.get('lon', 0)
        res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=uv_index_max&timezone=auto").json()
        uv_index = res['daily']['uv_index_max'][0]
        return uv_index, city
    except:
        return 0, "Unknown"

def get_skincare_roadmap(img_path):
    try:
        img = load_img(img_path, target_size=(256, 256))
        img_array = img_to_array(img) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))
        confidence = np.max(preds) * 100
        if confidence < 65.0: return None
        
        specific_type = class_names[np.argmax(preds)]
        uv_index, city = get_live_uv_data()
        clinical_group = next(g for g, classes in groups.items() if specific_type in classes)
        data = recommendations[clinical_group]

        return {
        'diagnosis': data['Diagnosis'],
         'summary': data['Description'],
         'morning': data['Morning_Routine'],
         'evening': data['Evening_Routine'],
          'primary': data['Primary_Treatments'], # New
         'supportive': data['Supportive_Ingredients'],
         'pro_advice': data['Professional_Advice'],
         'lifestyle': data['Expert_Tip'],
          'safety': data['Red_Flags'],
          'uv_index': uv_index,
         'city': city,
         'confidence': f"{confidence:.1f}",
         'search_query': f"buy {' '.join(data['Supportive_Ingredients'][:2])} skincare products"
         }
    except Exception as e:
        print(f"Error: {e}")
        return None