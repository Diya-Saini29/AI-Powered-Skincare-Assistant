# 🧴 ClearSkin AI — Acne Detection & Skincare Assistant

> A full-stack AI web application that classifies acne type from an uploaded image using Transfer Learning, integrates real-time UV Index data, and delivers personalized skincare guidance — all in under 200ms.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey) ![CNN](https://img.shields.io/badge/Model-Transfer%20Learning-green) ![Classes](https://img.shields.io/badge/Classes-5%20Acne%20Types-red)

---

## 🔍 What It Does

Most skincare apps treat all acne the same. ClearSkin AI differentiates between 5 clinical acne types and maps them to the correct treatment track — inflammatory vs. non-inflammatory — before generating advice. It also pulls live UV Index data for the user's environment, since UV exposure directly affects skin condition and product safety.

**Upload a photo → get a clinical classification → receive UV-aware skincare guidance → done in <200ms.**

---

## ⚡ Key Results

| Metric | Value |
|---|---|
| Acne Classes | 5 (Blackheads, Whiteheads, Papules, Pustules, Cysts) |
| Grouped Classification Accuracy | **87.75%** |
| Clinical Grouping | Inflammatory vs. Non-Inflammatory |
| Inference Response Time | **< 200ms** |
| UV Confidence Guardrail | 65% threshold before UV advice is shown |
| Training Images | 1,000+ clinical skin images |
| Data Split | 80% train / 10% validation / 10% test |

> **Note on confidence display:** The percentage shown in the UI is the Softmax probability for the predicted class on that specific image — not the overall model accuracy. Overall grouped classification accuracy is 87.75%, which is the correct metric to evaluate this model's performance.

---

## 🧠 How It Works


<img width="904" height="761" alt="image" src="https://github.com/user-attachments/assets/79eb8c3c-fbfb-4c02-b8e7-65c6e4abf19e" />


**Stage 1 — Acne Classification:**
A Transfer Learning model (MobileNetV2 backbone, frozen) processes the 256×256 RGB image. The final dense layers classify into one of 5 acne types: Blackheads, Whiteheads, Papules, Pustules, or Cysts.

**Stage 2 — Clinical Grouping:**
The 5 classes are mapped to clinical treatment tracks:
- 🟡 **Non-Inflammatory** — Blackheads, Whiteheads → gentler, exfoliant-focused routine
- 🔴 **Inflammatory** — Papules, Pustules, Cysts → active treatment, avoid harsh products

**Stage 3 — UV Index Integration:**
Live UV data is fetched from the Open-Meteo API. UV advice is only shown when model confidence exceeds **65%**, preventing low-confidence predictions from generating unsafe skincare guidance.

**Stage 4 — Report Generation:**
A full skincare report combining acne type, clinical group, UV risk level, and product recommendations is generated and displayed in under 200ms.

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| ML Model | TensorFlow / Keras, Transfer Learning (MobileNetV2) |
| Backend | Python, Flask, REST API |
| Frontend | HTML, CSS (index.html, result.html) |
| External API | Open-Meteo (real-time UV Index) |
| Dataset | AcneDataset (public, 5 classes, 1,000+ images) |
| Notebook | Jupyter (training + experimentation) |

---

## 📁 Repository Structure

```
├── app.ipynb                          # Main Flask application (notebook)
├── recommender.py                     # Skincare recommendation logic
├── skincare_classifier_model1.ipynb   # Baseline CNN training (v1)
├── skincare_classifier_TL_model2.ipynb # Transfer Learning model (v2, used in prod)
├── index.html                         # Upload interface
├── result.html                        # Results display page
├── requirements.txt
└── .gitignore
```

> ⚠️ Model weight files (`.keras`) are not tracked in this repo due to size. Train using the provided notebooks or contact me for the weights.

---

## 🚀 Setup & Run

### Installation

```bash
git clone https://github.com/Diya-Saini29/AI-Powered-Skincare-Assistant.git
cd AI-Powered-Skincare-Assistant
pip install -r requirements.txt
```

### Train the Model
Open and run `skincare_classifier_TL_model2.ipynb` in Jupyter to train and save the model weights.

### Run the App
```bash
# If running as a Python script
python app.py

# Or open app.ipynb in Jupyter and run all cells
```
Then open `http://localhost:5000` in your browser.

---

## 🧪 Model Training Details

| Parameter | Value |
|---|---|
| Base Model | MobileNetV2 (ImageNet weights, frozen backbone) |
| Input Size | 256 × 256 × 3 (RGB, normalized /255.0) |
| Custom Head | Global Average Pooling → Dense → Dropout → Softmax |
| Dataset | AcneDataset (public) |
| Classes | 5 (Blackheads, Whiteheads, Papules, Pustules, Cysts) |
| Train/Val/Test Split | 80% / 10% / 10% |
| Grouped Accuracy | **87.75%** (Inflammatory vs. Non-Inflammatory) |

Two model iterations were developed:
- `model1` — Custom CNN baseline (lower accuracy, higher overfitting)
- `model2` — Transfer Learning with MobileNetV2 (production model, 87.75% grouped accuracy)

---

## 👩‍💻 Author

**Diya Saini** — AI/ML Undergraduate, Thapar Institute of Engineering and Technology
[LinkedIn](https://linkedin.com/in/diya-saini-m) · [GitHub](https://github.com/Diya-Saini29) · sainidiya889@gmail.com
