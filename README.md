# Spam-Prediction-Classifier

# 📧 Spam Mail Classification using Machine Learning

A high-accuracy spam detection system built using **Natural Language Processing (NLP)** and **scikit-learn**, capable of classifying SMS/email messages as **Spam** or **Ham (Not Spam)**.  
This upgraded version is optimized to run in restricted environments with **no external library installations**, while still achieving **97–98.5% accuracy**.

---

## 🚀 Features

- Advanced text preprocessing (cleaning, normalization)
- TF-IDF feature extraction with **1–2 gram support**
- **Linear Support Vector Machine (LinearSVC)** — top performer for text classification
- Hyperparameter tuning using **GridSearchCV**
- Class imbalance handled with **class weights**
- Detailed evaluation: accuracy, precision, recall, F1-score
- **Confusion matrix visualization**
- Fully portable — requires **only scikit-learn**

---

## 📂 Dataset

The project uses the `mail_data.csv` dataset containing **5,572 labeled messages**.

| Column     | Description               |
|------------|---------------------------|
| Category   | “spam” or “ham”           |
| Message    | Raw SMS/email text        |

---

## 🧠 Workflow Overview

### **1. Data Loading**
- Load CSV dataset into a pandas DataFrame  
- Drop null or missing values  

### **2. Text Cleaning**
Performed using regex and string normalization:
- Lowercasing  
- Removing URLs  
- Removing numbers  
- Removing punctuation  
- Removing extra spaces  

### **3. Label Encoding**
- `ham` → **0**  
- `spam` → **1**

### **4. Train/Test Split**
- 80% training  
- 20% testing  

### **5. Feature Extraction (TF-IDF)**
- Max 5,000 features  
- 1–2 n-grams  
- English stopwords removal  
- Sublinear term frequency scaling  

### **6. Model Selection — Linear SVM**
LinearSVC is chosen because:
- Excellent performance on text
- Handles high-dimensional sparse data effectively
- Very fast and memory-efficient  
- High precision + high recall for spam detection  

### **7. Hyperparameter Tuning**
Performed via GridSearchCV on SVM parameter **C**.

### **8. Evaluation**
Includes:
- Accuracy score  
- Precision/Recall/F1-score  
- **Confusion matrix heatmap**  

---

## 📊 Performance

With TF-IDF + LinearSVC, the model consistently achieves:

| Metric | Score |
|--------|-------|
| **Accuracy** | 97–98.5% |
| **Spam Recall** | High |
| **Precision** | Very High |
| **F1 Score** | Excellent |

The model is stable and generalizes well to unseen messages.

---

## 🖥️ Usage

### **1. Train the Model**
Run the training script or notebook:

```bash
python train.py
