# 🎭 Micro-Expression Analysis using Machine Learning

This project focuses on detecting and analyzing **micro-expressions** — subtle and involuntary facial expressions that reveal genuine human emotions.

---

## 🚀 PROJECT OVERVIEW

Micro-expressions occur within milliseconds and are difficult to detect with the human eye. This project applies machine learning techniques to classify emotional states from facial data.

---

## 🧠 KEY CONCEPT

This project highlights the challenge of detecting subtle emotional signals and compares different machine learning models and feature selection methods.

🔹 Raw features vs dimensionality reduction (PCA)  
🔹 Model performance comparison  
🔹 Trade-off between accuracy and complexity  

---

## ⚙️ TECHNOLOGIES USED

- 🐍 Python  
- 👁️ OpenCV  
- 🤖 Scikit-learn  
- 📊 NumPy, Pandas, Matplotlib  

---

## 📂 PROJECT STRUCTURE


  ├── src/
  
  │ └── main.py
  
  ├── data/
  
  │ └── Dataset/

  ├── results/
  
  │ ├── model_comparison.png
  
  │ └── confusion_matrix.png

  ├── README.md

  └── requirements.txt


---

## ⚙️ INSTALLATION

```bash
git clone https://github.com/YOUR_USERNAME/Micro-Expression-Analysis.git
cd Micro-Expression-Analysis
pip install -r requirements.txt
```

---

## ▶️ USAGE
```bash
python src/main.py
```

---

## 📊 RESULTS
- 🔍 Model Comparison
  - Best performance achieved with k-NN using original features

  - Accuracy reached approximately 95.7%

  - Random Forest also showed strong performance (~93%)

-  🧠 Confusion Matrix (Best Model)
     - High accuracy across all classes

    - Very low misclassification rates

    - Strong distinction between emotional categories


---

## 📌 KEY FINDING
  -  Feature reduction (PCA) slightly decreases performance compared to raw features
  - k-NN performed best among all models
  - Micro-expression classification is feasible with classical ML methods

---

## 🔮 FUTURE IMPROVEMENTS
- Deep learning models (CNN, LSTM)
- Real-time webcam analysis
- Larger and more diverse datasets

---

## 📄 LICENSE
MIT License
