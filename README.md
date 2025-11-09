# Heart-Disease-Prediction

# 💓 Heart Disease Prediction Using Machine Learning

This project aims to predict the likelihood of heart disease in a patient using machine learning techniques. By analyzing key medical attributes, this model assists in early diagnosis and prevention.

---

## 📌 Project Overview

- This is a **binary classification** problem where the output is:  
  - **1** = Presence of heart disease  
  - **0** = No heart disease  

- The dataset consists of medical features like **age, sex, chest pain type, resting blood pressure, cholesterol levels**, etc.

- The model is built and executed in a **Google Colab Notebook** using Python-based machine learning libraries.

---

## 🧠 What I Did

### 🔍 1. Exploratory Data Analysis (EDA)

- Analyzed feature distribution and relationships.
- Visualized data using **histograms, count plots, heatmaps**, etc.
- Checked correlations between the input features and the target column.

### 🧹 2. Data Preprocessing

- Handled missing or inconsistent data.
- Converted categorical data into numerical values using **Label Encoding**.
- Standardized features using **StandardScaler** for optimization.

### 🤖 3. Model Building

Trained multiple machine learning models for prediction:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Naive Bayes  

### 📊 4. Model Evaluation

- Evaluated using metrics such as:  
  - **Accuracy**  
  - **Confusion Matrix**  
  - **Precision**  
  - **Recall**  
  - **F1-Score**

- Compared model performances and identified the best model.

### 🏆 5. Best Model Selection

- The **Random Forest Classifier** achieved the highest accuracy and was chosen as the final model for prediction.

---

## 🛠️ Technologies Used

- **Python**
- **Google Colab**
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 🚀 How to Run

1. Clone this repository or download the notebook.
2. Open the `.ipynb` file in **Google Colab** or **Jupyter Notebook**.
3. Run all cells to explore, train, and test the model.
4. Modify the input section to predict for new patients.

---

## 📈 Future Enhancements

- Deploy the model with **Streamlit** or **Flask**.
- Apply **Hyperparameter Tuning** for optimization.
- Add **real-time inputs** via UI for public use.
- Try deep learning approaches (ANNs) and compare.

---

## 📊 Dataset

This project uses a heart disease dataset with patient attributes like age, cholesterol level, blood pressure, etc.  


---

## 🙌 Acknowledgements

- Dataset reference: UCI Machine Learning Repository (or other source).
- Libraries used: thanks to open-source community.

---

