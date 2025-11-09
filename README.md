# Heart-Disease-Prediction

# 💓 Heart Disease Prediction Using Machine Learning and Django Web App

This project aims to predict the likelihood of heart disease in a patient using machine learning techniques. By analyzing key medical attributes, the system assists in early diagnosis and prevention. The model is trained in a notebook environment and deployed using a Django web application on PythonAnywhere for public use.

---

## 📌 Project Overview

- This is a **binary classification** problem where the output is:  
  - **1** = Presence of heart disease  
  - **0** = No heart disease  

- The dataset consists of medical features like **age, sex, chest pain type, blood pressure, cholesterol levels**, etc.

- The model was built and trained in a **Google Colab Notebook** using Python-based machine learning libraries and then deployed through Django.

---

## 🧠 Machine Learning Model Development

### 🔍 1. Exploratory Data Analysis (EDA)

- Analyzed feature distributions and relationships.
- Visualized data using **histograms, count plots, and heatmaps**.
- Checked correlations between the input features and the target column.

### 🧹 2. Data Preprocessing

- Handled missing or inconsistent data.
- Encoded categorical data using **Label Encoding**.
- Scaled features using **StandardScaler** for better model performance.

### 🤖 3. Model Building

Trained multiple machine learning models for prediction:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Naive Bayes  

### 📊 4. Model Evaluation

- Evaluated models using:  
  - **Accuracy**  
  - **Confusion Matrix**  
  - **Precision**  
  - **Recall**  
  - **F1-Score**

- Compared performance and selected the best model.

### 🏆 5. Best Model Selection

- The **Random Forest Classifier** outperformed other models and was saved as the final model for deployment.

---

## 🖥️ Web App Deployment (Django + PythonAnywhere)

To make the prediction model accessible via a user-friendly interface, a **Django web application** was developed.

### ✅ Key Features

- Simple form to collect patient details.
- Real-time prediction using the trained model (`.joblib` file).
- Clear result display with "Heart Disease Present" or "Not Present".

### ⚙️ Django Project Structure

- `views.py`: Includes logic to preprocess input and generate prediction.
- `urls.py`: Manages routing for web pages.
- `templates/`: Contains the HTML UI for form input.
- Model integration via **Joblib** for loading and prediction.

### 🌐 Live Application

Try the live demo here:  
🔗 **https://ml27priyam.pythonanywhere.com/**

---

## 🛠️ Technologies Used

### Machine Learning:
- **Python**
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`
- Environment: Google Colab

### Web Deployment:
- **Django** (Python Web Framework)  
- **HTML**, **CSS**
- **Joblib** (Model Serialization)  
- **PythonAnywhere** (Hosting)

---


