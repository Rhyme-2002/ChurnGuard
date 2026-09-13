# 🛡️ ChurnGuard

### Customer Churn Prediction & Supervised ML Benchmarking System

> A comprehensive machine learning project for predicting customer churn and systematically benchmarking multiple supervised classification algorithms using **5-Fold GridSearchCV**.

---

## 📌 Project Overview

**ChurnGuard** is an end-to-end customer churn prediction and supervised machine learning benchmarking project.

The main objective is to identify customers who are likely to churn and compare a wide range of supervised machine learning algorithms under a consistent evaluation framework.

Instead of developing a single predictive model, ChurnGuard evaluates multiple algorithms from different machine learning families using **hyperparameter tuning and 5-fold cross-validation**.

The project covers:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Categorical encoding
* Feature scaling where required
* Multiple supervised ML algorithms
* Hyperparameter optimization using `GridSearchCV`
* 5-fold cross-validation
* Model evaluation
* Model comparison
* Best-model selection
* Customer churn prediction

---

## 🎯 Objectives

The project aims to:

1. Analyze customer characteristics associated with churn.
2. Prepare the dataset for machine learning.
3. Apply different supervised classification algorithms.
4. Optimize model hyperparameters using GridSearchCV.
5. Compare models using common evaluation metrics.
6. Identify the best-performing classification model.
7. Build a reusable benchmarking workflow for customer churn prediction.

---

## 📊 Dataset

The current dataset contains **9,259 customer observations and 16 variables**.

### Target Variable

| Variable | Description                                    |
| -------- | ---------------------------------------------- |
| `churn`  | Customer churn indicator (`0 = No`, `1 = Yes`) |

### Features

| Variable            | Type        | Description                    |
| ------------------- | ----------- | ------------------------------ |
| `age`               | Numerical   | Customer age                   |
| `gender`            | Categorical | Customer gender                |
| `tenure_months`     | Numerical   | Customer tenure                |
| `contract_type`     | Categorical | Contract type                  |
| `monthly_charges`   | Numerical   | Monthly customer charges       |
| `total_charges`     | Numerical   | Total customer charges         |
| `internet_service`  | Categorical | Internet service type          |
| `payment_method`    | Categorical | Payment method                 |
| `support_calls`     | Numerical   | Number of support calls        |
| `late_payments`     | Numerical   | Number of late payments        |
| `online_security`   | Categorical | Online security subscription   |
| `tech_support`      | Categorical | Technical support subscription |
| `streaming_service` | Categorical | Streaming service              |
| `senior_citizen`    | Numerical   | Senior citizen indicator       |
| `family_members`    | Numerical   | Number of family members       |

---

# 🔬 Machine Learning Benchmark

ChurnGuard evaluates models from multiple supervised learning families.

### Linear

* Logistic Regression

### Distance-Based

* K-Nearest Neighbors (KNN)

### Tree-Based

* Decision Tree

### Bagging

* Random Forest
* Extra Trees

### Boosting

* AdaBoost
* Gradient Boosting
* XGBoost
* HistGradientBoosting

### Kernel-Based

* Support Vector Machine (SVM)

### Probabilistic

* Gaussian Naive Bayes

### Discriminant Analysis

* Linear Discriminant Analysis (LDA)
* Quadratic Discriminant Analysis (QDA)

### Neural Network

* Multi-Layer Perceptron (MLP)

---

## 🧠 Model Architecture

```text
                    ChurnGuard
                        │
                        ▼
                 Customer Dataset
                        │
                        ▼
              Data Processing & EDA
                        │
                        ▼
                 Train/Test Split
                        │
                        ▼
              Feature Preprocessing
                        │
              ┌─────────┴─────────┐
              │                   │
       Numerical Features   Categorical Features
              │                   │
        Scaling if needed     One-Hot Encoding
              │                   │
              └─────────┬─────────┘
                        ▼
              Multiple ML Algorithms
                        │
                        ▼
                  GridSearchCV
                        │
                  5-Fold CV
                        │
                        ▼
              Best Hyperparameters
                        │
                        ▼
                 Test Evaluation
                        │
                        ▼
                 Model Comparison
                        │
                        ▼
                  Best Model
                        │
                        ▼
              Customer Churn Prediction
```

---

# ⚙️ Data Processing

The preprocessing strategy depends on the machine learning algorithm.

### Numerical Features

For models that require scaling, numerical variables are standardized using:

```python
StandardScaler()
```

### Categorical Features

Categorical variables are converted into numerical representations using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Tree-Based Models

Tree-based algorithms such as:

* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting

do not require standard feature scaling.

---

# 🔍 Exploratory Data Analysis

The EDA stage investigates:

* Dataset structure
* Missing values
* Duplicate observations
* Numerical variable distributions
* Categorical variable distributions
* Churn distribution
* Churn rate
* Numerical feature relationships
* Categorical feature relationships
* Correlation patterns
* Potential outliers

Example analyses include:

```text
Customer Churn Distribution
        ↓
Numerical Feature Analysis
        ↓
Categorical Feature Analysis
        ↓
Churn vs Customer Characteristics
        ↓
Correlation Analysis
        ↓
Feature Insights
```

---

# 🎛️ Hyperparameter Optimization

Each model is tuned using **GridSearchCV with 5-fold cross-validation**.

General structure:

```python
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
```

The best model is selected based on cross-validation performance.

---

# 📏 Model Evaluation

Models are evaluated using multiple classification metrics.

### Accuracy

Measures the proportion of correctly classified customers.

### Precision

Measures how many customers predicted as churners actually churned.

### Recall

Measures how many actual churners were correctly identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers.

### Confusion Matrix

Provides:

```text
                 Predicted
                 0       1
Actual  0       TN      FP
        1       FN      TP
```

---

# 📈 Model Comparison

The final benchmarking stage compares all models using a common evaluation framework.

Example result structure:

| Model                | CV Accuracy | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
| -------------------- | ----------: | ------------: | --------: | -----: | -: | ------: |
| Logistic Regression  |           — |             — |         — |      — |  — |       — |
| KNN                  |           — |             — |         — |      — |  — |       — |
| Decision Tree        |           — |             — |         — |      — |  — |       — |
| Random Forest        |           — |             — |         — |      — |  — |       — |
| Extra Trees          |           — |             — |         — |      — |  — |       — |
| AdaBoost             |           — |             — |         — |      — |  — |       — |
| Gradient Boosting    |           — |             — |         — |      — |  — |       — |
| XGBoost              |        0.78 |           0.78|       0.76|    0.78|0.70 |       0.71 |
| HistGradientBoosting |           — |             — |         — |      — |  — |       — |
| SVM                  |           — |             — |         — |      — |  — |       — |
| Gaussian Naive Bayes |           — |             — |         — |      — |  — |       — |
| LDA                  |           — |             — |         — |      — |  — |       — |
| QDA                  |           — |             — |         — |      — |  — |       — |
| MLP Classifier       |           — |             — |         — |      — |  — |       — |

> Results will be updated after completing the full benchmarking process.

---

# 📁 Project Structure

```text
ChurnGuard/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── churn_data.csv
│
├── notebooks/
│   │
│   ├── 01_Data_Processing.py
│   ├── 02_EDA.py
│   │
│   ├── 03_Linear/
│   │   └── Logistic_Regression.py
│   │
│   ├── 04_Distance/
│   │   └── KNN.py
│   │
│   ├── 05_Tree/
│   │   └── Decision_Tree.py
│   │
│   ├── 06_Bagging/
│   │   ├── Random_Forest.py
│   │   └── Bagging.py
│   │
│   ├── 07_Boosting/
│   │   ├── AdaBoost.py
│   │   ├── Gradient_Boosting.py
│   │   ├── XGBoost.py
│   │   └── HistGradientBoosting.py
│   │
│   ├── 08_Kernel/
│   │   └── SVM.py
│   │
│   ├── 09_Probabilistic/
│   │   └── Gaussian_Naive_Bayes.py
│   │
│   ├── 10_Discriminant/
│   │   ├── LDA.py
│   │   └── QDA.py
│   │
│   └── 11_Neural_Network/
│       └── MLP_Classifier.py
│
├── results/
│   ├── model_comparison.csv
│   ├── best_models.csv
│   └── figures/
│
└── src/
    ├── preprocessing.py
    ├── evaluation.py
    └── model_comparison.py
```

---

# 🛠️ Technologies

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Selection

* GridSearchCV
* Cross-Validation

### Development

* Jupyter Notebook
* GitHub

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Rhyme-2002/ChurnGuard.git
```

Move into the project directory:

```bash
cd ChurnGuard
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start with the data processing notebook:

```text
notebooks/
└── 01_Data_Processing.ipynb
```

Then perform exploratory analysis:

```text
notebooks/
└── 02_EDA.ipynb
```

After preprocessing and EDA, run the model notebooks according to their categories.

---

# 📚 Learning Objectives

This project is also designed as a practical supervised machine learning learning project.

Through ChurnGuard, the following concepts are implemented:

* Classification
* Data preprocessing
* Encoding
* Feature scaling
* Train-test splitting
* Cross-validation
* Hyperparameter tuning
* GridSearchCV
* Ensemble learning
* Bagging
* Boosting
* Kernel methods
* Probabilistic classification
* Discriminant analysis
* Neural networks
* Model evaluation
* Model benchmarking

---

# 🔮 Future Improvements

Planned improvements include:

* Automated model benchmarking
* Interactive model comparison dashboard
* Feature importance analysis
* SHAP-based model explainability
* Probability-based churn risk scoring
* Automated best-model selection
* Streamlit prediction interface
* Model deployment
* API development using FastAPI

---

# 👨‍💻 Author

## Abu Sufiun Rhyme

**Data Analyst | Data Scientist | Applied Statistician**

B.Sc. (Honours) in Applied Statistics

M.Sc. in Applied Statistics and Data Science

University of Dhaka

### Connect with me

<p align="left">
    <a href="https://rhyme-2002.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/%20Portfolio-181717?style=for-the-badge&logo=googlechrome&logoColor=white" />
  </a>
  <a href="https://www.linkedin.com/in/abu-sufiun-rhyme/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
  <a href="mailto:asrhyme@isrt.ac.bd"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" /></a>
</p>

---

## ⭐ Project

If you find this project useful for learning supervised machine learning and model benchmarking, consider giving the repository a ⭐.

**ChurnGuard — Benchmarking Supervised Machine Learning for Customer Churn Prediction.**
