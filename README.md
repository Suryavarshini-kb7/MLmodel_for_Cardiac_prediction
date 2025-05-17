# ❤️ Heart Disease Prediction Project

## 📖 Overview
This project develops a **machine learning system** to predict **heart disease** based on patient medical data. It uses the `heart.csv` dataset, which contains **13 features** (e.g., age, cholesterol, chest pain type) and a **binary target** variable:
- `0` = No heart disease  
- `1` = Heart disease

Five machine learning models are implemented:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Each model is trained and evaluated to identify the best performer for this binary classification task.

---

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data loading and exploration
- **NumPy**: Numerical operations
- **Scikit-learn**:
  - `train_test_split`: Split data
  - `StandardScaler`: Feature scaling
  - Models: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`
  - Metrics: `accuracy_score`, `confusion_matrix`, `classification_report`
- **Seaborn & Matplotlib**: Confusion matrix visualizations
- **Google Colab**: Cloud environment with Drive integration

---

## 📊 Dataset

- **Source**: `heart.csv`  
- **Features**: 13 (e.g., age, sex, cholesterol, resting BP)
- **Target**: Binary (`0 = no disease`, `1 = disease`)
- **Size**: ~1,025 samples
- **Split**: 80% training (e.g., 820), 20% testing (e.g., 205)

---

## 🔄 Data Preprocessing

- Loading
- Exploration
- Feature & Target Separation:
  - **X** = all columns except target
  - **y** = target column
- Train-Test Split:  
  80/20 using `train_test_split`
- **Scaling**:  
  `StandardScaler` used to normalize features (for Logistic Regression)
---

## 🤖 Models

 1. Logistic Regression
 2. Decision Tree
 3. Random Forest
 4. Support Vector Machine (SVM)
 5. K-Nearest Neighbors (KNN)
---

## 📈 Training and Evaluation

- **Training**: All models trained on `X_train` and `y_train`
- **Evaluation Metrics**:
  - **Accuracy**: For both train/test sets
  - **Confusion Matrix**: TP, TN, FP, FN
  - **Visualization**: Confusion matrix heatmaps with "YES"/"NO" labels
  - **Classification Report**: Precision, recall, F1-score

- **Prediction**:
  - The **Random Forest** model is used for sample patient predictions
