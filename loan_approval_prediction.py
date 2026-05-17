import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

# 1. Data Collection (Generating Dummy Dataset)
def generate_dummy_data(filename="loan_data.csv"):
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'ApplicantIncome': np.random.normal(50000, 20000, n_samples),
        'CoapplicantIncome': np.random.normal(10000, 5000, n_samples),
        'LoanAmount': np.random.normal(200000, 50000, n_samples),
        'Loan_Amount_Term': np.random.choice([120, 240, 360, 480], n_samples),
        'Credit_History': np.random.choice([0.0, 1.0], n_samples, p=[0.2, 0.8]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values to demonstrate preprocessing
    df.loc[np.random.choice(df.index, 50), 'LoanAmount'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'Credit_History'] = np.nan
    
    # Determine Loan_Status (Target Variable) based on some logic
    # Higher chance of approval if Credit_History is 1 and Income is high
    approval_prob = (df['Credit_History'] == 1.0) * 0.6 + (df['ApplicantIncome'] > 40000) * 0.3 + 0.05
    # Cap probability between 0 and 1
    approval_prob = np.clip(approval_prob, 0, 1)
    df['Loan_Status'] = np.where(np.random.rand(n_samples) < approval_prob, 'Y', 'N')
    
    df.to_csv(filename, index=False)
    print(f"Dummy dataset created: {filename}")
    return filename

# Main pipeline
def main():
    print("--- Loan Approval Prediction System ---\n")
    
    # 1. Data Collection
    csv_file = "loan_data.csv"
    if not os.path.exists(csv_file):
        generate_dummy_data(csv_file)
    
    df = pd.read_csv(csv_file)
    print("Dataset Loaded. Shape:", df.shape)
    
    # 2. Data Preprocessing
    print("\n--- Data Preprocessing ---")
    # Handling missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    print("Missing values handled.")
    
    # Encoding Categorical Variables
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    print("Categorical variables encoded.")
    
    # 3. Exploratory Data Analysis (EDA)
    print("\n--- Exploratory Data Analysis ---")
    # Generating a correlation matrix plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("Correlation matrix saved as 'correlation_matrix.png'.")
    plt.close()

    # 4. Feature Selection
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # 5. Model Training
    print("\n--- Model Training ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train) 
    dt_pred = dt_model.predict(X_test)
    
    # 6. Model Evaluation
    print("\n--- Model Evaluation ---")
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
    print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
    print("Classification Report (Logistic Regression):\n", classification_report(y_test, lr_pred))
    
    print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_pred))
    print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
    
    # 7. Prediction Output
    print("\n--- Sample Prediction ---")
    sample_idx = 0
    sample_applicant = X_test.iloc[sample_idx:sample_idx+1] 
    actual_status = y_test.iloc[sample_idx]
    
    sample_scaled = scaler.transform(sample_applicant)
    prediction = lr_model.predict(sample_scaled)[0]
    
    print("Applicant Details:")
    for col in sample_applicant.columns:
        print(f"  {col}: {sample_applicant[col].values[0]}")
        
    status_decoder = le_dict['Loan_Status'].inverse_transform([prediction, actual_status])
    print(f"\nPredicted Loan Status: {'Approved' if status_decoder[0] == 'Y' else 'Rejected'}")
    print(f"Actual Loan Status: {'Approved' if status_decoder[1] == 'Y' else 'Rejected'}")

if __name__ == "__main__":
    main()
