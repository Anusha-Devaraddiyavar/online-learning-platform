import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset from Google Colab path
print("Trying to load dataset...")
try:
    df = pd.read_csv("/content/student_data_200.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    # Adding a realistic dummy dataset just in case the file is missing so the code still runs
    print("CSV not found! Creating 200 rows of dummy data to test the code...")
    np.random.seed(42)
    dummy_data = {
        'Study_Hours': np.random.randint(1, 10, 200),
        'Attendance': np.random.randint(50, 100, 200),
        'Previous_Marks': np.random.randint(40, 100, 200)
    }
    df = pd.DataFrame(dummy_data)
    # Simple logic to determine Pass/Fail for dummy data
    score = (df['Study_Hours'] * 10) + df['Attendance'] + df['Previous_Marks']
    df['Result'] = ['Pass' if s > 175 else 'Fail' for s in score]

# 2. Print first few rows to understand data
print("\nFirst 5 rows of our dataset:")
print(df.head())

# 3. Do basic preprocessing
print("\nCleaning data...")

# fill missing numbers with the average (simple fix)
df = df.fillna(df.mean(numeric_only=True))

# Convert Result column into 1 and 0 because ML models like numbers
# Pass = 1, Fail = 0
if 'Result' in df.columns:
    df['Result'] = df['Result'].map({'Pass': 1, 'Fail': 0})
    # fill any missing text conversions with 0 just to be safe
    df['Result'] = df['Result'].fillna(0)

# 4. Select only important columns
print("Selecting important features...")
X = df[['Study_Hours', 'Attendance', 'Previous_Marks']] # input columns
y = df['Result'] # output column we want to predict

# 5. Split data into training and testing (80/20)
print("Splitting data into 80% training and 20% testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model (Upgraded to Random Forest for more accurate & exact predictions)
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

# 7. Predict test data
print("Making predictions on test data...")
y_pred = model.predict(X_test)
print("Prediction done!")

# 8. Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# 9. Plot at least one simple graph using matplotlib
print("\nPlotting graph...")
plt.figure(figsize=(8, 5))
plt.scatter(df['Study_Hours'], df['Result'], color='green', alpha=0.5)
plt.title('Study Hours vs Result (1=Pass, 0=Fail)')
plt.xlabel('Study Hours')
plt.ylabel('Result (1=Pass, 0=Fail)')
plt.yticks([0, 1]) # only show 0 and 1 on y-axis
plt.grid(True)
plt.show()

# 10. Take user input manually and predict Pass/Fail
print("\n--- Let's predict for a new student ---")
try:
    # getting input from user
    hours = float(input("Enter Study Hours (e.g. 5): "))
    attendance = float(input("Enter Attendance % (e.g. 85): "))
    marks = float(input("Enter Previous Marks (e.g. 75): "))
    
    # create a new dataframe for this student
    new_student = pd.DataFrame({
        'Study_Hours': [hours],
        'Attendance': [attendance],
        'Previous_Marks': [marks]
    })
    
    # predict using our trained model
    prediction = model.predict(new_student)
    
    print("\n--- Final Result ---")
    if prediction[0] == 1:
        print("Prediction: The student is likely to PASS!")
    else:
        print("Prediction: The student is likely to FAIL.")
except ValueError as e:
    print(f"Invalid input! Please enter numbers only next time. (Error details: {e})")
