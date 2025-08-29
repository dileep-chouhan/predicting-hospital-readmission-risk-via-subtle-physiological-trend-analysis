import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_patients = 500
data = {
    'HeartRate': np.random.randint(60, 100, num_patients),
    'BloodPressure': np.random.randint(90, 140, num_patients),
    'OxygenSaturation': np.random.randint(95, 100, num_patients),
    'Readmitted': np.random.randint(0, 2, num_patients) # 0: No, 1: Yes
}
df = pd.DataFrame(data)
# Introduce some correlation for realism (higher heart rate and lower oxygen saturation increase readmission risk)
df['Readmitted'] = np.where((df['HeartRate'] > 90) | (df['OxygenSaturation'] < 97), 1, df['Readmitted'])
# --- 2. Data Cleaning and Feature Engineering (Minimal in this synthetic example) ---
# In a real-world scenario, this would involve handling missing values, outliers, etc.
# For this example, data is already clean.
# --- 3. Predictive Modeling ---
X = df[['HeartRate', 'BloodPressure', 'OxygenSaturation']]
y = df['Readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Physiological Variables')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
plt.figure(figsize=(8,6))
sns.countplot(x='Readmitted', data=df)
plt.title('Readmission Count')
plt.savefig('readmission_count.png')
print("Plot saved to readmission_count.png")