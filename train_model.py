import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. Load and Clean Data
df = pd.read_csv('dataset/dataset.csv')
symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]

def clean(s):
    return str(s).strip().replace(" ", "_") if pd.notna(s) else s

for col in symptom_cols:
    df[col] = df[col].apply(clean)

# 2. Advanced Feature Engineering: One-Hot Encoding
all_symptoms = sorted(list(set(df[symptom_cols].values.flatten()) - {np.nan, 'nan', None}))
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for i, row in df.iterrows():
    for col in symptom_cols:
        val = row[col]
        if val in all_symptoms:
            X.loc[i, val] = 1

y = df['Disease'].str.strip()

# 3. Demonstrating ML Skill: Cross-Validation
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.2%}")

# 4. Final Fit and Save
model.fit(X, y)
with open('health_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'symptoms': all_symptoms}, f)