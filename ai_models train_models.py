import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Generate synthetic training data
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'publications': np.random.randint(5, 80, n),
    'citations': np.random.randint(10, 500, n),
    'teaching_feedback': np.random.uniform(3.5, 5.0, n),
    'projects': np.random.randint(0, 15, n),
    'admin_roles': np.random.randint(0, 5, n),
    'years_experience': np.random.randint(3, 35, n),
    'phd_completed': np.random.choice([0, 1], n),
    'promoted': np.random.choice([0, 1], n, p=[0.65, 0.35])
})

X = data.drop('promoted', axis=1)
y = data['promoted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=7, random_state=42)
model.fit(X_train, y_train)

print("AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

model.save_model('ai_models/career_model.json')
print("Model trained and saved successfully!")
