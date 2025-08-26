import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# ✅ Step 1: Add column names
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

# ✅ Step 2: Load and label the CSV
df = pd.read_csv("processed.cleveland.csv", header=None, names=columns)

# ✅ Step 3: Clean data
df = df.replace("?", pd.NA)
df = df.dropna()
df = df.astype(float)

# ✅ Step 4: Simplify target (1 = Disease, 0 = No Disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# ✅ Step 5: Add Patient IDs
df['PatientID'] = range(1, len(df) + 1)

# ✅ Step 6: Prepare training data
X = df.drop(['target', 'PatientID'], axis=1)
y = df['target']

# ✅ Step 7: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Step 8: Train decision tree
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

# ✅ Step 9: Visualize tree
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=["🟠 LOW RISK", "🔵 AT RISK"],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("PreventiScan_Heart_Tree")
graph.view()

# ✅ Step 10: Predict patient risk
df['PredictedRisk'] = clf.predict(X)
df['PredictedRisk'] = df['PredictedRisk'].apply(lambda x: "🔵 AT RISK" if x == 1 else "🟠 LOW RISK")

# ✅ Step 11: Show risk summary
total = len(df)
at_risk = df[df['PredictedRisk'] == "🔵 AT RISK"]
low_risk = df[df['PredictedRisk'] == "🟠 LOW RISK"]

print("\n" + "="*70)
print("🫀 PREVENTISCAN HEART DISEASE RISK SUMMARY")
print("="*70)
print(f"👥 Total Patients Analyzed: {total}")
print(f"🔵 At Risk Patients       : {len(at_risk)}")
print(f"🟠 Low Risk Patients      : {len(low_risk)}")
print("="*70 + "\n")

# ✅ Step 12: Preview top patients
print("👤 FIRST 10 PATIENT RISK PREDICTIONS:")
print(df[['PatientID', 'age', 'chol', 'trestbps', 'thalach', 'PredictedRisk']].head(50))
print("\n📁 Full patient report saved to 'PreventiScan_RiskReport.csv'")

# ✅ Step 13: Export report
df[['PatientID', 'age', 'sex', 'chol', 'trestbps', 'thalach', 'oldpeak', 'PredictedRisk']].to_csv("PreventiScan_RiskReport.csv", index=False)
