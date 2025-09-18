# Task 03 - Decision Tree Classifier (Bank Marketing Dataset)
# Author: Deepika

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import graphviz

# 1. Load dataset
df = pd.read_csv("bank.csv", sep=";")   # Or bank-additional-full.csv
print("✅ Dataset Loaded")
print("First 5 rows:\n", df.head())

# 2. Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# 3. Split Features & Target
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Build & Train Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)
print("✅ Decision Tree trained")

# 5. Predictions & Evaluation
y_pred = clf.predict(X_test)
print("\n📊 Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Feature Importance Bar Chart (POP + SAVE)
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title("Feature Importances")
plt.savefig("feature_importance.png")
plt.show()   # 👈 Pops up bar chart
print("✅ Feature importance chart saved as feature_importance.png")

# 7. Graphviz Readable Tree (POP + SAVE)
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=3   # 👈 limit depth for better readability
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_readable", format="png")
graph.view()   # 👈 Pops up tree
print("✅ Readable Decision Tree saved as decision_tree_readable.png")

# 8. Human-readable Text Rules
tree_rules = export_text(clf, feature_names=list(X.columns), max_depth=3)
with open("decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)
print("✅ Tree rules saved as decision_tree_rules.txt")

# 9. Save Model
joblib.dump(clf, "bank_dt_model.joblib")
print("✅ Model saved as bank_dt_model.joblib")
