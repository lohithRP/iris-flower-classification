# ==========================================================
# Title: Train and Save Iris Flower Classification Model
# ==========================================================

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. Load the Iris Dataset
# ----------------------------------------------------------
print("Loading Iris dataset...")
iris = load_iris()

# Convert to DataFrame for better readability
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("\nSample Data:")
print(df.head())

# ----------------------------------------------------------
# 2. Split the Dataset
# ----------------------------------------------------------
X = df.iloc[:, :-1]   # features
y = df.iloc[:, -1]    # labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 3. Train the Model
# ----------------------------------------------------------
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 4. Evaluate the Model
# ----------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 5. Save the Model
# ----------------------------------------------------------
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n✅ Model saved successfully as iris_model.pkl")

# ----------------------------------------------------------
# 6. Plot the Confusion Matrix (for visualization)
# ----------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Iris Classification Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
