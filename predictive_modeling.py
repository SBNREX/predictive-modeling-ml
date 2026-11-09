
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# 1. Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("\n===== DATASET INFORMATION =====")
print("Dataset shape:", X.shape)
print("Target distribution:")
print(pd.Series(y).value_counts())
print("Missing values:", X.isnull().sum().sum())

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Initialize and train models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = []
predictions = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = pred
    probabilities[name] = prob

    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC Score": auc
    })

    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("AUC Score:", round(auc, 4))
    print(classification_report(
        y_test,
        pred,
        target_names=data.target_names
    ))

# 4. Model comparison
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# 5. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=data.target_names,
        yticklabels=data.target_names,
        ax=ax
    )

    ax.set_title(name + " Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()

# 6. ROC curves
plt.figure(figsize=(8, 6))

for name, prob in probabilities.items():
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.savefig("roc_curves.png", dpi=300)
plt.show()

# 7. Accuracy comparison
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()

print("\nProject completed successfully!")
print("Visualizations saved in your project folder.")

