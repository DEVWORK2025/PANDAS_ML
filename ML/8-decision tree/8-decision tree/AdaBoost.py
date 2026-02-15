# Quick start: AdaBoost classifier with decision stumps
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1) Data
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 2) Base learner (weak learner)
base_stump = DecisionTreeClassifier(max_depth=1, random_state=42)

# 3) AdaBoost
ada = AdaBoostClassifier(
    estimator=base_stump,      # sklearn >=1.2; use 'base_estimator' in older versions
    n_estimators=200,
    learning_rate=0.5,
    algorithm="SAMME.R",       # "SAMME.R" is faster, "SAMME" supports non-probabilistic base learners
    random_state=42
)
ada.fit(X_train, y_train)

# 4) Evaluation
y_pred = ada.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))