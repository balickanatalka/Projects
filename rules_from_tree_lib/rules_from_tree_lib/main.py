import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from id3_c45_classifier import ID3C45Classifier

# wczytanie danych z folderu nadrzędnego
df = pd.read_csv("../../data_discretized.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = ID3C45Classifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("Średnia accuracy:", sum(accuracies) / len(accuracies))