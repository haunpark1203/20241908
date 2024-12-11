import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Load and prepare the dataset
iris = pd.read_csv("iris.csv")
X_iris = iris.drop("variety", axis=1)
X_iris = X_iris.drop("sepal.width", axis=1)
X_iris = X_iris.drop("petal.width", axis=1)
y_iris = iris['variety']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

# Hyperparameter tuning
n_neighbors = int(input("Enter the number of neighbors (k) for KNN: "))

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate performance
accuracy = round(accuracy_score(y_test, y_pred), 2)
balanced_accuracy = round(balanced_accuracy_score(y_test, y_pred), 2)

# Display model performance
print(f"Model Accuracy: {accuracy}")
print(f"Balanced Accuracy: {balanced_accuracy}")

# User input for prediction
user_input = input("Enter the sepal.length and petal.length for the iris in the format 'sepal_length,petal_length': ")

try:
    sl, pl = user_input.split(',')
    sl = float(sl)
    pl = float(pl)

    new_test_data = pd.DataFrame({'sepal.length': [sl], 'petal.length': [pl]})

    prediction = knn.predict(new_test_data)
    print("Predicted Label: ", prediction[0])
except Exception as e:
    print("Invalid input. Please enter the values in the correct format. Error:", e)