import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

st.title("AID Streamlit ML App Deployment Practice")

# Load and prepare the dataset
iris = pd.read_csv("iris.csv")
X_iris = iris.drop("variety", axis=1)
X_iris = X_iris.drop("sepal.width", axis=1)
X_iris = X_iris.drop("petal.width", axis=1)
y_iris = iris['variety']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

# Hyperparameter tuning in main section
st.header("Hyperparameter Tuning")
n_neighbors = st.slider("Select number of neighbors (k):", 1, 15, 3)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate performance
accuracy = round(accuracy_score(y_test, y_pred), 2)
balanced_accuracy = round(balanced_accuracy_score(y_test, y_pred), 2)

# Display model performance
st.header("Model Performance")
st.write(f"Model Accuracy: {accuracy}")
st.write(f"Balanced Accuracy: {balanced_accuracy}")

# User input for prediction
st.header("Iris Species Classifier")
user_text = st.text_input("Enter the sepal.length and petal.length for the iris. Please enter in the form of a,b.")

try:
    sl, pl = user_text.split(',')
    sl = float(sl)
    pl = float(pl)

    new_test_data = pd.DataFrame({'sepal.length': [sl], 'petal.length': [pl]})

    prediction = knn.predict(new_test_data)
    st.write("Predicted Label: ", prediction[0])
except:
    st.write("You entered invalid input. Please enter the correct value.")
