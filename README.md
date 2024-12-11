
# KNN Iris Classifier

This project demonstrates a K-Nearest Neighbors (KNN) classifier for the Iris dataset. The application allows users to:
1. Tune the hyperparameter `k` (number of neighbors).
2. Evaluate the model's performance (accuracy and balanced accuracy).
3. Predict the iris species based on user-provided input.

## Features
- Hyperparameter tuning for KNN (`k` selection).
- Model training and evaluation.
- Console-based user input for custom predictions.
- Performance metrics: Accuracy and Balanced Accuracy.

## Requirements
Ensure you have the following installed:
- Python 3.x
- pandas
- scikit-learn

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CAU-AID-Class.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CAU-AID-Class
   ```
3. Install required libraries:
   ```bash
   pip install pandas scikit-learn
   ```
4. Ensure the `iris.csv` dataset is available in the project directory.

## Usage 
1. Run the Script.
   ```bash
   python knn_iris_classifier.py
   ```
3. Enter the number of neighbors (k) when prompted
4. View the model's accuracy and balanced accuracy in the console

## Example
__Console Interaction:__
   ```bash
   Enter the number of neighbors (k) for KNN: 3
   Model Accuracy:0.95
   Balanced Accuracy: 0.96
   Enter the sepal.length and petal.length for the iris in the format 'sepal_length,petal_legth': 5.1,1.8
   Predicted label: Virginica
   ```
## Notes
- Ensure that the nput for predictions is numeric and follow the correct format (>value1,value2>)
- Modify the dataset loading path in the script if necessary

## License
This project is licensed under the MIT LICENSE

## Author
Developed by >Haun Park, haunpark1203@gmail.com


