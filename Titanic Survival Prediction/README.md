# Titanic Survival Prediction

## Project Overview
This project uses Machine Learning to predict whether a passenger survived the Titanic disaster. It uses passenger-related information to train a classification model and make survival predictions.

This is a beginner-friendly AI/ML project developed using Python.

## Objectives
- Understand and explore passenger data.
- Preprocess data for Machine Learning.
- Train a classification model.
- Predict passenger survival.
- Evaluate model performance using a confusion matrix and classification report.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Algorithm
**Random Forest Classifier**

Random Forest is a supervised Machine Learning algorithm that combines multiple decision trees to make predictions for classification tasks.

## Project Features
- Data loading and preprocessing
- Passenger survival prediction
- Random Forest model training
- Confusion matrix visualization
- Classification report with precision, recall, and F1-score

## Project Structure
```text
Titanic-Survival-Prediction/
├── titanic_survival_prediction.py
├── titanic.csv
├── confusion_matrix.png
└── README.md
```

*Note: The dataset and confusion matrix image are optional. Include them only if they are part of your project.*

## Installation

1. Install Python on your computer.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Download or clone this repository.
4. Open the project folder in VS Code.
5. Run the Python file:

```bash
python titanic_survival_prediction.py
```

Make sure the dataset path in the code matches the location of your CSV file.

## Model Evaluation
The project evaluates the model using a confusion matrix and a classification report.

The confusion matrix compares actual passenger outcomes with predicted outcomes. The classification report provides precision, recall, F1-score, and support for each class.

The Random Forest confusion matrix in my current test output shows 3 correct predictions out of 4 test samples, equivalent to 75% accuracy on that small test set. This result is only an example of the current test output and does not represent guaranteed performance on other data.

## Learning Outcomes
- Basic understanding of Machine Learning
- Data preprocessing with Pandas
- Training a classification model
- Evaluating model predictions
- Visualizing results with Matplotlib and Seaborn

## Future Improvements
- Evaluate the model on a larger test set.
- Improve data preprocessing and feature selection.
- Compare Random Forest with other classification algorithms.
- Tune model parameters to improve performance.

## Author
Created as a beginner AI/ML portfolio project using Python.

## License
This project is available for educational and learning purposes.
