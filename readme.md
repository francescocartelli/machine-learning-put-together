# Machine Learning Put Together
_Python implementations of few standard ML algorithms for classification, models evaluation and preprocessing._ 

## Folder Structure
```
src
.
│
├── classifiers
│   ├── gaussian_c.py
│   ├── gmm.py
│   ├── logistic_reg.py
|   ├── svm.py
│   └── utils.py 
│
├── examples
|   ├── datasets.py
│   ├── gaussian_c_example.py
│   ├── logistic_reg_example.py
│   └── svm_example.py
│
├── measuring_predictions.py
│
└── preprocessing.py
```

## Prerequirements
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)<br><br>
The following package are used in this project:
- **Numpy**
- **Scipy**

## Classifiers
- **Gaussian classifiers**:
	- **Multivariate Gaussian classifier**
	- **Naive Bayes Gaussian classifier**
	- **Tied Covariance Gaussian classifier**
- **Logistic Regression**
- **Support Vector Machine**
- **Gaussian Mixture Model**

### How to use them?
Each classifier is represented by a class.<br>
The initialization of each classifier requires at least training data and labels.
Training data is in the form of a numpy matrix NxM (each row is an attribute, each column is a training sample).
Labels are in the form of a numpy vector.<br>
Invoke the classifier *"training_something..."* method for training the model and setting the learning parameters.<br>
Invoke the classifier *"evaluate_something..."* method with the evaluation data set as parameter to compute the evaluation score 
(in the the form of log-likelihood, log-likelihood-ratio or others) for the models.<br>
Use the returned evaluation score to compute the prediction.

## Models Evaluation
Measuring prediction of models and evaluating scores.

## Plotting
Useful tools for plotting results and comparing models.

## Examples

- **Gaussian classifiers example**: Straightforward application of Gaussian classifiers on dataset 1. 
- **Logistic Regression example**: Straightforward application of logistic regression on dataset 1. 
- **Support Vector Machine example**: Straightforward application of SVM on dataset 1. 
- **Gaussian Mixture Model example**: Straightforward application of GMM on dataset 1. 