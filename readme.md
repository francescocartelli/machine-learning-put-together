# Machine Learning Put Together
_Python implementations of few standard ML algorithms for classification, models evaluation and preprocessing._ 

## Folder Structure
```
src
.
│
├── classifiers
|   ├── __init__.py
│   ├── gaussian_c.py
│   ├── gmm.py
│   ├── logistic_reg.py
│   └── svm.py
│
├── examples
|   ├── iris
|   |   ├── data_util.py
|   |   ├── gaussian_c_example.py
|   |   ├── gmm_example.py
|   |   ├── logistic_reg_example.py
|   |   ├── preprocessing_example.py
|   |   └── svm_example.py
|   |
|   └── pulsar
|        ├── data
|        |   ├── test.txt
|        |   └── train.txt
|        |
|        └── measuring_example.py
│
├── measuring_predictions
|   ├── __init__.py
│   └── measuring_predictions.py
│
├── plotting
|   ├── __init__.py
│   └── plotting.py
│
├── preprocessing
|   ├── __init__.py
│   ├── gaussianization.py
│   ├── lda.py
│   └── pca.py
│
└── utils.py
```

## Prerequirements
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)<br><br>
The following packages are used in this project:
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

## Measuring Predictions
Measuring prediction of models and evaluating scores.

## Plotting
Useful tools for plotting results and comparing models by using measuring_prediction module.

## Examples
### Iris
All following examples are executed on iris dataset.
- **Gaussian classifiers example**: Straightforward application of Gaussian classifiers. 
- **Gaussian Mixture Model example**: Straightforward application of GMM classifier. 
- **Logistic Regression example**: Straightforward application of logistic regression classifier.
- **Preprocessing**: Plotting of iris dataset after application of LDA and PCA. 
- **Support Vector Machine example**: Straightforward application of SVM classifier. 
### Pulsar
All following examples are executed on pulsar dataset.
