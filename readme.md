
<div align="center">
  <h1>Machine Learning Put Together</h1>
  <div align="center"><p><i>Python implementations of few standard ML algorithms for classification, models evaluation and preprocessing.</i></p></div>
  <div align="center">
    <a href="https://www.python.org/"><img src="http://ForTheBadge.com/images/badges/made-with-python.svg"></a>
  </div>
</div>

## Prerequirements   
The following packages are used in this project:  
- **Matplotlib**  
- **Numpy**  
- **Scipy**  
- **Sklearn**  
  
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
|        |   └── HTRU_2.csv  
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
Useful tools for plotting results and comparing models by using measuring_predictions module.  
  
## Examples  
Based on dataset.  
### Iris  
Dataset at https://archive.ics.uci.edu/ml/datasets/iris.  
- **Gaussian classifiers example**: Straightforward application of Gaussian classifiers and leave-one-out version applied to the classifiers.   
- **Gaussian Mixture Model example**: Straightforward application of GMM classifier.   
- **Logistic Regression example**: Straightforward application of logistic regression classifier.  
- **Preprocessing**: Plotting of iris dataset after application of LDA and PCA.   
- **Support Vector Machine example**: Straightforward application of SVM classifier.   
### HTRU2  
Dataset at https://archive.ics.uci.edu/ml/datasets/HTRU2.  
- **Measuring**: Evalutating predictions score between different classifiers.  
  
  
## Author  
- *Francesco Cartelli* (https://github.com/francescocartelli)
