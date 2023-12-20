# Machine Learning Project #1: Early stage diabetes risk prediction  
This is one of the machine learning projects which I completed as part of [DataTalksClub Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). 

## Problem and data description 
Diabetes is a group of common endocrine diseases characterized by sustained high blood sugar levels. If left untreated, the disease can lead to various health complications, including disorders of the cardiovascular system, eye, kidney, and nerves.[3] Untreated or poorly treated diabetes accounts for approximately 1.5 million deaths every year. Therefore, it is important for individuals to know if they are at high risk of developing diabetes, so they can utilise early intervention strategies and prevent the disease from getting more debilitating.  

In this project, I use a dataset which comprises crucial sign and symptoms of individuals who either exhibit early signs of diabetes or are at risk of developing diabetes. The variables included in the dataset provide valuable insights into potential indicators of diabetes onset. The dataset encompasses diverse information, ranging from demographic details to specific symptoms associated with diabetes. The data set is available at https://www.kaggle.com/datasets/tanshihjen/early-stage-diabetes-risk-prediction as well as inside the data folder in this repository.

The aim of the project is to create an early-stage diabetes warning service for individuals. For this, I will train several machine learning models and then deploy the best-performing model to a web service so that individuals can enter their attributes and in return they receive a notification regarding their early-stage diabetes risk prediction. 

Attributes Description:  

Age (1-20 to 65): Age range of the individuals.  
Sex (1. Male, 2. Female): Gender information.  
Polyuria (1. Yes, 2. No): Presence of excessive urination.  
Polydipsia (1. Yes, 2. No): Excessive thirst.  
Sudden Weight Loss (1. Yes, 2. No): Abrupt weight loss.  
Weakness (1. Yes, 2. No): Generalized weakness.  
Polyphagia (1. Yes, 2. No): Excessive hunger.  
Genital Thrush (1. Yes, 2. No): Presence of genital thrush.  
Visual Blurring (1. Yes, 2. No): Blurring of vision.  
Itching (1. Yes, 2. No): Presence of itching.  
Irritability (1. Yes, 2. No): Display of irritability.  
Delayed Healing (1. Yes, 2. No): Delayed wound healing.  
Partial Paresis (1. Yes, 2. No): Partial loss of voluntary movement.  
Muscle Stiffness (1. Yes, 2. No): Presence of muscle stiffness.  
Alopecia (1. Yes, 2. No): Hair loss.  
Obesity (1. Yes, 2. No): Presence of obesity.  
Class (1. Positive, 2. Negative): Diabetes classification.  

Exploratory data analysis (EDA) and model training are included in notebook.ipynb. The data were divided into three sets: training, validation, and test. Four different models were trained and their AUC scores are given in the following table.  

| Model | AUC in training set | AUC in validation set | AUC in test set | Final model |
| ----- | --------------------| --------------------- | --------------- | ----------- |
| Logistic regression | 0.941 | 0.936                 |                 |             |
| Decision tree       | 0.983 | 0.969                 |                 |             |
| Random forest       | 1     | 0.995                 | 0.999           |  *          |
| XBGoost             | 1     | 0.994                 |                 |             |

Random forest was selected as our final model as it achieved the highest AUC scores. A separate python file named ``train.py`` was created for final training of the full training data (training + validation) with the hyperparameter settings which were determined in the previous step (i.e. using only the training set). Additionally, a 5-fold cross validation is performed in ``train.py``. Finally, AUC score on the test set is computed and the model is saved to 'rf_model_diabetes.bin' . To run ``train.py``,

1) Clone this repository on your computer: ``git clone https://github.com/topahande/machine-learning-project-1.git``
2) Go the the directory machine-learning-project-1: ``cd machine-learning-project-1``
3) Run the command: ``python train.py``  
   The output should look like this:


### Model deployment

The final model was deployed with Flask (see ``predict.py`` and ``predict-test.py``). ``predict-test.py`` contains information of two individuals taken from the test data. The following code should return the decision for these two individuals.

In a terminal, run following commands:

``pip install gunicorn``

``gunicorn --bind 0.0.0.0:9696 predict:app``

In another  terminal, run the following command:  

``python predict-test.py``  

The output should look like this:


### Dependency and environment management. 

``pip install pipenv``

``pipenv install numpy scikit-learn==1.3.0 flask gunicorn``

``pipenv run gunicorn --bind 0.0.0.0:9696 predict:app``

In another  terminal, run the following command:  

``python predict-test.py``  


### Containerization  

``docker run -it --rm --entrypoint=bash python:3.11.7-slim``

``docker build -t diabetes-project .``

``docker run -it --rm -p 9696:9696 diabetes-project``

In another  terminal, run the following command:  

``python predict-test.py`` 







