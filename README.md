# Machine Learning Project #1: Early-stage diabetes risk prediction  
This project is one of the machine learning projects which I completed as part of [DataTalksClub Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). 

## Problem and data description 
Diabetes is a group of common endocrine diseases characterized by sustained high blood sugar levels. If left untreated, the disease can lead to various health complications, including disorders of the cardiovascular system, eye, kidney, and nerves. Untreated or poorly treated diabetes accounts for approximately 1.5 million deaths every year [[Wikipedia](https://en.wikipedia.org/wiki/Diabetes)]. Therefore, it is important for individuals to know if they are at high risk of developing diabetes so that they can utilise early intervention strategies and prevent the disease from getting more debilitating.  

In this project, I used a dataset which comprises crucial signs and symptoms of individuals who either exhibit early signs of diabetes or are at risk of developing diabetes. The variables included in the dataset provide valuable insights into potential indicators of diabetes onset. The dataset encompasses diverse information, ranging from demographic details to specific symptoms associated with diabetes. The data set is available at https://www.kaggle.com/datasets/tanshihjen/early-stage-diabetes-risk-prediction as well as inside the [data](https://github.com/topahande/machine-learning-project-1/tree/main/data) folder in the current repository.

The aim of this project is to create an early-stage diabetes warning service for individuals. For this, I trained several machine learning models and then deployed the best-performing model to a web service so that individuals can enter their attributes and in return they receive a notification regarding their early-stage diabetes risk prediction. 

Attributes Description:  

- Age: Age of the individuals.  
- Sex (1. Male, 2. Female): Gender information.  
- Polyuria (1. Yes, 2. No): Presence of excessive urination.  
- Polydipsia (1. Yes, 2. No): Excessive thirst.  
- Sudden Weight Loss (1. Yes, 2. No): Abrupt weight loss.  
- Weakness (1. Yes, 2. No): Generalized weakness.  
- Polyphagia (1. Yes, 2. No): Excessive hunger.  
- Genital Thrush (1. Yes, 2. No): Presence of genital thrush.  
- Visual Blurring (1. Yes, 2. No): Blurring of vision.  
- Itching (1. Yes, 2. No): Presence of itching.  
- Irritability (1. Yes, 2. No): Display of irritability.  
- Delayed Healing (1. Yes, 2. No): Delayed wound healing.  
- Partial Paresis (1. Yes, 2. No): Partial loss of voluntary movement.  
- Muscle Stiffness (1. Yes, 2. No): Presence of muscle stiffness.  
- Alopecia (1. Yes, 2. No): Hair loss.  
- Obesity (1. Yes, 2. No): Presence of obesity.  
- Class (1. Positive, 2. Negative): Diabetes classification (This is our target variable). 

## Exploratory data analysis (EDA) and model training  

Exploratory data analysis (EDA) and model training are included in [notebook.ipynb](https://github.com/topahande/machine-learning-project-1/blob/main/notebook.ipynb). In the notebook, the data is divided into three sets: training, validation, and test. Four different models were trained using the training set and their AUC scores are given in the following table:  

| Model | AUC in training set | AUC in validation set | AUC in test set | Final model |
| ----- | --------------------| --------------------- | --------------- | ----------- |
| Logistic regression | 0.941 | 0.936                 |                 |             |
| Decision tree       | 0.983 | 0.969                 |                 |             |
| Random forest       | 1     | 0.995                 | 0.999           |  *          |
| XBGoost             | 1     | 0.994                 |                 |             |

## Exporting the training code of the final model to python script

Random forest was selected as the final model as it achieved the highest AUC scores. A separate python file named [train.py](https://github.com/topahande/machine-learning-project-1/blob/main/train.py) was created for final training of the full training data (training + validation) with the hyperparameter settings which were determined in the previous step (i.e. using only the training set). Additionally, a 5-fold cross validation is performed in [train.py](https://github.com/topahande/machine-learning-project-1/blob/main/train.py). Finally, AUC score on the test set is computed and the final model is saved to ``rf_model_diabetes.bin`` using pickle. To run [train.py](https://github.com/topahande/machine-learning-project-1/blob/main/train.py), perform following steps in your terminal:

1) Clone this repository in a folder on your computer: ``git clone https://github.com/topahande/machine-learning-project-1.git``
2) Go to the directory machine-learning-project-1: ``cd machine-learning-project-1``
3) Run the command: ``python train.py``  
   The output should look like this:

![predict-test-output](https://github.com/topahande/machine-learning-project-1/blob/main/train_screenshot.png)

### Model deployment

The final model was deployed using Flask with Gunicorn as WSGI HTTP server (see [predict.py](https://github.com/topahande/machine-learning-project-1/blob/main/predict.py) and [predict-test.py](https://github.com/topahande/machine-learning-project-1/blob/main/predict-test.py)). Note that Gunicorn works only on Linux and Mac OS. If you are on Windows computer, you could try using waitress instead of Gunicorn.   
[predict-test.py](https://github.com/topahande/machine-learning-project-1/blob/main/predict-test.py) contains information of two individuals taken from the test data in json format. The following codes should return the decision for these two individuals (make sure that you are in directory ``machine-learning-project-1``).

In a terminal, run following commands:

``pip install gunicorn``  (If on Windows: ``pip install waitress``)

``gunicorn --bind 0.0.0.0:9696 predict:app`` (If on Windows:``waitress-serve --listen=0.0.0.0:9696 predict:app``)

In another  terminal, run the following command:  

``python predict-test.py``  

The output should look like this:

![predict-test-output](https://github.com/topahande/machine-learning-project-1/blob/main/predict_test_screenshot.png)

### Dependency and environment management  

I used pipenv for creating a virtual environment:  

``pip install pipenv``

``pipenv install numpy scikit-learn==1.3.0 flask gunicorn``  (This creates two files: [Pipfile](https://github.com/topahande/machine-learning-project-1/blob/main/Pipfile) and [Pipfile.lock](https://github.com/topahande/machine-learning-project-1/blob/main/Pipfile.lock))

To run our service in this environment with the dependencies given in [Pipfile](https://github.com/topahande/machine-learning-project-1/blob/main/Pipfile), run the following code:

``pipenv run gunicorn --bind 0.0.0.0:9696 predict:app``

In another  terminal, run the following command:  

``python predict-test.py``  

This should produce the same output:

![predict-test-output](https://github.com/topahande/machine-learning-project-1/blob/main/predict_test_screenshot.png)

### Containerization  

Containerization was done using Docker (see [Dockerfile](https://github.com/topahande/machine-learning-project-1/blob/main/Dockerfile)). 

First, run ``python:3.11.7-slim`` base image with Docker:  

``docker run -it --rm --entrypoint=bash python:3.11.7-slim``

Then, build the docker image and name it ``diabetes-project`` (using the specifications given in [Dockerfile](https://github.com/topahande/machine-learning-project-1/blob/main/Dockerfile)):  

``docker build -t diabetes-project .``  

Now, we can run our docker image:

``docker run -it --rm -p 9696:9696 diabetes-project``

In another  terminal, run the following command:  

``python predict-test.py`` 

Again, this should produce the output:

![predict-test-output](https://github.com/topahande/machine-learning-project-1/blob/main/predict_test_screenshot.png)

## TO DO: Cloud deployment 

Additionally, we can deploy our service to cloud or kubernetes cluster (local or remote). This is on my to-do list.








