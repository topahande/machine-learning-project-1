#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

person_id_1 = 'id1'
person_1 = {
    "gender": "Female",
    "polyuria": "Yes",
    "polydipsia": "Yes",
    "sudden_weight_loss": "No",
    "weakness": "Yes",
    "polyphagia": "Yes",
    "genital_thrush": "No",
    "visual_blurring": "Yes",
    "itching": "Yes",
    "irritability": "No",
    "delayed_healing": "Yes",
    "partial_paresis": "Yes",
    "muscle_stiffness": "Yes",
    "alopecia": "No",
    "obesity": "No",
    "age": 55
}
person_id_2 = 'id2'
person_2 = {
    "gender": "Male",
    "polyuria": "No",
    "polydipsia": "No",
    "sudden_weight_loss": "No",
    "weakness": "No",
    "polyphagia": "Yes",
    "genital_thrush": "Yes",
    "visual_blurring": "No",
    "itching": "No",
    "irritability": "No",
    "delayed_healing": "No",
    "partial_paresis": "No",
    "muscle_stiffness": "No",
    "alopecia": "No",
    "obesity": "No",
    "age": 45
}

response_1 = requests.post(url, json=person_1).json()
response_2 = requests.post(url, json=person_2).json()

print(response_1)

if response_1['diabetes'] == True:
    print('Sending positive result to %s' % person_id_1)
else:
    print('Sending negative result to %s' % person_id_1)

print(response_2)

if response_2['diabetes'] == True:
    print('Sending positive result to %s' % person_id_2)
else:
    print('Sending negative result to %s' % person_id_2)
