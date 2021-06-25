# SpamEmailClassification
Author: Gyeongwon James Kim 
Date: June 18th, 2021 - June 25, 2021 
Interactive Supervised Machine Learning model tool that performs Spam Email Classification using spam email data


SpamClassifier.py 
Using supervised machine learning, train and test a model that predicts whether an email is spam or not spam. Allows the user to select the type of ML model from Support Vector Machine, K-neighbors classifier, and Perceptron. ML model is trained and tested on success ratio and f1-score. User can save ML model in pickle file format to project folder.

EmailConverter.py 
Creates dictionary of the word bank used for the ML model. Converts raw text to an array where each element corresponds to the number of times the word appeared in the email. 

Predict.py 
User-interactive tool that allows user to load a trained ML model then input test email text. After converting email to array using EmailConverter.py, uses the ML model to predict spam or non-spam email.
