# SpamEmailClassification
Author: Gyeongwon James Kim \
Date: June 18th, 2021 - June 25, 2021 

This module is an Interactive Supervised Machine Learning model tool that performs Spam Email Classification using spam email data. \
Data from https://www.kaggle.com/nitishabharathi/email-spam-dataset and https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv were used to generate training and test samples. The input is an integer array in which the elements correspond to the number of times a specific word was used in the contents of the email, given a dictionary with common words. The same dictionary was used for both datasets in order to train the model on datasets of various sources. \
\
The user has the flexibility to choose the type of model used for training: SVM, K-neighbor-classifier, and Perceptron models, which were all imported from the sklearn library. After training the model, the test set was used to evaluate the accuracy and f1-score of the model's binary classification, which varied depending on the model used. The perceptron model performed the best with over 90% binary accuracy as well as around .80 f1-score. Following, the k-neighbors-classifier performed well with approximately 88% accuracy and .78 f1-score (1 and 5 neighbors were used, generating similar results). The Support Vector Machine performed the poorest with 85% accuracy and 0.57 f1-score. Finally, the user is able tosave models to pickle files, which can then be imported in Predict.py and used with user inputed email contents. \
\
One limiation of this module isthat it is restricted to only 3 types of models. However, this can be remedied by adding more models in the future. In addition, this spam email classifier takes into account just the contents of the email, without putting importance in the subject or the sender, which can possibly generate better models with higher acccuracy and f1-score. 


\
**Features** \
SpamClassifier.py \
Using supervised machine learning, train and test a model that predicts whether an email is spam or not spam. Allows the user to select the type of ML model from Support Vector Machine, K-neighbors classifier, and Perceptron. ML model is trained and tested on success ratio and f1-score. User can save ML model in pickle file format to project folder. This is where the main function is located, which loads the dataset, creates a selected model, then trains and evaluates the model. \
EmailConverter.py \
Creates dictionary of the word bank used for the ML model. Converts raw text to an array where each element corresponds to the number of times the word appeared in the email. These collection of methods act as helper tools in which data can be converted to fit the input shape of the ML models. \
Predict.py \
User-interactive tool that allows user to load a trained ML model then input test email text. After converting email to array using EmailConverter.py, uses the ML model to predict spam or non-spam email. This file when run allows the user to import already trained models, which can be used to classify an email as spam or non spam. The user can interact with this module by inputting the email contents, then receiving the results from the imported model. 
