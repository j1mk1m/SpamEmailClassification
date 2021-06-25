# Spam Email Classifier
# James Kim
# June 18th, 2021 - current

# Description: using supervised machine learning, train and test a model that predicts whether an email is a spam or non spam email
#               allows user to select type of ML model then trains the model
#               allows user to save model to pickle file

import numpy as np 
import pandas as pd 
#import tensorflow as tf
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import EmailConverter

def main():
    
    #load data from file
    data = load_all_data()

    #split dataset into training and test sets
    [X_train, y_train, X_test, y_test] = split_data(data)

    #define model from user input
    model = define_model()

    # train model
    fit_model(model, X_train, y_train)

    # predict model and analyze results
    evaluate_model(model, X_test, y_test)

    #save paramters of the model
    user_save_model(model)
    

### METHODS ###

# Input: None | Output: dataset in numpy array format
# Using user input, select file to import dataset from, then import data in numpy array format
def load_all_data():
    data = load_data()
    more = input("Load data from another file? (y/n): ")
    while more == 'y':
        data_temp = load_data()
        data = np.concatenate((data, data_temp), axis=0)
        more = input("Load data from another file? (y/n): ")
    return data

def load_data():
    filename = input("Enter file name: ") or "emails.csv"
    try:
        data = pd.read_csv(os.path.join("Datasets", filename), header=0)
        data = data.to_numpy()
        print("data succesfully loaded")
        data = check_data(data)
        return data
    except:
        print("File not found or data could not be loaded")
        print("Using default data")
        return get_default_data()

def check_data(data):
    try:
        if isinstance(data[0, 1], float) or isinstance(data[0, 1], int):
            return data 
    except:
        print("Could not check data type")
        return data
    try:
        print("Converting data...")
        new_data = [[i] for i in range(len(data))]
        length = len(data[0])
        dictionary = EmailConverter.get_dictionary()
        for i in range(len(data)):
            word_array = EmailConverter.convert_email(data[i, 1].split(), dictionary)
            for freq in word_array:
                new_data[i].append(freq)
            new_data[i].append(data[i][length-1])
        print("data successfully converted")
        return np.array(new_data)
    except:
        print("Could not convert email to array")
        return data

def get_default_data():
    default_data = []
    for i in range(50):
        new_row = [0 for i in range(3002)]
        default_data.append(new_row)
    for i in range(50):
        new_row = [1 for i in range(3002)]
        default_data.append(new_row)
    return default_data


# Input: dataset | Output: training and test subsets of data
# Divide dataset to train and test sets with features and labels
def split_data(data):
    # divide data to train and test sets
    print("Splitting data...")
    feature_no = len(data[0])
    np.random.shuffle(data)
    holdout = int(0.7 * len(data))
    training_set = data[:holdout]
    test_set = data[holdout:]
    X_train = [row[1:feature_no-1] for row in training_set]
    y_train = [row[feature_no-1] for row in training_set]
    X_test = [row[1:feature_no-1] for row in test_set]
    y_test = [row[feature_no-1] for row in test_set]
    print("Data split complete")
    return [X_train, y_train, X_test, y_test]


# Input: None | Output: ML model
# Using user input, select one ML model to implement: perceptron, support vector machine, k-neighbors classifier
def define_model():
    user_inputed_model = get_user_inputed_model()
    if user_inputed_model == 'p':
        model = get_perceptron_model()
        print("Perceptron model created")
    elif user_inputed_model == 's':
        model = get_svm_model()
        print("SVM model created")
    else: 
        model = get_kneighbors_model()
        print("KNeighborClassifier model created")
    return model

def get_user_inputed_model():
    user_inputed_model = input("Enter type of model (Perception = p, K-neighbors = k, SVM = s: ")
    acceptable_input = ['p', 's', 'k']
    while not acceptable_input.__contains__(user_inputed_model):
        user_inputed_model = input("Enter type of model (Perception = p, K-neighbors = k, SVM = s: ")
    return user_inputed_model

def get_perceptron_model():
    return Perceptron()

def get_svm_model():
    return svm.SVC()

def get_kneighbors_model():
    neighbors = input("Enter number of neighbors: ")
    if not neighbors.isnumeric:
        print("Using default value of n_neighbors = 1")
        neighbors = 1
    neighbors = int(neighbors)
    return KNeighborsClassifier(n_neighbors = neighbors)


# Input: ML Model, train dataset | Output: None
# Fit the ML model using training data
def fit_model(model, X_train, y_train):
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete")


# Input: ML model, test dataset | Output: None
# Using test dataset, evaluate the accuracy, precision, recall, and f1score of the model
def evaluate_model(model, X_test, y_test):
    f1 = input("Get f1score? (y/n): ")
    if not f1 == 'y':
        accuracy = model.score(X_test, y_test)
        print_results(accuracy)
    else:
        [truepos, trueneg, falsepos, falseneg] = test_predictions(model, X_test, y_test)
        [precision, recall, f1score] = get_f1score(truepos, trueneg, falsepos, falseneg)
        accuracy = get_accuracy(truepos, trueneg, falsepos, falseneg)
        print_results(precision, recall, f1score, accuracy)

def get_f1score(truepos, trueneg, falsepos, falseneg):
    precision = truepos / (truepos + falsepos)
    recall = truepos / (truepos + falseneg)
    f1score = 2 * precision * recall / (precision + recall)
    return [precision, recall, f1score]

def get_accuracy(truepos, trueneg, falsepos, falseneg):
    accuracy = (truepos + trueneg) / (truepos + trueneg + falsepos + falseneg)
    return accuracy

def test_predictions(model, X_test, y_test):
    prediction = model.predict(X_test)
    truepos = trueneg = falsepos = falseneg = 0
    for i in range(len(prediction)):
        if prediction[i] == 0:
            if y_test[i] == 0:
                trueneg += 1
            else:
                falseneg += 1
        else:
            if y_test[i] == 1:
                truepos += 1
            else:
                falsepos += 1
    return [truepos, trueneg, falsepos, falseneg]

def print_results(accuracy):
    print("Accuracy: ", accuracy)

def print_results(precision, recall, f1score, accuracy):
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1score)
    print("Accuracy: ", accuracy)


# Input: ML model | Output: None
# Save ML model to a file given by user
def user_save_model(model):
    save = input("Save model? (y/n): ")
    if save == 'y':
        save_model(model)

def save_model(model):
    name = input("Enter file name to save to: ")
    try:
        f = open(os.path.join("ML Models", name),"wb")
        pickle.dump(model,f)
        f.close()
        print("model succesfully saved as ", name)
    except:
        print("model could not be saved")



main()