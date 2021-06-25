# Predict
# User-interactive tool to predict spam emails in real time using saved ML models

import numpy as np 
import EmailConverter
import pickle
import os


def main():
    #load model
    filename = input("Enter model filename to load: ")
    try:
        model = pickle.load(open(os.path.join("ML Models", filename), 'rb'))
        print("Model succesfully loaded")
    except:
        print("Could not load model")
        return
    
    # prepare email content
    email_contents = input("Enter email contents: ")
    words = email_contents.split()
    word_array = EmailConverter.convert_email(words)

    #use model to predict
    prediction = model.predict([word_array])
    if prediction[0] == 1:
        print("Spam")
    else:
        print("Not Spam")


main()