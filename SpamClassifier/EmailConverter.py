# Email Converter
# Provides tool to convert text to np array

import numpy as np 
import pandas as pd 
import csv
import os


def get_dictionary():
    try:
        header = get_header()
        word_dictionary = header[1:len(header)-1]
        return word_dictionary
    except:
        print("Could not get dictionary")


def get_header():
    try:
        header = []
        filename = "emails.csv"
        with open(os.path.join("Datasets", filename)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                header = row
                break
        return header
    except:
        print("Could not get header")
        print("Using sample word bank")
        return ["sample", "word", "bank"]

def load_emails(filename):
    try:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    except:
        print("Could not load data")

def convert_email(words, word_dictionary):
    freq_array = [0 for i in range(len(word_dictionary))]
    for word in words:
        try:
            index =  word_dictionary.index(word)
            freq_array[index] = freq_array[index] + 1
        except:
            continue
    return freq_array







