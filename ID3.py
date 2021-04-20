import pandas as pd
import math
import numpy as np
eps = np.finfo(float).eps

#Implementation of the ID3 Algorithm
def get_data():
    #Urls used to get to the data from the folder
    train_data_url = 'TrainData.csv'
    test_data_url = 'TestData.csv'

    #pd.read used to read the data into a dataframe 
    #header = None allows for column numbers to be used to identify the data

    train_data_frame = pd.read_csv(train_data_url, header = None)
    train_data_frame.drop(index=train_data_frame.index[0], 
            axis=0, 
            inplace=True)
    train_data_frame = train_data_frame.dropna()
    train_data_frame.head()

    test_data_frame = pd.read_csv(test_data_url, header = None)
    test_data_frame.drop(index=test_data_frame.index[0], 
            axis=0, 
            inplace=True)
    test_data_frame = test_data_frame.dropna()
    test_data_frame.head()
    return(train_data_frame, test_data_frame)

def format_data(train_data_frame, test_data_frame):
    from sklearn.tree import DecisionTreeClassifier
    dec_tree = DecisionTreeClassifier(criterion="entropy")

    for col in train_data_frame.columns:
        train_data_frame[col]=train_data_frame[col].astype('category').cat.codes

    for col in test_data_frame.columns:
        test_data_frame[col]=test_data_frame[col].astype('category').cat.codes
    
    return (train_data_frame, test_data_frame, dec_tree)

def train_dec_tree(dec_tree, train_data_frame):
    train_X = train_data_frame.drop(0,axis=1)
    train_Y = train_data_frame[0]

    dec_tree.fit(train_X,train_Y)
    return dec_tree

def test_data(dec_tree, test_data_frame):
    test_X = test_data_frame.drop(0, axis=1)
    test_Y = test_data_frame[0]

    y_pred = dec_tree.predict(test_X)
    return (test_Y, y_pred)

def get_accuracy(test_Y, y_pred):
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(test_Y, y_pred)
    print(accuracy)

#Builds the tree based off of the train data
def main():
    (train_data_frame, test_data_frame) = get_data()
    (train_data_frame, test_data_frame, dec_tree) = format_data(train_data_frame, test_data_frame)
    dec_tree = train_dec_tree(dec_tree, train_data_frame)
    test_Y, y_pred = test_data(dec_tree, test_data_frame)
    get_accuracy(test_Y, y_pred)
if __name__ == '__main__':
    main()


