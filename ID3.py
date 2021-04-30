import pandas as pd
import math
import numpy as np
eps = np.finfo(float).eps
from sklearn import tree
#from matplotlib import pyplot as plt #uncomment impot statement when printing the decision tree
from sklearn.model_selection import train_test_split

def get_data():
    #Urls used to get to the data from github
    train_data_url = 'TrainData.csv'
    test_data_url = 'TestData.csv'

    #pd.read used to read the data into a dataframe 
    #header = None allows for column numbers to be used to identify the data

    train_data_frame = pd.read_csv(train_data_url, header = 0)
    #train_data_frame.drop(index=train_data_frame.index[0], 
    #        axis=0, 
    #        inplace=True)
    train_data_frame = train_data_frame.dropna()
    train_data_frame.head()

    test_data_frame = pd.read_csv(test_data_url, header = 0)
    #test_data_frame.drop(index=test_data_frame.index[0], 
    #        axis=0, 
    #        inplace=True)
    test_data_frame = test_data_frame.dropna()
    test_data_frame.head()
    return (train_data_frame, test_data_frame)

#onehot encodes all of the data to eliminate the potetnial weight bias stemming from numerical grouping of features
def oneHotData(df):
    y = pd.get_dummies(df.Category, prefix='Category')
    df = pd.concat([df, y], axis=1)

    #OneHot the Category feature
    df= df.drop(['Category'], axis=1)

    y = pd.get_dummies(df.DayOfWeek, prefix='DayOfWeek')
    df = pd.concat([df, y], axis=1)

    #OneHot the DayOfWeek feature
    df= df.drop(['DayOfWeek'], axis=1)

    y = pd.get_dummies(df.Time, prefix='Time')
    df = pd.concat([df, y], axis=1)

    #OneHot the Time feature
    df= df.drop(['Time'], axis=1)

    y = pd.get_dummies(df.PdDistrict, prefix='PdDistrict')
    df = pd.concat([df, y], axis=1)
    #droping the country column 
    df = df.drop(['PdDistrict'], axis=1)
    return df

#Formats the data to be ready to input into the sklearn decision tree function
def format_data(train_data_frame, test_data_frame):
    from sklearn.tree import DecisionTreeClassifier

    train_data_frame = oneHotData(train_data_frame)
    train_data_frame['Resolution'] = train_data_frame['Resolution'].replace({"ARREST, BOOKED": 1, "NONE": 0})
    test_data_frame = oneHotData(test_data_frame)
    test_data_frame['Resolution'] = test_data_frame['Resolution'].replace({"ARREST, BOOKED": 1, "NONE": 0})
    dec_tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=len(train_data_frame.columns))

    return (train_data_frame, test_data_frame, dec_tree)


#trains the decision tree using the sklearn ID3 decision tree algorithm. returns the decision tree
def train_dec_tree(dec_tree, train_data_frame):
    train_X = train_data_frame.drop("Resolution",axis=1)
    train_Y = train_data_frame["Resolution"]

    dec_tree.fit(train_X,train_Y)
    return dec_tree

#classifies the data based on the decision tree created from the training data. returns the prediction
def test_data(dec_tree, test_data_frame):
    test_X = test_data_frame.drop("Resolution", axis=1)
    test_Y = test_data_frame["Resolution"]

    y_pred = dec_tree.predict(test_X)
    return (test_Y, y_pred)

#returns the accuracy of the ID3 algorithm implementation
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

    #uncomment the next 2 lines to print the decision full decision tree
    #fig = plt.figure(figsize=(75, 50))
    #tree.plot_tree(dec_tree, fontsize=10, feature_names=list(test_data_frame.columns), class_names=list(['NONE', 'ARREST, BOOKED'])) 


if __name__ == '__main__':
    main()


