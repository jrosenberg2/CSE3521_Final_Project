import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_data():
    #Urls used to get to the data from the folder
    train_data_url = 'TrainData.csv'
    test_data_url = 'TestData.csv'

    #pd.read used to read the data into a dataframe 
    #header = None allows for column numbers to be used to identify the data
    train_data_frame = pd.read_csv(train_data_url, header = 0)
    train_data_frame = train_data_frame.dropna()
    test_data_frame = pd.read_csv(test_data_url, header = 0)
    test_data_frame = test_data_frame.dropna()
    df = pd.concat([train_data_frame, test_data_frame])
    return(df)

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

def logisticRegression(df):

    #Makes our 
    df['Resolution'] = df['Resolution'].replace({"ARREST, BOOKED": 1, "NONE": 0})

    x = df.drop('Resolution',axis = 1)
    y = df.Resolution

    #This will separate 25%( default value) of the data into a subset for testing part and the remaining 75% will be used for our training subset.
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

    #ensures the convergence of the LogisticRegression model
    logistic_regression = LogisticRegression(max_iter = 150)

    logistic_regression.fit(x_train, y_train)

    #Predict whether an arrest will be made or not
    y_pred = logistic_regression.predict(x_test)
    
    return (y_test, y_pred)

def getAccuracy(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)

#Builds the tree based off of the train data
def main():
    df = get_data()
    df = oneHotData(df)
    (y_test, y_pred) = logisticRegression(df)
    getAccuracy(y_test, y_pred)

if __name__ == '__main__':
    main()
