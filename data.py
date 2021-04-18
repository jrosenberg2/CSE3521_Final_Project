import pandas as pd 

def main():
    df = get_data()
    (train_data, test_data) = split_data(df)
    save_new_data(train_data, test_data)

def get_data():
    #import the csv file
    df = pd.read_csv('Police_Department_Incidents_-_Previous_Year__2016_.csv')
    #drop all of the columns that we do not need (keep the ones that have data we can use in an ID3 algorithm)
    df = df.drop(['IncidntNum', 'Descript', 'Date', 'Address', 'X', 'Y', 'Location', 'PdId'], axis=1)

    #Keep only the data where either someone was arrested or no resolution, which is approx. 98% of the data.
    df1 = df[df.Resolution == 'ARREST, BOOKED']
    df2 = df[df.Resolution == 'NONE']
    df = pd.concat([df1, df2])

    for n in range(len(df)):
        time = list(df.Time.iloc[n])

        """ If 30 mins of the hour have passed, round up to the next hour """
        if time[3] == '3' or time[3] == '4' or time[3] == '5':
            time[3] = '0'
            time[4] = '0'
            temp = (int(time[1]) + 1)

            """ If rounding up caused the second digit to go from 9 to 10, make the second digit 0 and increment the first """
            if(temp == 10):
                temp = 0
                time[0] = str(int(time[0]) + 1)
                time[1] = str(temp)

                """ If the second digit was not rounded to 10 """
            else:   
                time[1] = str(temp)

            """ If 30 mins of the hour have not passed, round down """
        else:
            time[3] = '0'
            time[4] = '0'

        """ If rounding caused 24hour clock to read '24:00', then make that '00:00' as it should be """
        if time[0] == '2' and time[1] == '4':
            time[0] = '0'
            time[1] = '0'
        df.Time.iloc[n] = "".join(time)
        return df

def split_data(df):
    # file was sorted into arrested/none, so shuffle to randomize
    df = df.sample(frac=1)
    split = int(len(df)*0.8)
    train_data = df.iloc[0:split]
    test_data = df.iloc[split+1:len(df)-1]

    # Counts number of arrests in each dataframe. Prints (total#arrests, 80% of total, train#arrests, 20% of total, test#arrests) 
    # to verify that the 80% of the data has approximately 80% of the total arrests. We obviously dont want a test file with no 
    # arrests on it. If the second and third numbers(likewise the fouth and fifth numbers) are not very close, run the program 
    # again and get a new test/train split
    testarrest = 0
    for n in range(len(test_data)):
        if test_data.Resolution.iloc[n] == 'ARREST, BOOKED':
            testarrest += 1
    trainarrest = 0
    for n in range(len(train_data)):
        if train_data.Resolution.iloc[n] == 'ARREST, BOOKED':
            trainarrest += 1
    totalarrest = 0
    for n in range(len(df)):
        if df.Resolution.iloc[n] == 'ARREST, BOOKED':
            totalarrest += 1
    print(totalarrest, totalarrest*0.8, trainarrest, totalarrest*0.2, testarrest)
    return(train_data, test_data)

def save_new_data(train_data, test_data):
    #save to seperate csv file to be used many times without changing the ordering of the rows, so that accuracy is always the same
    train_data.to_csv("TrainData.csv", index=False)
    test_data.to_csv("TestData.csv", index=False)

if __name__ == '__main__':
    main()


