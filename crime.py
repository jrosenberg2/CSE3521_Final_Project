import pandas as pd 

#import the csv file
df = pd.read_csv('FinalProject\Police_DepartmentIncidents-_PreviousYear__2016.csv')
#drop all of the columns that we do not need
df = df.drop(['IncidntNum', 'X', 'Y', 'Location', 'PdId'], axis=1)
#Keep only the data where either someone was arrested or none
df1 = df[df.Resolution == 'ARREST, BOOKED']
df2 = df[df.Resolution == 'NONE']
df = pd.concat([df1, df2])



#df=df.sample(frac=1)