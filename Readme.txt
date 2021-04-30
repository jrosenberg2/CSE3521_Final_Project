Jacob Rosenberg
Jared Schneider
William Xu
4/30/2021
CSE 3521 Final Project

File Descriptions:
data.py - Takes the data from Kaggle (stored as csv) and mainpulates it to only include the columns we use. Saves the new train and test data files as csvs in the local folder
Police_Department_Incidents_-_Previous_Year__2016_.csv - Full dataset from Kaggle (https://www.kaggle.com/roshansharma/sanfranciso-crime-dataset)
TestDatas.csv - Created test data set
TrainData.csv - Created train data set
ID3.py - Implementation and application of the ID3 algorithm on the data
LogisticRegression.py - Implementation and application of the Logistic Regression algorithm on the data

How to run:
*Note: Code can also be run from the web using google colab by clicking the "Runtime" tab followed by "Run all"
    **The ID3 implementation on google colab can be found at https://colab.research.google.com/drive/1PR_NQmtBx8vS1gfSH0MhsUj-f7KQvz1x?usp=sharing
    **The Logistic Regression implementation on google colab can be found at https://colab.research.google.com/drive/16rL_gsDUFXiayUWrpscD5UAinPOfUXn0?usp=sharing
1. Open Folder in Editor (VSCode)
2. Ensure you have a recent version of Python (ex. Python 3.9.4) installed.
        You can find your version of Python in your files or through the command line
3. Open a terminal/PowerShell session
        Click the "Terminal" menu tab on the upper naviagion bar of VSCode.
4. cd in the terminal into the path of the folder
5. type either "py <name of python file>" to run that file
        Ex. >>>py ID3.py
        
The accuracy of the test will be printed in the command line