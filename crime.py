import pandas as pd
import math
import numpy as np
eps = np.finfo(float).eps

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
    print(train_data_frame.head())
    test_data_frame = pd.read_csv(test_data_url, header = None)
    test_data_frame.drop(index=test_data_frame.index[0], 
        axis=0, 
        inplace=True)
    print(test_data_frame.head())
    return(test_data_frame, train_data_frame)
    
    

#Function used to get the file entropy
def get_root_entropy(data_frame):
  entropy = 0
  values = data_frame[0].unique()
  for value in values:
    fraction = data_frame[0].value_counts() [value]/len(data_frame[0])
    entropy += -fraction*(np.log2(fraction))
  return entropy

#Function used to get the entropy of an attribute
def get_entropy(data_frame, attribute):
  entropy = 0
  unique_variables = data_frame[attribute].dropna().unique()
  target_variables = data_frame[0].unique()
  for variable in unique_variables:
    single_entropy = 0
    for target_variable in target_variables:
      num = len(data_frame[attribute][data_frame[attribute] == variable][data_frame[0] == target_variable]) #numerator
      den = len(data_frame[attribute][data_frame[attribute] == variable]) #denominator
      fraction = (num*1.0)/den
      single_entropy += -fraction*(math.log(fraction+eps))
    fraction2 = den/len(data_frame)
    entropy += -fraction2*single_entropy
  return(abs(entropy))

#Function used to calculate gain based on the file entropy and the entropy of a single attribute.
#Puts all gains in a list and then calculates and returns the attribute with the greatest gain
def get_gain(data_frame):
    Entropy_att = []
    IG = []
    for key in data_frame.keys()[1:]:
        IG.append(get_root_entropy(data_frame)-get_entropy(data_frame,key))
    #print(IG)
    return data_frame.keys()[1:][np.argmax(IG)]

#Function that returns a subtable of data
def get_subtable(data_frame, attribute, target):
  return data_frame[data_frame[attribute] == target].reset_index(drop=True)

#Function that implements the id3 algorithm to recursively build the decision tree
def build_tree(data_frame, tree = None):
  gain_node = get_gain(data_frame)
  print("step 1")
  values = np.unique(data_frame[gain_node])
  print("step 2")
  if tree is None:
    tree = {}
    tree[gain_node] = {}
  for value in values:
    subtable = get_subtable(data_frame, gain_node, value)
    clValue,counts = np.unique(subtable[0],return_counts=True)
    if len(counts)==1:
      tree[gain_node][value] = clValue[0]

    else:
      tree[gain_node][value] = build_tree(subtable)

  return tree

#Printed representation of the decision tree
def see_tree():
    import pprint
    pprint.pprint(t)

#Function used to go through the decision tree and make a prediction
def classify(instance, tree, default=None):
    attribute = list(tree.keys())[0]
    print(attribute)
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        
        if isinstance(result, dict): # this is a tree, delve deeper
            return classify(instance, result)
        else:
            return result # this is a label
    else:
        return default

def get_accuracy(t):
    test_data_frame['predicted'] = test_data_frame.apply(     # <---- test_data source
                                          classify, 
                                          axis=1, 
                                          args=(t, 'p') ) # <---- train_data tree

    print('Accuracy is ' + str( sum(test_data_frame[0]==test_data_frame['predicted'] ) / (1.0*len(test_data_frame.index)) ))
#Builds the tree based off of the train data
def main():
    (test_data_frame, train_data_frame) = get_data()
    print("made it here")
    t = build_tree(train_data_frame)
    print("made it here")
    get_accuracy(t)
if __name__ == '__main__':
    main()