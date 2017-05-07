#data processing

#importing the library

import numpy as np 
import matplotlib.pyplot as plt # to plot chart
import pandas as pd #this is the best library to import data set

#---------------------------#----------------------------------------------

                    #importing the dataset
dataset = pd.read_csv('Data.csv')

#here we will create independent variable matrix X
# here in this dataset the independent variables are country, age , salary
X = dataset.iloc[:, :-1].values #here ':' represents we have taken all rows and ':-1' represents we have taken all columns except the last one
 
#now we will create dependent variable vector or matrix Y 
# here in this dataset the independent variable is 'Purchased'
Y = dataset.iloc[:,3].values #'3' represent we have selected only last column which is purchased column

#----------------------#-------------------------------------------------------

                    #Taking Care of Missing Data
# As we can see in our dataset some blocks are missing we have to handle this bcz if we dont the program is going to through error
# so the Strategy that we use here to fix missing data is we are going to take 'MEAN' of all the data present in that respective column and fill that mean inside the empty block
# we are going to use library to find MEAN for us

from sklearn.preprocessing import Imputer # sklearn is 'scikit-learn' it contains amazing libraries to make machine learnning models

# from sklearn we take this 'preprocessing' library that contains classes method to preprocess any type of data
# here 'Imputer' class will allow us to take care of all the missing data

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0) #here we created object to find mean of column
imputer = imputer.fit(X[:,1:3]) # here we fit imputer object to matrix X #column 1 and 2 are having nan values that is why 1:3 
X[:,1:3] = imputer.transform(X[:,1:3]) # here we replace data in matrix X with transformed data came from imputer
                   
#----------------------------------#------------------------------------------------------

                    #Encoding Categorical data                    
#here country and purchased are the categorical data german,spain and yes,no respectively
#here also we will use scikit-learn library

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#In this case it is good to encode categorical variables to specific mathematical no. but it should not happen that machine would compare this categories according to the ascending or descending order of the no,. is given
# to avoid this we will use dummy variables which will include column for each category and if the value is equal to that category we will make that specific row '1'
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray();

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#--------------------------------------------#------------------------------------------

                #spliting the dataset into training set and test set
# here we are going to build our machine learning model on training set
# and we are going to test our ML model on the test set

#here we are going to use cross_validation library from sklearn

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#-------------------------------------#--------------------------------------
   
                             #Feature Scaling
#here we are going to scale all variables of dataset which is comparable in the same scale 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
