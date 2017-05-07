#Data Processing

#in R language unlike python idexing starts from index '1'

#importing the Dataset
dataset = read.csv('Data.csv')  

#Taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Age
                     )
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Salary
)

#Encoding Categorical Data
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No','Yes'),
                         labels = c(0,1))

#splitting the dataset into training set and test set
# we are going toneed caTool library included here we can import it through code or mark the package in the packages section

set.seed(123) #req to set to get specific sequence of result
split = sample.split(dataset$Purchased , SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)


# Feature Scaling

#training_set = scale(training_set) if we try to scale all the columns in this set we will get an error because here two categorical variables are assigned factors they are not numericals

training_set[,2:3] =  scale(training_set[,2:3])
test_set[,2:3] =  scale(test_set[,2:3])












