import pandas as pd
import numpy as np
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


# read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

# #The dataset contains attributes such as player name and team name. 
# #We know that they are not useful for classification and thus do not 
# #include them as features. 
# # feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
# #     '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
# #     'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']


# removing age, g, gs attirbutes as well as removing all percentage attributes
feature_columns = [ 'MP', 'FG', 'FGA', '3P', '3PA', \
    '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']


# pandas dataframe
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

# specifying folds of data set
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(nba_feature, nba_class)


training_accuracy = []
testing_accuracy = []


for train_index, test_index in skf.split(nba_feature, nba_class ):
	

	# getting fold data as test set and train set
	train_feature, test_feature = nba_feature.iloc[train_index], nba_feature.iloc[test_index]
	train_class, test_class = nba_class.iloc[train_index], nba_class.iloc[test_index]

	# print(test_feature, test_class)

	# creating combined pandas data frame with features and class
	combined_tainining_data_set = pd.concat([train_feature, train_class], axis=1 )
	combined_test_data_set = pd.concat([test_feature, test_class], axis=1)

	# removing players who played less than 5 min
	refined_combined_training_data_set = combined_tainining_data_set.loc[combined_tainining_data_set['MP'] > 5]


	# getting maximum accuracy for C =-0.7 from research
	clf = svm.SVC(kernel='linear', C=0.7)
	#  training model
	clf.fit(refined_combined_training_data_set[feature_columns], refined_combined_training_data_set[class_column])
	# getting traning set accuracy
	training_accuracy.append(clf.score(refined_combined_training_data_set[feature_columns], refined_combined_training_data_set[class_column]))
	#  getting test set accuracy
	testing_accuracy.append(clf.score(combined_test_data_set[feature_columns], combined_test_data_set[class_column]))

print("k fold Training set score: {}".format(training_accuracy))
print("k fold Test set score: {}".format(testing_accuracy))
print("k fold Mean training accuracy: {}".format(sum(training_accuracy)/len(training_accuracy)))
print("k fold Mean testing accuracy :{}".format(sum(testing_accuracy)/len(testing_accuracy)))

# --------------------------------------------------------------------------------------------------------------------------------

#  creating final model using train, test split method

train_feature, test_feature, train_class, test_class = train_test_split(
	    nba_feature, nba_class, stratify=nba_class, train_size=0.75, test_size=0.25)


# cmbining training data
combined_tainining_data_set = pd.concat([train_feature, train_class], axis=1 )

# removing players who played less than 5 min
refined_combined_training_data_set = combined_tainining_data_set.loc[combined_tainining_data_set['MP'] > 5]

# C = 0.7 SVM regularization parameter
clf = svm.SVC(kernel='linear', C= 0.7)
# training classifier
clf.fit(refined_combined_training_data_set[feature_columns], refined_combined_training_data_set[class_column])

print("-----75%, 25% model------")

print("Training set score: {:.3f}".format(clf.score(train_feature, train_class)))
print("Test set score: {:.3f}".format(clf.score(test_feature, test_class)))

prediction = clf.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))








