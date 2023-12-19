import pandas as pd
import numpy as np
import os
import glob
import sys
import argparse
from scipy import stats

from sklearn import preprocessing
from sklearn import model_selection as ms

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit

from sklearn.svm import SVC
from sklearn import tree

from sklearn.metrics import balanced_accuracy_score as bas

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler



# Questions when this has worked:   Feature selection + more metrics 


def main():
    # Command line argument setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dt", "--decision_trees", help="use decision trees classifier")
    argParser.add_argument("-svm", "--support_vector_machine", help="use support vector machine classifier")
    args = argParser.parse_args()

    dataloader = Dataloader(sys.argv[2])

    if args.decision_trees is not None:
        decision_trees = Decision_Trees(dataloader)
        decision_trees.train()

    if args.support_vector_machine is not None:
        svm = SVM(dataloader)
        svm.train()


    print("Finished")



class Decision_Trees():

    """Decision Tree classification"""
    
    def __init__(self, dataloader):
        self.train_set, self.target = dataloader.get()

    def train(self):
        print("Classifier: Decision Trees")

        # TODO: implement decision tree classifier (https://scikit-learn.org/stable/modules/tree.html)

        #X = self.train_set
        #y = self.target

        X = np.random.randn(20,3)                                          #Dummy
        y = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])            #Dummy

        X = preprocessing.normalize(X, norm='l2')
        X_train, X_val, y_train, y_val = ms.train_test_split(X, y, train_size=0.8, random_state=42)

        clf = tree.DecisionTreeClassifier(max_depth=6, max_leaf_nodes=13)
        clf.fit(X_train, y_train)

        cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        cross_val_score = ms.cross_val_score(clf, X_train, y_train, cv=cv)

        predict = clf.predict(X_val)

        print()
        print('cross validation scores: ',cross_val_score)
        print('accuracy score: ',bas(y_val,predict))




class SVM():

    """Support Vector Machine Classification"""

    def __init__(self, dataloader):
        self.train_set, self.target = dataloader.get()

    def train(self):
        print("Classifier: Support Vector Machine")

        #(https://scikit-learn.org/stable/modules/svm.html)

        #X = self.train_set
        #y = self.target

        X = np.random.randn(20,3)                                          #Dummy
        y = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])            #Dummy

        X = preprocessing.normalize(X, norm='l2')
        X_train, X_val, y_train, y_val = ms.train_test_split(X, y, train_size=0.8, random_state=42)

        svc = SVC()

        rand_list = {'clf__kernel':['rbf'],
                     'clf__C': stats.uniform(1, 100),
                     'clf__gamma': stats.uniform(0.01, 1)}
        
        
        grid_list = {'clf__kernel':['rbf'],
                     'clf__C': np.arange(2, 10, 2),
                     'clf__gamma': np.arange(0.1, 1, 0.2)}

        model = Pipeline([
            ('sampling', RandomOverSampler(random_state=0)),
            #('sampling', RandomUnderSampler(random_state=0)),
            #('sampling', SMOTE()),
            ('clf', svc)
            ])
        
        cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        
        # Random search
        clf = RandomizedSearchCV(model, param_distributions = rand_list, n_iter = 20, n_jobs = 4, cv = cv, random_state = 42, scoring = 'balanced_accuracy', verbose=49) 
        clf.fit(X_train, y_train) 
        clf.cv_results_
        
        """
        # Gridsearch
        clf = GridSearchCV(model, param_grid = grid_list, n_jobs = 4, cv = cv, scoring = 'balanced_accuracy', verbose=49) 
        clf.fit(X_train, y_train) 
        clf.cv_results_
        """

        predict = clf.predict(X_val)

        print('accuracy score: ',bas(y_val,predict))




class Dataloader():

    """Load and prepare Dataloader for sklearn classifiers"""


    def __init__(self,path):
        """Get all train data csv files from folder"""

        self.path = os.path.join(os.getcwd(), str(path))


    def get(self):
        """return train data csvs concatenated as one numpy array and targets"""

        train_set = None
        target = None

        train_data_dict = {}

        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, str(filename))
            file = pd.read_csv(file_path, low_memory=False)
            train_data_dict[filename] = file

        recipes = train_data_dict['recipes.csv']
        requests = train_data_dict['requests.csv']
        reviews = train_data_dict['reviews.csv']
        diet = train_data_dict['diet.csv']
        

        #TODO: Data tidying and transformation into train_set of type numpy array, split off of target


        return train_set, target



main()
