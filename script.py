import pandas as pd
import numpy as np
import os
import glob
import sys
import argparse


# Questions when this has worked:   Feature selection + more metrics 


def main():
    # Command line argument setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dt", "--decision_trees", help="use decision trees classifier")
    argParser.add_argument("-svm", "--support_vector_machine", help="use support vector machine classifier")
    args = argParser.parse_args()

    dataloader = Dataloader(sys.argv[1])

    if args.decision_trees is not None:
        decision_trees = Decision_Trees(dataloader)
        decision_trees.train()

    if args.support_vector_machine is not None:
        svm = SVM(dataloader)
        svm.train()


    print("Finished")



class Decision_Trees():
    
    def __init__(self, dataloader):
        self.train_set, self.target = dataloader.get()

    def train(self):
        print("Decision Trees!")
        print("Balanced Accuracy Score: 100% ;)")

        # TODO: implement decision tree classifier (https://scikit-learn.org/stable/modules/tree.html)

        pass



class SVM():

    def __init__(self, dataloader):
        self.train_set, self.target = dataloader.get()

    def train(self):
        print("Support Vector Machines!")
        print("Balanced Accuracy Score: 100% ;)")

        # TODO: implement support vector machine classifier (https://scikit-learn.org/stable/modules/svm.html)

        pass




class Dataloader():
    """Load and prepare Dataloader for sklearn classifiers"""

    def __init__(self,path):
        """Get all train data csv files from folder"""

        self.train_data = glob.glob(os.path.join(path, "*.csv"))

    def get(self):
        """return train data csvs concatenated as one numpy array and targets"""

        train_set = None
        target = None
        
        for file in self.train_data:
            df = pd.read_csv(file).to_numpy()
            pass

        # TODO: implement train data concatenation, conversion to one numpy array, split in train set and target

        pass

        return train_set, target



main()
