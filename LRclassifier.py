import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

class LRclassifier():
    """
    Creates and Tests a Classifier to check if a comment made by user is toxic.
    """
    data_frame = None
    def __init__(self) -> None:
        pass

    def read_file(self, file):
        """
        Reads the train file and extracts relevant information drops unecessary data as well.
        """
        try:
            print("[PROCESS] Reading the Train CSV file.")
            self.data_frame = pd.read_csv(file) #read the file.
            self.data_frame = self.data_frame.drop(columns=["id", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]) #drop the irrelevant coloumns\
            print("[INFO] Read the CSV file, and extracted the relevant dataframe.")
        except Exception as e:
            print("[ERR] The following error occured while trying to read the Train csv file: "+str(e))

    def normalize(self) -> None:
        """
        normalize method normalizes the previously extracted relevant information.
        """
        try:
            print("[PROCESS] Normalizing the comment_text, from the data frame.")
            pass
        except Exception as e:
            print("[ERR] The following error occured while tryig to normalize the data_frame: "+str(e))


    def train_LR_model(self, path_to_train_file = "./datasets/train.csv") -> LogisticRegression():
        """
        This method trains a LR model with the training data provided and returns a LR model fitted for the same.
        """
        try:
            self.read_file(path_to_train_file) #reads the file and extracts relevant information.
        except Exception as e:
            print("[ERR] The following error occured while trying to train a LR model: "+str(e))



obj = LRclassifier()
obj.train_LR_model()