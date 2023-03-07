import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class LRclassifier():
    """
    Creates and Tests a Classifier to check if a comment made by user is toxic.
    """
    data_frame = None
    vectorizer = None
    X = None
    y = None
    X_train = None
    X_test = None
    X_vectorized = None
    y_train = None
    y_test = None
    model = None
    threshold = None
    def __init__(self) -> None:
        """
        set up variables.
        """
        self.vectorizer = TfidfVectorizer()

    def read_file(self, file):
        """
        Reads the train file and extracts relevant information drops unecessary data as well.
        """
        try:
            print("[PROCESS] Reading the Train CSV file.")
            self.data_frame = pd.read_csv(file) #read the file.
            if "test" not in file.lower():
                self.data_frame = self.data_frame.drop(columns=["id", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]) #drop the irrelevant coloumns\
            else:
                self.data_frame = pd.DataFrame(self.data_frame["comment_text"], columns=["comment_text"])
            print("[INFO]    Read the CSV file, and extracted the relevant dataframe.")
        except Exception as e:
            print("[ERR] The following error occured while trying to read the Train csv file: "+str(e))

    def normalize_single_comment(self,comment) -> str:
        """
        This method normalizes an individual instance of comment.
        """
        try:
            if comment is None: #check if a comment is None.
                return None
            comment = comment.lower() #lower it's case for case consistency.
            comment = re.sub("'", "", comment) #convert words like can't to cant, didn't to didnt
            comment = " ".join(re.findall("[a-z]+", comment)) #capture only text data.

            if len(comment.strip(" ")) == 0 or len(comment.rstrip()) == 0: #check if above steps produce an empty string.
                return None
            else:
                return comment #if all goes right return a string.
        except Exception as e:
            print("[ERR] The following error occured while trying normalize a particular comment: "+str(e))
    
    def normalize(self) -> None:
        """
        normalize method normalizes the previously extracted relevant information.
        """
        try:
            print("[PROCESS] Normalizing the comment_text, from the data frame.")
            self.data_frame["comment_text"] = self.data_frame["comment_text"].apply(self.normalize_single_comment)
            #print(self.data_frame)
            print("[INFO]    Normalized the data successfully!")
        except Exception as e:
            print("[ERR] The following error occured while tryig to normalize the data_frame: "+str(e))

    def create_word_embeddings(self) -> None:
        """
        Creates feature vectors for our LR model.
        """
        try:
            print("[PROCESS] Creating Feature vectors for the data, please wait this might take some time.")
            toxicData = self.data_frame.loc[self.data_frame["toxic"] == 1]
            print("[INFO]    Extracted toxic comments.")

            nontoxicData = self.data_frame.loc[self.data_frame["toxic"] == 0][:60000]
            print("[INFO]    Extracted 60,000 nontoxic comments.")
            
            self.data_frame = pd.concat([toxicData, nontoxicData])
            x_sample = [ str(x) for x in self.data_frame["comment_text"]]
            self.y= self.data_frame["toxic"]
            print("[PROCESS] Transforming the comments into Feature Vectors please wait this migth take time!")
            self.X = self.vectorizer.fit_transform(x_sample)
            print("[INFO]    Feature Vectors created successfully!")
        except Exception as e:
            print("[ERR] The following error occured while trying to create feature vectors: "+str(e))

    def load_model(self) -> bool:
        """
        This model tries to load a previously trained classifier to save time.,
        """
        try:
            print("[INFO]    Trying to load a previously trained model!")
            self.model = joblib.load("LRclassifier.joblib")
            print("[INFO]    Aha! A previously trained classifier exists and it has been initiated!")
            return True
        except:
            print("[INFO]    Sorry! No previously Trained model could be found, will create a new one!")

    def save_model(self, model):
        """
        This method saves a model that was just trained newly.
        """
        try:
            print("[INFO]    Saving the newly trained model as: LRclassifier.joblib within the same directory.")
            joblib.dump(model, "LRclassifier.joblib")
        except Exception as e:
            print("[ERR] The following error occured while trying to save the model: "+str(e))


    def train_LR_model(self, path_to_train_file = "./datasets/train.csv") -> LogisticRegression():
        """
        This method trains a LR model with the training data provided and returns a LR model fitted for the same.
        """
        try:
            print("-"*50)
            print(" LR Training by Harsha Vardhan Khurdula :)")
            print("-"*50)
            self.read_file(path_to_train_file) #reads the file and extracts relevant information.
            self.normalize() #normalize the data
            self.create_word_embeddings() #create the feature vectors.
            print("[INFO]    Defining the Threshold to be: 0.3 Manually.")
            print("[PROCESS] Splitting this data, inorder get model metrics!")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
            self.threshold = 0.3
            if self.load_model():
                return self.model
            else:
                
                print("[PROCESS] Creating a new instance of Logistic Regression.")
                print("[INFO]    Extending the max iterations to 1000 from 100 to this instance.")
                self.model = LogisticRegression(max_iter=1000)
                print("[INFO]    Created a new instance of Logistic Regression Class.")
                print("[PROCESS] Training LRmodel, please wait this might take some time.")
                self.model = self.model.fit(self.X_train, self.y_train)
                print("[INFO]    Model has been trained Successfully, returning the instance of this model!")
                self.save_model(self.model)
                return self.model
        except Exception as e:
            print("[ERR] The following error occured while trying to train a LR model: "+str(e))
    
    def traditional_test(self, model) -> float():
        """
        This method tests the trained model on split data and returns the accuracy score of the model.
        The split will be based on vectorizated Trained X as well.
        """
        try:
            print("-"*50)
            print(" LR Model Metrics ")
            print("-"*50)
            predictions = model.predict_proba(self.X_test)[:, 1]
            predictions = np.where(predictions >= self.threshold, 1, 0)
            _accuracy = accuracy_score(predictions, self.y_test)
            _precision, _recall, _F1_score, _ = precision_recall_fscore_support(self.y_test, predictions, average='binary')
           
            print("[INFO]    Here are the LRmodel metrics: ")
            print("[METRIC] Accuracy Score: %f"%(_accuracy))
            print("[METRIC] Precision: %f"%(_precision))
            print("[METRIC] F1-Score: %f"%(_F1_score))
            print("[METRIC] Recall: %f"%(_recall))
            # plot_confusion_matrix(self.y_test, predictions, labels=["Not Toxic", "Is Toxic"])
            # plot_roc_curve(self.y_test,predictions)
            
        except Exception as e:
            
            print("[ERR] The following error occured while trying to test the model for accuracy! "+str(e))

    def make_prediction(self, comment, model) -> float():
        """
        This model makes a prediction, and returns a list in the following ways:
        returned value: list(comment, toxicity_score, classification)
        """
        try:
            
            normalized_comment = self.vectorizer.transform([comment])
            toxicity_score = model.predict_proba(normalized_comment)[:, 1]
            classification = np.where(toxicity_score >= self.threshold, 1, 0)
            result = [comment, toxicity_score[0], classification[0]]
            return result
        except Exception as e:
            print("[ERR] The following error occured while trying to classify the comment:%s ERR:%s "%(comment, str(e)))

    def test_LR_model(self, path_to_test_file= "./datasets/test.csv", LRmodel=None) -> None:
        """
        This method, is used to test the model and yield outputs for the provided testset.
        This method must also yield the metrics, that are statistics computation which quantify the model's performance.
        """
        try:
            
            if LRmodel is None: #check if a model has been given as input else throw an exception asking for one.
                raise Exception("No Logisitic Regression model has been passed to the test method, please provide the appropriate parameters.")
            
            print("[INFO]    Analyzing the model metrics please wait.")
            self.traditional_test(LRmodel) #Aquire model metrics.

            print("-"*50)
            print(" LR Testing ")
            print("-"*50)
            print("[PROCESS] Reading data from Test file!")
            self.read_file(path_to_test_file) #read the test file.
            print("[PROCESS] Normalizing the Comments please wait!")

            self.normalize() #normalize the comments within the test file.
            # print(self.data_frame.head())6
            result_data = list()
            print("[PROCESS] Performing classifications for testset please wait this might take a while!")
            total = len(self.data_frame["comment_text"])
            counter = 0
            for comment in self.data_frame["comment_text"]:
                if comment is None:
                    continue
                prediction = self.make_prediction(comment, LRmodel)
                result_data.append(prediction)
                counter = counter + 1
                progress = "[PROCESS] Progess: " + str(int((counter/total)*100)) + "% DONE."
                print(progress, end="\r")
            data_frame= pd.DataFrame(result_data, columns=["comment_text", "toxicity_score", "toxic"])
            print("[INFO]    ALL COMPUTATIONS DONE.")
            print("[INFO]    Writing results to _output.csv within the same directory!")
            data_frame.to_csv("_output.csv")
            print("[SUCCESS] ALL RESULTS HAVE BEEN DUMPED TO: _output.csv")
        except Exception as e:
            print("[ERR] The following error while trying to perform test on the LRmodel: "+str(e))

