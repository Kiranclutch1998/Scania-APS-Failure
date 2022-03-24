"""
This is the Entry point for Prediction.
"""

# Doing the necessary imports
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocessing
from file_operations import file_methods, load_model
from application_logging import logger


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data = pd.read_csv('EDA + training/aps_failure_training_set.csv')

            """doing the data preprocessing"""

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            # replacing 'na' values with np.nan as discussed in the EDA part

            data = data.replace('na', np.NaN)

            data['class'] = data['class'].replace(['neg','pos'],[0,1])

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='class')

            # object datatype to float datatype
            X = X.apply(pd.to_numeric)

            # removing unused columns
            col = ['cd_000', 'ab_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000', 'br_000', 'cr_000']
            X = preprocessor.remove_columns(data, col)

            # replacing null values with mean
            X = preprocessor.handleMissingValues(X)

            X = preprocessor.scale_numerical_columns(X)

            X = preprocessor.pcaTransformation(X)

            # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3, random_state=36)

            model = SVC(kernel='rbf', C=1.0, random_state=0)
            model.fit(x_train, y_train)

            # saving the model to the directory.
            file_op = file_methods.File_Operation(self.file_object, self.log_writer)

            file_op.save_model(model, 'SVC')

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, e)
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception
