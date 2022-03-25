import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
        This class shall  be used to clean and transform the data before prediction.

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self, data, columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns
        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)  # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in remove_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def handleMissingValues(self, data):
        self.data = data
        try:
            imputer = SimpleImputer(missing_values=np.NaN, strategy='median', copy=True)
            self.new_array = imputer.fit_transform(self.data)  # impute the missing values
            # convert the nd-array returned to the step above to a Dataframe
            self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)
            self.logger_object.log(self.file_object,
                                   'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Columns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels=label_column_name,
                               axis=1)  # drop the columns specified and separate the feature columns
            self.Y = data[label_column_name]  # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def pcaTransformation(self, data):
        try:
            self.data = data
            pca = PCA(n_components=104)
            new_data = pca.fit_transform(self.data)
            principal_x = pd.DataFrame(new_data, index=self.data.index)
            self.logger_object.log(self.file_object,
                               'pcaTransformation Successful.Exited the pcaTransformation method of the Preprocessor class')
            return principal_x

        except Exception as e:
            self.logger_object.log(self.file_object,
                           'Exception occurred in pcaTransformation method of the Preprocessor class. Exception message:  ' + str(
                               e))
            self.logger_object.log(self.file_object,
                           'pcaTransformation Unsuccessful. Exited the pcaTransformation method of the Preprocessor class')
            raise Exception()

    def scale_numerical_columns(self, data):
        """
                    Method Name: scale_numerical_columns
                    Description: This method scales the numerical values using the Standard scaler.
                    Output: A dataframe with scaled values
                    On Failure: Raise Exception

                                     """
        self.logger_object.log(self.file_object,
                               'Entered the scale_numerical_columns method of the Preprocessor class')

        self.data = data

        try:
            for col in self.data.columns:
                self.data[col] = self.data[col].replace(np.NaN, self.data[col].mean())

            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.data)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.data.columns, index=self.data.index)

            self.logger_object.log(self.file_object,
                                   'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()

    def handleImbalance(self, X, Y):

        sample = SMOTE(sampling_strategy=0.3)

        X_bal, y_bal = sample.fit_resample(X, Y)

        return X_bal, y_bal