import pandas as pd
import numpy as np
from file_operations import file_methods
from data_preprocessing import preprocessing
from application_logging import logger


class prediction:

    def __init__(self, path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.path = path

    def predictionFromModel(self):

        try:
            self.log_writer.log(self.file_object, 'Start of Prediction')
            data = pd.read_csv(self.path["inFile"])

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            # replacing 'na' values with np.nan as discussed in the EDA part

            data = data.replace('na', np.NaN)

            # object datatype to float datatype
            X = data
            X = X.apply(pd.to_numeric)

            # removing unused columns
            col = ['cd_000', 'ab_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000', 'br_000', 'cr_000']
            X = preprocessor.remove_columns(data, col)

            # replacing null values with mean
            X = preprocessor.handleMissingValues(X)

            X = preprocessor.scale_numerical_columns(X)

            X = preprocessor.pcaTransformation(X)

            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
            model = file_loader.load_model('SVC')

            result = []  # initialize blank list for storing predictions

            for val in (model.predict(X)):
                result.append(val)
            result = pd.DataFrame(result, columns=['Predictions'])
            result['Predictions'] = result['Predictions'].map({ 0: 'neg', 1: 'pos'})
            path = self.path["outFile"] + '/' + "Predictions.csv"
            result.to_csv(path, header=True)  # appends result to prediction file
            self.log_writer.log(self.file_object, 'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occurred while running the prediction!! Error:: %s' % ex)
            raise ex
        return path





