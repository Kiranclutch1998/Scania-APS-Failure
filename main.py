from wsgiref import simple_server
import pandas as pd
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from trainingModel import trainModel
from predictFromModel import prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.form.to_dict() is not None:
            path = request.form.to_dict()
            aps_data = pd.read_csv(path["inFile"])

            predModelObj = prediction(path)  # object initialization
            predModelObj.predictionFromModel()

            return 'Process Complete. Please check Output Directory.' + path['outFile']

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        path = 'EDA + training/aps_failure_training_set.csv'

        trainModelObj = trainModel() #object initialization
        trainModelObj.trainingModel() #training the model for the files in the table

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")


port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 5000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
