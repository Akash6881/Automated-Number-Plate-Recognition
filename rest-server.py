from wsgiref import simple_server
from flask import Flask, request, Response, render_template
from flask_cors import CORS,cross_origin
from predict_number_plate import DetectNumberPlate


application = Flask(__name__)

CORS(application)


inputFileName = "image1.jpeg"
imagePath = "images/" + inputFileName

class model:
    def __init__(self):
        self.path_to_frozen_graph = 'model_files/frozen_inference_graph.pb'
        self.path_to_label = 'model_files/classes.pbtxt'
        print("Akash Das")
        self.predict_obj = DetectNumberPlate(self.path_to_frozen_graph, self.path_to_label)
        print("Akash Das 22")


@application.route("/", methods = ["GET"])
@cross_origin()
def home():
    return render_template('index.html')


@application.route("/predict", methods = ["POST"])
@cross_origin()
def predict():
    inpImage = request.json["image"]


if __name__ == '__main__':
    name = "a"
    model_obj = model()
    model_obj.predict_obj.predict(imagePath)

    host = "127.0.0.1"
    port = 5000
    httpd = simple_server.make_server(host,port, application)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()