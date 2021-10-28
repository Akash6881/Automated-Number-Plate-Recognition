from wsgiref import simple_server
from flask import Flask, Request, Response, render_template
from flask_cors import CORS,cross_origin
application = Flask(__name__)

CORS(application)

@application.route("/", methods = ["GET"])
@cross_origin()
def home():
    return render_template('index.html')

if __name__ == '__main__':
    host = "127.0.0.1"
    port = 5000
    httpd = simple_server.make_server(host,port, application)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()