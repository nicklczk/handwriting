# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:01:50 2018

@author: Sean Rice
"""

from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
#from flask.ext.jsonpify import jsonify

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'Greetings': ["Hello world", "Hola mundo"]} # Fetches first column that is Employee ID
        
api.add_resource(HelloWorld, '/hello') # Route_1

if __name__ == '__main__':
     app.run(port='5002')