#!/usr/bin/env python

# from flask import Flask
# from flask import request
# from flask import render_template
import os
import sys
project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(project_path)

from flask import * 
app = Flask(__name__)
if os.getenv('FLASK_ENV'):
    env = os.getenv('FLASK_ENV')
    app.config.from_object("conf.{}".format(env))
    print "Running flask with environment: {}".format(env)
else:
    app.config.from_object('conf.dev')
    print "Running flask using default conf.dev"

from graph import *

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query_url', methods=['POST'])
def query_url():
    return request.form['URL']

@app.route('/query_raw', methods=['POST'])
def query_raw():
    return request.form['raw_text']
