# from flask import Flask
# from flask import request
# from flask import render_template
from flask import * 
import os
app = Flask(__name__)

if os.getenv('FLASK_CONFIG'):
    conf = os.getenv('FLASK_CONFIG')
    app.config.from_envvar('FLASK_CONFIG')
    print "Running flask with config file {}".format(conf)
elif os.getenv('FLASK_ENV'):
    env = os.getenv('FLASK_ENV')
    app.config.from_object("conf.{}".format(env))
    print "Running flask with environment: {}".format(env)
else:
    app.config.from_object('conf.dev')
    print "Running flask using default conf.dev"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query_url', methods=['POST'])
def query_url():
    return request.form['URL']

@app.route('/query_raw', methods=['POST'])
def query_raw():
    return request.form['raw_text']

