from flask import render_template, request, jsonify
from app import app

import sys
import os
import json
from datetime import datetime
local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)
sys.path.append(local_ref('../../core'))
from bank import ParamBank
from pipeline import AdaTextPipeline


pbank = ParamBank()
pbank.load()
pipe = AdaTextPipeline(pbank)


@app.route('/', methods = ['POST', 'GET'])
def index():
    return render_template("index.html")


@app.route('/evaluate_text', methods=['GET'])
def evaluate_text():
    input_str = request.args.get('input_str', None, type=str)
    print('[{}] Retrieved input: {}'.format(str(datetime.now()), input_str))
    print('[{}] Executing Ada Pipeline'.format(str(datetime.now())))
    if not input_str is None:
        res_json = pipe.do(input_str)
        print('[{}] Returning JSON: {}'.format(str(datetime.now()), res_json))
        return jsonify(result=json.dumps(res_json))
    return jsonify(result='None')
