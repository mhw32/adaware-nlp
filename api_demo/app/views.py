from flask import render_template, request, jsonify
from app import app

import sys
import os
local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)
sys.path.append(local_ref('../../core'))
from bank import ParamBank
from pipeline import AdaSentencePipeline


pbank = ParamBank()
pbank.load()
pipe = AdaSentencePipeline(pbank)


@app.route('/', methods = ['POST', 'GET'])
def index():
    return render_template("index.html")


@app.route('/evaluate_text', methods=['GET'])
def evaluate_text():
    input_str = request.args.get('input_str', None, type=str)
    res_json = pipe.do(input_str) if not input_str is None else 'None'
    return jsonify(result=res_json)
