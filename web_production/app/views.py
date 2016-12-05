from flask import render_template, request, jsonify
from app import app

import sys
import os
import json
from datetime import datetime
local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)
sys.path.append(local_ref('../../core'))
sys.path.append(local_ref('../../graph_tools'))
from bank import ParamBank
from pipeline import AdaTextPipeline
from nlp_to_graph import SemanticGraph
from graph_viz import visualize_graph


# pbank = ParamBank()
# pbank.load()
# pipe = AdaTextPipeline(pbank)
# sgraph = SemanticGraph()

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/core', methods = ['POST', 'GET'])
def core():
    return render_template('core.html')

@app.route('/graph', methods = ['POST', 'GET'])
def graph():
    return render_template('graph.html')

@app.route('/evalute_graph', methods=['GET'])
def evaluate_graph():
    input_str = request.args.get('input_str', None, type=str)
    print('[{}] Retrieved input: {}'.format(str(datetime.now()), input_str))
    print('[{}] Executing Ada Pipeline'.format(str(datetime.now())))
    if not input_str is None:
        res_json = pipe.do(input_str)
        print('[{}] Returning JSON: {}'.format(str(datetime.now()), res_json))
        res_json = jsonify(result=json.dumps(res_json))

        print('[{}] Creating AdaGraph.'.format(str(datetime.now())))
        sgraph.doc_json_to_graph(res_json)

        # choose 1% of nodes to activate by randomly
        num_nodes = sgraph.graph.num_nodes
        num_sample = num_nodes * 0.01
        if num_sample == 0:
            num_sample += 1
        source_list = np.random.choice(range(num_nodes), num_sample)

        print('[{}] Spreading AdaGraph.'.format(str(datetime.now())))
        sgraph.graph.spreading_activation(source_list)

        # visualize the graph (return path to saved file)
        print('[{}] Generating AdaGraph PNG.'.format(str(datetime.now())))
        out_png_file = 'tmp/adagraph_output.png'
        graph_viz(sgraph.graph, out_png_file)
        return jsonify(result=out_png_file)

    return jsonify(result='None')

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
