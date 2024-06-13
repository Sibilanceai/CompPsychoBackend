# app.py
import os
import sys
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import networkx as nx

import logging

logging.basicConfig(level=logging.DEBUG)

# Adjust the path to include the directory where TE_graph.py is located
sys.path.insert(0, '/Users/ajithsenthil/Desktop/SibilanceAIWebsite/CompPsychoBackend/src/data_analysis')

# Change the current working directory
os.chdir('/Users/ajithsenthil/Desktop/SibilanceAIWebsite/CompPsychoBackend/src/data_analysis')

from TE_graph_app import generate_all_graphs  # Import the function to generate graphs

app = Flask(__name__, static_folder='static', template_folder='templates')
# after creating the Flask app instance
CORS(app)
all_graphs = generate_all_graphs()  # Call this function to get the graphs

@app.route('/')
def home():
    logging.info(f"Rendering home page")
    return render_template('index.html')

@app.route('/get_graph/<int:timestep>')
def get_graph(timestep):
    logging.info(f"Request for graph at timestep: {timestep}")
    if 0 <= timestep < len(all_graphs):
        graph = all_graphs[timestep]
        try:
            data = nx.readwrite.json_graph.cytoscape_data(graph)
            logging.info(f"Serving graph with {len(data['elements']['nodes'])} nodes and {len(data['elements']['edges'])} edges at timestep {timestep}")
            # Log details about each edge's weight for debugging
            for edge in data['elements']['edges']:
                weight = edge['data']['weight']
                logging.debug(f"Edge from {edge['data']['source']} to {edge['data']['target']} with weight: {weight}")
            return jsonify(data)
        except Exception as e:
            logging.error(f"Error processing graph data: {e}")
            return jsonify({"error": "Error processing graph data"}), 500
    else:
        logging.error(f"Timestep {timestep} out of range.")
        return jsonify({"error": "Timestep out of range"}), 404

@app.route('/get_total_timesteps')
def get_total_timesteps():
    total = len(all_graphs)
    logging.info(f"Total timesteps available: {total}")
    return jsonify({"totalTimesteps": total})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

