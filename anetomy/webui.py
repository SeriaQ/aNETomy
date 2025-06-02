from flask import Flask, render_template, request, jsonify
from .inspector import Inspector
# from graphviz import Digraph

import torch
from torch import nn
import torch.nn.functional as F
from nebulae import *

app = Flask(__name__)

# define the toy model
class Conv1x1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        y = self.conv(x)
        return y

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv1 = Conv1x1(out_channels)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, affine=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.pad(x, [1,1,1,1])
        x = self.conv(x)
        x = self.conv1(x)
        y = self.bn(x)
        y = F.relu(y, inplace=True)
        return y

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1568, 10)

    def forward(self, x):
        x = F.relu(x, inplace=True)
        x = x.reshape(-1,)
        y = self.lin(x)
        return y
          
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = BasicConv2d(3, 8, kernel_size=3)
        self.mpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.dense = Dense()

    def forward(self, x, scale=1):
        x *= scale
        z = self.conv2(x)
        y = self.conv1(x)
        y = self.mpool1(y+z)
        y = self.dense(y)
        return y

# input dummy data to visualize
# net = NeuralNetwork()
# dummy_x = torch.randn(1, 3, 28, 28)
# dummy_s = 2
net = nah.Resnet_V2_50((224, 224, 3))#resnet50()
dummy_x = torch.randn(1, 3, 224, 224)
ant = Inspector(graph_path='./toynn.png')
ant.dissect(net, dummy_x,)#scale=dummy_s)
max_depth = 0  # Global max_depth, default matches sidebar



@app.route('/')
def display_graph():
    global ant, max_depth
    ant.draw(max_depth)
    svg_data = ant.drawer.graph.pipe(format='svg').decode('utf-8')
    return render_template('graph.html', svg_data=svg_data)

@app.route('/refresh', methods=['POST'])
def refresh():
    global ant, max_depth
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        element_type = data.get('type')  # 'node' or 'subgraph'
        element_id = data.get('id')      # ID of clicked element
        if not element_type or not element_id:
            return jsonify({'error': 'Missing type or id in request'}), 400
        # refresh logic goes here
        if element_type == 'subgraph':
            element_id = element_id[8:]
            ant.draw(max_depth, to_collapse=element_id)
        else:
            ant.draw(max_depth, element_id)
        svg_data = ant.drawer.graph.pipe(format='svg').decode('utf-8')
        return jsonify({'message': f'Refreshed graph for node {element_id}', 'svg_data': svg_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/set_depth', methods=['POST'])
def set_depth():
    global ant, max_depth
    try:
        data = request.get_json()
        max_depth = data.get('max_depth')
        if not isinstance(max_depth, int) or max_depth < -1:
            return jsonify({'error': 'Invalid max depth'}), 400
        # Update graph with max_depth
        ant.drawer.to_expand = set()
        ant.drawer.to_collapse = set()
        ant.draw(max_depth)
        svg_data = ant.drawer.graph.pipe(format='svg').decode('utf-8')
        return jsonify({'message': f'Set max depth to {max_depth}', 'svg_data': svg_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/export', methods=['POST'])
def export():
    global ant
    try:
        data = request.get_json()
        format = data.get('format')
        if format not in ['png', 'svg']:
            return jsonify({'error': 'Invalid format'}), 400
        # Render graph in requested format
        ant.drawer.export()
        svg_data = ant.drawer.graph.pipe(format='svg').decode('utf-8')
        return jsonify({'message': f'Export to {ant.drawer.graph_path + '.' + ant.drawer.export_format}', 'svg_data': svg_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/search', methods=['POST'])
def search():
    global ant
    data = request.get_json()
    query = data.get('query', '').lower()
    results = []
    seen_ids = set()  # Track unique IDs to avoid duplicates

    # Get node names
    # Adjust based on your graph structure
    for node_id in ant.drawer.rendered_nodes:  # Use graph.nodes for explicit node list
        if query in node_id.lower() and node_id not in seen_ids:
            results.append({
                'type': 'node',
                'id': node_id,
                'display_name': node_id
            })
            seen_ids.add(node_id)

    # Get subgraph names
    for subgraph_id in ant.drawer.graph_stack.keys():
        if subgraph_id: # omit empty string
            display_name = '<G> ' + subgraph_id # G stands for graph
            if query in display_name.lower() and subgraph_id not in seen_ids:
                results.append({
                    'type': 'subgraph',
                    'id': subgraph_id,
                    'display_name': display_name
                })
                seen_ids.add(subgraph_id)

    return jsonify({'results': results})

@app.route('/center', methods=['POST'])
def center():
    global ant
    data = request.get_json()
    type_selected = data.get('type')  # 'node' or 'subgraph'
    id_selected = data.get('id')  # e.g., 'Node_1' or 'cluster_Conv_0'
    
    # Verify the node or subgraph exists (using your manual records)
    # Example: Check against your node/subgraph lists
    if type_selected == 'node' and id_selected not in ant.drawer.rendered_nodes:  # Replace with your node list
        return jsonify({'error': f'Node {id_selected} not found'}), 404
    if type_selected == 'subgraph' and id_selected not in ant.drawer.graph_stack.keys():  # Replace with your subgraph list
        return jsonify({'error': f'Subgraph {id_selected} not found'}), 404

    try:
        # Generate current SVG without modifying graph state
        svg_data = ant.drawer.graph.pipe(format='svg').decode('utf-8')
        if not svg_data:
            return jsonify({'error': 'Failed to generate SVG'}), 500
        return jsonify({'message': f'Centered on {type_selected} {id_selected}', 'svg_data': svg_data})
    except Exception as e:
        return jsonify({'error': f'SVG generation failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=7860, host='0.0.0.0')