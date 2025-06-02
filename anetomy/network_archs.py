import os
from copy import deepcopy
from graphviz import Digraph
from flask import Flask, render_template, request, jsonify


class NetNode():
    def __init__(self, name, label, color, depth, tensor=None, type='op'):
        self.name = name
        self.label = label
        self.color = color
        self.depth = depth
        self.type = type
        if type == 'in':
            self.inputs_by_depth = {depth: [tensor]}
            self.outputs_by_depth = {}
            self.shape = tensor.shape
        elif type == 'out':
            self.inputs_by_depth = {}
            self.outputs_by_depth = {depth: [tensor]}
            self.shape = tensor.shape
        self.shared_names = [name]
        self.last_op = None
        self.nest_sep = []
        # save changes in buffer until inner_mark is done in case of multiple input
        self.last_op_buffer = None
        self.nest_sep_buffer = []
        self.prev = []
        self.next = []

    def add_child(self, child, log=False):
        if log:
            print(self.name, '#%d'%(self.depth), '->', child.name, '#%d'%(child.depth))
        if child not in self.next:
            self.next.append(child)
        # complete upstream info
        if self not in child.prev:
            child.prev.append(self)

    def inner_mark(self, curr_mod, in_depth):
        # |> mark it if the last module is a leaf module
        # |> note that it doesn't mean this is a leaf node
        if self.last_op is None:
            self.last_op_buffer = curr_mod
        else:
            if in_depth <= self.nest_sep[-1]:
                self.last_op.type = 'bim' if self.last_op.type=='bim' else 'leaf'
            elif self.last_op.type == 'bim':
                return False
            self.last_op_buffer = curr_mod
        self.nest_sep_buffer.append(in_depth)
        return True

    def is_root(self):
        return len(self.prev) == 0
    
    def is_leaf(self):
        return len(self.next) == 0
    
    def is_invalid_pendant(self):
        invalid = True
        if self.is_root():
            for n in self.next:
                if n.type != 'rot':
                    invalid = False
                    break
        if self.is_leaf():
            for p in self.prev:
                if p.type != 'rot':
                    invalid = False
                    break
        return invalid





class NetVue():
    def __init__(self, graph_path, export_format, in_msg, out_msg):
        self.graph_path = graph_path[:-4]
        self.export_format = export_format
        self.in_msg = in_msg
        self.out_msg = out_msg
        self.to_expand = set()
        self.to_collapse = set()
        self.app = Flask(__name__, static_folder='static', template_folder='templates')
        self._setup_routes()

    def _setup_routes(self):
        # >| define Flask routes
        @self.app.route('/')
        def display_graph():
            self.draw(0)
            svg_data = self.graph.pipe(format='svg').decode('utf-8')
            return render_template('graph.html', svg_data=svg_data)

        @self.app.route('/refresh', methods=['POST'])
        def refresh():
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
                    self.draw(self.max_depth, to_collapse=element_id)
                else:
                    self.draw(self.max_depth, element_id)
                svg_data = self.graph.pipe(format='svg').decode('utf-8')
                return jsonify({'message': f'Refreshed graph for node {element_id}', 'svg_data': svg_data})
            except Exception as e:
                return jsonify({'error': str(e)}), 400

        @self.app.route('/set_depth', methods=['POST'])
        def set_depth():
            try:
                data = request.get_json()
                max_depth = data.get('max_depth')
                if not isinstance(max_depth, int) or max_depth < -1:
                    return jsonify({'error': 'Invalid max depth'}), 400
                # Update graph with max_depth
                self.to_expand = set()
                self.to_collapse = set()
                self.draw(max_depth)
                svg_data = self.graph.pipe(format='svg').decode('utf-8')
                return jsonify({'message': f'Set max depth to {max_depth}', 'svg_data': svg_data})
            except Exception as e:
                return jsonify({'error': str(e)}), 400

        @self.app.route('/export', methods=['POST'])
        def export():
            try:
                data = request.get_json()
                format = data.get('format')
                if format not in ['png', 'svg']:
                    return jsonify({'error': 'Invalid format'}), 400
                # Render graph in requested format
                self.export()
                svg_data = self.graph.pipe(format='svg').decode('utf-8')
                return jsonify({'message': f'Export to {self.graph_path + '.' + self.export_format}', 'svg_data': svg_data})
            except Exception as e:
                return jsonify({'error': str(e)}), 400

        @self.app.route('/search', methods=['POST'])
        def search():
            data = request.get_json()
            query = data.get('query', '').lower()
            results = []
            seen_ids = set()  # Track unique IDs to avoid duplicates

            # Get node names
            # Adjust based on your graph structure
            for node_id in self.rendered_nodes:  # Use graph.nodes for explicit node list
                if query in node_id.lower() and node_id not in seen_ids:
                    results.append({
                        'type': 'node',
                        'id': node_id,
                        'display_name': node_id
                    })
                    seen_ids.add(node_id)

            # Get subgraph names
            for subgraph_id in self.graph_stack.keys():
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

        @self.app.route('/center', methods=['POST'])
        def center():
            data = request.get_json()
            type_selected = data.get('type') # 'node' or 'subgraph'
            id_selected = data.get('id') # e.g., 'Node_1' or 'cluster_Conv_0'
            
            # Verify the node or subgraph exists (using your manual records)
            # Example: Check against your node/subgraph lists
            if type_selected == 'node' and id_selected not in self.rendered_nodes:  # Replace with your node list
                return jsonify({'error': f'Node {id_selected} not found'}), 404
            if type_selected == 'subgraph' and id_selected not in self.graph_stack.keys():  # Replace with your subgraph list
                return jsonify({'error': f'Subgraph {id_selected} not found'}), 404

            try:
                # Generate current SVG without modifying graph state
                svg_data = self.graph.pipe(format='svg').decode('utf-8')
                if not svg_data:
                    return jsonify({'error': 'Failed to generate SVG'}), 500
                return jsonify({'message': f'Centered on {type_selected} {id_selected}', 'svg_data': svg_data})
            except Exception as e:
                return jsonify({'error': f'SVG generation failed: {str(e)}'}), 500

    def run(self, host='127.0.0.1', port=7880):
        self.app.run(debug=False, port=port, host=host)
    
    def draw(self, max_depth=0, to_expand=(), to_collapse=(), in_data=None):
        # >| -1:  expand all layers
        # >| N:  collapse layers that are deeper than N or in the input list
        assert hasattr(self, 'entrance'), \
            'ANETOMY ERROR ៙ missing inputs for drawing, please dissect network beforehand.'
        assert isinstance(to_expand, (list, tuple, str)), \
            'ANETOMY ERROR ៙ the modules to be expanded must be passed in as an array or string.'

        self.graph = Digraph(name='aNETomy', graph_attr={'fontname': "Helvetica,Arial,sans-serif"})
        self.max_depth = max_depth if max_depth >= 0 else float('inf')
        self.seen = [] # whether has been created
        self.expanded = [] # whether has been expanded
        to_expand = list(to_expand) if isinstance(to_expand, (list, tuple)) else [to_expand]
        to_expand = set(to_expand)
        to_collapse = list(to_collapse) if isinstance(to_collapse, (list, tuple)) else [to_collapse]
        to_collapse = set(to_collapse)
        # expand it if it is collapsed, vice versa
        for te in to_expand:
            self.to_expand.add(te)
            try:
                self.to_collapse.remove(te)
            except:
                pass
        for tc in to_collapse:
            self.to_collapse.add(tc)
            try:
                self.to_expand.remove(tc)
            except:
                pass
        # print(self.to_expand, self.to_collapse)
        assert len(self.to_expand & self.to_collapse) == 0
        
        self.graph_stack = {'': self.graph}
        self.rendered_nodes = []
        self.node_stack = []
        if in_data is None:
            in_data = len(self.entrance) * [None]
        assert len(in_data) == len(self.entrance)
        
        # >| starts from entrance nodes
        self.out_idx = 0
        for i, ent in enumerate(self.entrance):
            if self._is_invisible(ent):
                continue
            message = self.in_msg[min(i, len(self.in_msg)-1)]
            self.graph.node(ent.name, f"""<
                            <table border="0" cellborder="0" cellpadding="3" bgcolor="white">
                                <tr>
                                    <td bgcolor="black" align="center" colspan="2"><font color="white">{ent.label} D:0</font></td>
                                </tr>
                                <tr>
                                    <td align="left" port="r5">{message}</td>
                                </tr>
                            </table>
                            >""",
                            color=ent.color, penwidth="2", 
                            style="filled", fillcolor="white", 
                            fontname="Courier New", shape="box")
            self.rendered_nodes.append(ent.name)
            if in_data[i] is not None:
                self.graph.node(ent.name + '_viz', ' ',
                                color=ent.color, penwidth="2", 
                                style="filled", fillcolor="white", 
                                fontname="Courier New", shape="box",
                                image=in_data[i], imagepos='mc', 
                                imagescale='true', width='1.1', height='1.1', fixedsize='true')
                self.graph.edge(ent.name, ent.name + '_viz', constraint='false')
            self.seen.append(ent.name)
            self.node_stack.append([ent])
            self._expand(ent)
        # link all subgraphs to their upstreams
        for a, b in self.scope_links:
            if a in self.graph_stack.keys() and b in self.graph_stack.keys():
                self.graph_stack[a].subgraph(self.graph_stack[b])

    def export(self):
        self.graph.render(self.graph_path, view=False, format=self.export_format)
        os.remove(self.graph_path)

    def _is_invisible(self, root):
        for c in root.next:
            if c.depth <= self.max_depth or c.name in self.to_expand:
                return False
        return True

    def _to_be_collapsed(self, root):
        if root.type == 'op':
            scope = root.scope + [root.name]
        else:
            scope = root.scope
        for s in scope:
            if s in self.to_collapse:
                return True
        return False

    def _deepest_common_acient(self, a, b):
        s = ''
        for i in range(min(len(a.scope), len(b.scope))):
            if a.scope[i] == b.scope[i]:
                s = a.scope[i]
            else:
                return s
        return s
    
    def _expand(self, root, stack_in=True):
        if root.name in self.expanded:
            return
        if stack_in:
            self.node_stack.append([])
        for c in root.next:
            # print(root.shared_names, '==>', c.shared_names, '||', c.scope)
            # deal with child according to its type    
            if c.type == 'out':
                if c.is_leaf():
                    if c.name not in self.seen:
                        if c.depth == 0:
                            message = self.out_msg[min(self.out_idx, len(self.out_msg)-1)]
                            self.out_idx += 1
                        else:
                            message = ''
                        self.graph_stack[c.scope[-1]].node(c.name, f"""<
                                <table border="0" cellborder="0" cellpadding="3" bgcolor="white">
                                    <tr>
                                        <td bgcolor="black" align="center" colspan="2"><font color="white">{c.label} D:{c.depth}</font></td>
                                    </tr>
                                    <tr>
                                        <td align="left" port="r5">{message}</td>
                                    </tr>
                                </table>
                                >""",
                                color=c.color, penwidth="2", 
                                style="filled", fillcolor="white", 
                                fontname="Courier New", shape="Mrecord")
                        self.rendered_nodes.append(c.name)
                        self.seen.append(c.name)
                    nearest_node = self.node_stack[-2] # get last layer
                    while not isinstance(nearest_node, NetNode):
                        nearest_node = nearest_node[-1]
                    subg = self.graph_stack[self._deepest_common_acient(nearest_node, c)]
                    subg.edge(nearest_node.name, c.name, ' x '.join([str(s) for s in c.shape]))
                else:
                    self._expand(c, False)
            elif c.type == 'in':
                raise KeyError
            elif c.type == 'op' and (c.depth < self.max_depth or c.name in self.to_expand) and not self._to_be_collapsed(c):
                if c.name not in self.seen:
                    subg = Digraph(name='cluster_%s'%c.name, graph_attr={'fontname': "Helvetica,Arial,sans-serif"})
                    subg.attr(style='dashed', color='teal', label=c.label, margin='20 30 20 30') # Top, Right, Bottom, Left
                    self.graph_stack[c.name] = subg
                    self.seen.append(c.name) # only record but not to expand cuz the input will penetrate in
            elif (c.type in ('bim', 'leaf') and (c.depth <= self.max_depth) and not self._to_be_collapsed(c)) or \
                (c.type == 'op' and c.depth == self.max_depth and not c.name in self.to_expand) or \
                (c.type == 'op' and c.name in self.to_collapse) or \
                (c.scope[-1] in self.to_expand and not self._to_be_collapsed(c)):
                if c.name not in self.seen:
                    if c.type == 'bim':
                        message = '\n'.join(['<tr><td align="left" port="r5">%s</td></tr>'%a for a in c.attrs])
                    else:
                        message = '<tr><td align="left" port="r5">Attr: Null</td></tr>'
                    self.graph_stack[c.scope[-1]].node(c.name, f"""<
                                <table border="0" cellborder="0" cellpadding="3" bgcolor="white">
                                    <tr>
                                        <td bgcolor="black" align="center" colspan="2"><font color="white">{c.label} D:{c.depth}</font></td>
                                    </tr>
                                    {message}
                                </table>
                                >""",
                                color=c.color, penwidth="2", 
                                style="filled", fillcolor="white", 
                                fontname="Courier New", shape="Mrecord")
                    self.rendered_nodes.append(c.name)
                    self.seen.append(c.name)
                nearest_node = self.node_stack[-2] # get last layer
                while not isinstance(nearest_node, NetNode):
                    nearest_node = nearest_node[-1]
                subg = self.graph_stack[self._deepest_common_acient(nearest_node, c)]
                subg.edge(nearest_node.name, c.name, ' x '.join([str(s) for s in root.shape]))
                self.node_stack[-1].append(c)
                self._expand(c)
            elif c.type in ('op', 'bim', 'leaf') and (c.depth > self.max_depth or self._to_be_collapsed(c)):
                continue
            else:
                raise TypeError('NEBULAE ERROR ៙ current node %s is rotten or recorded inproperly.'%c.name)
        self.expanded.append(root.name)
        if stack_in:
            del self.node_stack[-1]

    def dry_draw(self):
        # >| -1:  expand all layers
        # >| N:  collapse layers that are deeper than N or in the input list
        self.max_depth = float('inf')
        self.seen = [] # whether has been created
        self.expanded = [] # whether has been expanded
        self.scope_stack = ['']
        self.scope_links = []
        
        # >| starts from entrance nodes
        for i, ent in enumerate(self.entrance):
            if self._is_invisible(ent):
                continue
            self.seen.append(ent.name)
            ent.scope = ['']
            self._dry_expand(ent)

    def _dry_expand(self, root):
        if root.name in self.expanded:
            return
        restored_stack = deepcopy(self.scope_stack)
        for c in root.next:
            if root.depth == c.depth:
                self.scope_stack = deepcopy(restored_stack)
            # print(root.depth, c.depth, root.shared_names, '->', c.shared_names, '||', self.scope_stack)
            if c.type == 'out':
                assert root.type in ('op', 'bim', 'leaf')
                for name in c.shared_names:
                    subs_name = self.scope_stack[-1]
                    if name.startswith(subs_name):
                        if len(self.scope_stack) > 1:
                            self.scope_links.append((self.scope_stack[-2], self.scope_stack[-1]))
                            del self.scope_stack[-1]
                c.scope = deepcopy(self.scope_stack)
                if c.is_leaf():
                    if c.name not in self.seen:
                        self.seen.append(c.name)
                else:
                    self._dry_expand(c)
            elif c.type == 'in':
                raise KeyError
            elif c.type == 'op':
                assert root.type in ('in', 'out')
                if c.name not in self.seen:
                    c.scope = deepcopy(self.scope_stack)
                    self.scope_stack.append(c.name)
                    self.seen.append(c.name) # only record but not to expand cuz the input will penetrate in
            elif c.type in ('bim', 'leaf'):
                assert root.type in ('in', 'out')
                if c.name not in self.seen:
                    c.scope = deepcopy(self.scope_stack)
                    self.seen.append(c.name)
                self._dry_expand(c)
            else:
                raise TypeError('NEBULAE ERROR ៙ current node %s is rotten or recorded inproperly.'%c.name)
        self.expanded.append(root.name)