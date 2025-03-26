import os
from graphviz import Digraph


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



class Drawer():
    def __init__(self, graph_path, save_type, in_msg, out_msg):
        self.graph_path = graph_path[:-4]
        self.save_type = save_type
        self.in_msg = in_msg
        self.out_msg = out_msg
        self.graph = Digraph(name='AINN', graph_attr={'fontname': "Helvetica,Arial,sans-serif"})

    def draw(self, entrance, max_depth, to_expand, in_data=None):
        # >| -1:  expand all layers
        # >| N:  collapse layers that are deeper than N or in the input list
        assert isinstance(to_expand, (list, tuple, str)), \
            'NEBULAE ERROR ៙ the modules to be expanded must be passed in as an array or string.'
        self.max_depth = max_depth if max_depth >= 0 else float('inf')
        self.seen = [] # whether has been created
        self.expanded = [] # whether has been expanded
        self.to_expand = list(to_expand) if isinstance(to_expand, (list, tuple)) else [to_expand]
        self.graph_stack = [self.graph]
        self.node_stack = []
        if in_data is None:
            in_data = len(entrance) * [None]
        assert len(in_data) == len(entrance)
        
        # >| starts from entrance nodes
        self.out_idx = 0
        for i, ent in enumerate(entrance):
            # if (os.path.dirname(ent.name) not in self.to_expand) or (ent.name in self.to_expand):
            #     continue
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
            if in_data[i] is not None:
                self.graph.node(ent.name + '_viz', ' ',
                                color=ent.color, penwidth="2", 
                                style="filled", fillcolor="white", 
                                fontname="Courier New", shape="box",
                                image=in_data[i], imagepos='mc', 
                                imagescale='true', width='1.1', height='1.1', fixedsize='true')
                self.graph.edge(ent.name, ent.name + '_viz', constraint='false')
            self.seen.append(ent.name)
            self.node_stack.append([ent.name])
            self.within_subg = 1
            self._expand(ent)
        self.graph.render(self.graph_path, view=False, format=self.save_type)
        os.remove(self.graph_path)

    def _expand(self, root, stack_in=True):
        if root.name in self.expanded:
            return
        if stack_in:
            self.node_stack.append([])
        for c in root.next:
            # print(root.shared_names, '==>', c.shared_names)
            # import pdb;pdb.set_trace()
            if c.type == 'out':
                assert root.type in ('op', 'bim', 'leaf')
                for name in c.shared_names:
                    subg_name = self.graph_stack[-1].name[8:]
                    if name.startswith(subg_name):
                        if len(self.graph_stack) > 1:
                            self.graph_stack[-2].subgraph(self.graph_stack[-1])
                            del self.graph_stack[-1]
                if c.is_leaf():
                    if c.name not in self.seen:
                        if c.depth == 0:
                            message = self.out_msg[min(self.out_idx, len(self.out_msg)-1)]
                            self.out_idx += 1
                        else:
                            message = ''
                        self.graph_stack[-1].node(c.name, f"""<
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
                        self.seen.append(c.name)
                    node_name = self.node_stack[-2] # get last layer
                    while not isinstance(node_name, str):
                        node_name = node_name[-1]
                    subg = self.graph_stack[-self.within_subg]
                    self.within_subg = 1
                    subg.edge(node_name, c.name, ' x '.join([str(s) for s in c.shape]))
                else:
                    self._expand(c, False)
            elif c.type == 'in':
                raise KeyError
            elif (c.type in ('bim', 'leaf') and c.depth <= self.max_depth) or (c.type == 'op' and c.depth == self.max_depth):
                assert root.type in ('in', 'out')
                if c.name not in self.seen:
                    # import pdb; pdb.set_trace()
                    if c.type == 'bim':
                        message = '\n'.join(['<tr><td align="left" port="r5">%s</td></tr>'%a for a in c.attrs])
                    else:
                        message = '<tr><td align="left" port="r5">Attr: Null</td></tr>'
                    self.graph_stack[-1].node(c.name, f"""<
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
                    self.seen.append(c.name)
                node_name = self.node_stack[-2] # get last layer
                while not isinstance(node_name, str):
                    node_name = node_name[-1]
                subg = self.graph_stack[-self.within_subg]
                self.within_subg = 1
                subg.edge(node_name, c.name, ' x '.join([str(s) for s in root.shape]))
                self.node_stack[-1].append(c.name)
                self._expand(c)
            elif (c.type == 'op' and c.depth < self.max_depth) or c.name in self.to_expand:
                assert root.type in ('in', 'out')
                if c.name not in self.seen:
                    subg = Digraph(name='cluster_%s'%c.name, graph_attr={'fontname': "Helvetica,Arial,sans-serif"})
                    subg.attr(style='dashed', color='teal', label=c.label, margin='20 30 20 30') # Top, Right, Bottom, Left
                    self.graph_stack.append(subg)
                    self.within_subg += 1
                self.seen.append(c.name) # only record but not to expand cuz the input will penetrate in
            elif c.type in ('op', 'bim', 'leaf') and c.depth > self.max_depth:
                continue
            else:
                raise TypeError('NEBULAE ERROR ៙ current node %s is rotten or recorded inproperly.'%c.name)
        self.expanded.append(root.name)
        if stack_in:
            del self.node_stack[-1]