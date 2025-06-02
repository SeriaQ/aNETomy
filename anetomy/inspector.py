import torch
from copy import deepcopy
from functools import wraps as fnwraps
from typing import Iterable, Mapping
from torch import nn
import torch.nn.functional as F
from .func_manager import FuncManager, FunctionModule
from .network_archs import NetNode, NetVue



def ver2num(version, vbits=2):
    version = version.split('.')
    number = 0
    for v in version:
        v = v.split('+')
        if len(v)>1:
            v = int(v[0], 16)
        else:
            v = int(v[0])
        number = number * 10 ** vbits + v
    return number

PT_VER = ver2num(torch.__version__)

class Inspector():
    def __init__(self, in_msg=(), out_msg=(), graph_path='./anetomy.png'):
        if graph_path.endswith('.png'):
            export_format = 'png'
        elif graph_path.endswith('.svg'):
            export_format = 'svg'
        else:
            raise NameError('NEBULAE ERROR ៙ graph path must be a png file or svg file.')
        if isinstance(in_msg, tuple):
            in_msg = in_msg + ('',)
        elif isinstance(in_msg, list):
            in_msg = in_msg + ['']
        if isinstance(out_msg, tuple):
            out_msg = out_msg + ('',)
        elif isinstance(out_msg, list):
            out_msg = out_msg + ['']
        assert len(in_msg) > 0 and len(out_msg) > 0
        self.tensors = []
        self.ops = []
        self.op_sn = {}
        self.fn_mng = FuncManager()
        self.viewer = NetVue(graph_path, export_format, in_msg, out_msg)

        def hook_func_old(func):
            @fnwraps(func)
            def wrapper(*args, **kwargs):
                fm = FunctionModule(func.__name__)
                if isinstance(args[0], (tuple, list)): # binary operators
                    flatten_input = tuple(args[0])+args[1:]+tuple(kwargs.values())
                else:
                    flatten_input = args+tuple(kwargs.values())
                hooked_input = self._before_forward(fm, flatten_input)
                ret = func(*args, **kwargs)
                self._after_forward(fm, hooked_input, ret)
                return ret
            return wrapper

        def hook_func_new(func):
            @fnwraps(func)
            def wrapper(*args, **kwargs):
                fm = FunctionModule(func.__name__)
                hooked_args, hooked_kwargs = self._before_forward(fm, args, kwargs)
                ret = func(*args, **kwargs)
                self._after_forward(fm, hooked_args, hooked_kwargs, ret)
                return ret
            return wrapper
        
        if PT_VER >= 2e4:
            self._hook_func = hook_func_new
            self._before_forward = self._before_forward_new
            self._after_forward = self._after_forward_new
        else:
            self._hook_func = hook_func_old
            self._before_forward = self._before_forward_old
            self._after_forward = self._after_forward_old
    
    def _hook(self, net):
        # avoid adding hooks one more time
        if not hasattr(net, '_record'):
            net._record = {'attr': None, 'times': 0}
            if PT_VER >= 2e4:
                net.register_forward_pre_hook(self._before_forward, with_kwargs=True)
                net.register_forward_hook(self._after_forward, with_kwargs=True)
            else:
                net.register_forward_pre_hook(self._before_forward)
                net.register_forward_hook(self._after_forward)
            
    def _change_format(self, export_format):
        self.viewer.export_format = export_format
    
    def dissect(self, net, *dummy_args, **dummy_kwargs):
        self.main_body = net
        self.fn_mng.decorate_funcs(self._hook_func)
        net.apply(self._hook)
        net(*dummy_args, **dummy_kwargs)
        for t in self.tensors:
            if t.last_op is not None:
                t.last_op.type = 'bim' if t.last_op.type=='bim' else 'leaf'
        self.fn_mng.undecorate_funcs()
        # clear invalid nodes
        for op in self.ops:
            if op.type in ('bim', 'leaf') and op.depth == 0:
                op.type = 'rot'
        self.viewer.entrance = [t for t in self.tensors if t.is_root() and (not t.is_invalid_pendant())]
        self.viewer.dry_draw()

    def render(self, max_depth=0, to_expand=(), to_collapse=()):
        self.viewer.draw(max_depth, to_expand, to_collapse)
        self.viewer.export()

    def run(self, host='127.0.0.1', port=7880):
        self.viewer.run(host, port)

    def _check_in_place(self, out_name, in_names):
        for name in in_names:
            op, stage = name.split('@')
            if stage.startswith('input') and op == out_name.split('@')[0] or op.startswith('__setitem__'):
                return True
        return False
    
    def _arr2str(self, arr):
        s = []
        if isinstance(arr, str):
            s = arr
        elif isinstance(arr, Iterable):
            for a in arr:
                s.append('%d'%a)
            s = ', '.join(s)
        else:
            s = '%d'%arr
        return s
    
    def _record_builtin_attr(self, op):
        if isinstance(op, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.curr_net.attrs = ['In Chs: %s'%self._arr2str(op.in_channels), 'Out Chs: %s'%self._arr2str(op.out_channels), \
                          'Kernel: %s'%self._arr2str(op.kernel_size), 'Stride: %s'%self._arr2str(op.stride), \
                          'Pad: %s'%self._arr2str(op.padding), 'Dilation: %s'%self._arr2str(op.dilation), \
                            'Group: %s'%self._arr2str(op.groups)]
            
        elif isinstance(op, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            self.curr_net.attrs = ['In Chs: %s'%self._arr2str(op.in_channels), 'Out Chs: %s'%self._arr2str(op.out_channels), \
                          'Kernel: %s'%self._arr2str(op.kernel_size), 'Stride: %s'%self._arr2str(op.stride), \
                          'Pad: %s'%self._arr2str(op.padding), 'Out Pad: %s'%self._arr2str(op.output_padding), \
                            'Dilation: %s'%self._arr2str(op.dilation), 'Group: %s'%self._arr2str(op.groups)]
        elif isinstance(op, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, \
                             nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            self.curr_net.attrs = ['Kernel: %s'%self._arr2str(op.kernel_size), \
                                   'Stride: %s'%self._arr2str(op.stride), \
                                    'Pad: %s'%self._arr2str(op.padding)]
        elif isinstance(op, nn.Linear):
            self.curr_net.attrs = ['In Chs: %s'%self._arr2str(op.in_features), \
                                   'Out Chs: %s'%self._arr2str(op.out_features)]
    
    def _return_or_create(self, tensor, name, label):
        # >| create a new node for tensors shown up the first time
        # >| duplicate those penetrating from root node
        # >| push them into an input or output list indexed by their depth
        is_in_place = False
        for t in self.tensors:
            for k, val in t.inputs_by_depth.items():
                for i, v in enumerate(val):
                    if tensor is v:
                        u = v.clone()
                        if self.curr_stage == 'in':
                            self.to_be_cloned.append((t, k+1, u, name))
                            self.to_be_linked.append((t, self.curr_net))
                            self.net_depth = max(self.net_depth, k+1)
                            return u
                        elif self.curr_stage == 'out': # only happens on in-place op
                            assert self._check_in_place(name, t.shared_names)
                            t.inputs_by_depth[k][i] = v.clone()
                            is_in_place = True
                            break
                        else:
                            raise KeyError('NEBULAE ERROR ៙ curr_stage must be either "in" or "out".')
                if is_in_place:
                    break
            for k, val in t.outputs_by_depth.items():
                for i, v in enumerate(val):
                    if tensor is v:
                        if self.curr_stage == 'in':
                            u = v.clone()
                            self.to_be_cloned.append((t, k, u, name))
                            self.to_be_linked.append((t, self.curr_net))
                            self.net_depth = max(self.net_depth, k)
                            return u
                        elif self.curr_stage == 'out':
                            if self._check_in_place(name, t.shared_names):
                                t.outputs_by_depth[k][i] = v.clone()
                                is_in_place = True
                                break
                            u = v.clone()
                            self.to_be_cloned.append((t, k-1, u, name))
                            self.to_be_linked.append((self.curr_net, t))
                            t.depth = min(t.depth, max(0, k-1))
                            t.name = name
                            t.label = label
                            return u
                        else:
                            raise KeyError('NEBULAE ERROR ៙ curr_stage must be either "in" or "out".')
                if is_in_place:
                    break
            if is_in_place:
                break
                        
        if self.curr_stage == 'in':
            self.to_be_cloned.append(NetNode(name, label, '#CC66FF', 0, tensor, self.curr_stage))
            self.to_be_linked.append(self.curr_net) # child
        elif self.curr_stage == 'out':
            self.to_be_cloned.append(NetNode(name, label, '#CC66FF', self.net_depth, tensor, self.curr_stage))
            self.to_be_linked.append(self.curr_net) # parent
        return tensor

    def _collect_tensors(self, tensors, name, label, idx=0):
        bucket = []
        nelem = 0
        if isinstance(tensors, torch.Tensor):
            struct = self._return_or_create(tensors, name + '@%sput_%d'%(self.curr_stage, idx), 
                                                    label + '@%sput_%d'%(self.curr_stage, idx))
            bucket.append(struct)
            nelem += 1
        elif isinstance(tensors, Mapping):
            struct = type(tensors)()
            for k, v in tensors.items():
                b, s, n = self._collect_tensors(v, name, label, idx)
                bucket.extend(b)
                struct[k] = s
                idx += n
                nelem += n
        elif isinstance(tensors, Iterable) and (not isinstance(tensors, (str, torch.Size))):
            struct = []
            for t in tensors:
                b, s, n = self._collect_tensors(t, name, label, idx)
                bucket.extend(b)
                struct.append(s)
                idx += n
                nelem += n
            struct = type(tensors)(struct)
        else:
            struct = tensors
        return bucket, struct, nelem

    def _before_forward_old(self, net, in_args):
        code = net.__name__ if hasattr(net, '__name__') else net.__class__.__name__
        # wrap input as tensors for arithmetic ops
        op_code = code.strip('_')
        if op_code.startswith('add') or op_code.startswith('iadd') or \
            op_code.startswith('sub') or op_code.startswith('isub') or \
            op_code.startswith('mul') or op_code.startswith('imul') or \
            op_code.startswith('div') or op_code.startswith('idiv'):
            list_in_args = []
            for ia in in_args:
                if not isinstance(ia, torch.Tensor):
                    list_in_args.append(torch.tensor(ia))
                else:
                    list_in_args.append(ia)
            in_args = tuple(list_in_args)
        # assign serial number to refrain from giving same names
        if code in self.op_sn.keys():
            self.op_sn[code] += 1
            name = code + '_%d' % self.op_sn[code]
            label = code + '_%d' % self.op_sn[code]
        else:
            self.op_sn[code] = 0
            name = code + '_0'
            label = code + '_0'
        self.net_depth = 0 # reset
        self.curr_stage = 'in'
        self.to_be_cloned = []
        self.to_be_linked = []
        self.curr_net = NetNode(name, label, '#FF3333', self.net_depth)
        # mark if this is a built-in module
        if self.fn_mng._belongs_to_builtin(net):
            self.curr_net.type = 'bim'
            if isinstance(net, FunctionModule):
                if net.__name__ in ('cat', 'stack'):
                    self.curr_net.attrs = ['Axis: %d'%(in_args[2])]
            else:
                self._record_builtin_attr(net)
            if not hasattr(self.curr_net, 'attrs'):
                self.curr_net.attrs = {'Attr: Null'}
        _, _in_args, m = self._collect_tensors(in_args, name, label)
        self.curr_net.depth = self.net_depth # input correct depth
        # hide it if this is a submodule of a built-in module
        is_visible = True
        tensor_buffer = []
        for i in range(len(self.to_be_cloned)):
            cloned = self.to_be_cloned[i]
            linked = self.to_be_linked[i]
            if isinstance(cloned, NetNode): # impossible to be hidden
                continue
            else:
                t, k, _, _ = cloned
                is_visible = t.inner_mark(self.curr_net, k)
                if not is_visible:
                    break
                tensor_buffer.append(t)
        if is_visible:
            for t in tensor_buffer:
                t.last_op = deepcopy(t.last_op_buffer)
                t.nest_sep = deepcopy(t.nest_sep_buffer)
            for i in range(len(self.to_be_cloned)):
                cloned = self.to_be_cloned[i]
                linked = self.to_be_linked[i]
                if isinstance(cloned, NetNode): # impossible to be hidden
                    # cloned.depth = self.net_depth
                    self.tensors.append(cloned)
                    cloned.add_child(linked)
                    cloned.inner_mark(self.curr_net, self.net_depth)
                else:
                    t, k, u, n = cloned
                    if n not in t.shared_names:
                        t.shared_names.append(n)
                    if k not in t.inputs_by_depth.keys():
                        t.inputs_by_depth[k] = [u]
                    else:
                        t.inputs_by_depth[k].append(u)
                    linked[0].add_child(linked[1])
        else:
            del self.curr_net
            return in_args
        if m == 0: # no tensors
            return in_args
        net.__dict__['_net_node'] = self.curr_net
        return _in_args

    def _after_forward_old(self, net, in_args, outs):
        self.curr_stage = 'out'
        self.to_be_cloned = []
        self.to_be_linked = []
        nnode = net.__dict__.get('_net_node', None)
        if nnode is None:
            return outs
        if net is not self.main_body:
            self.ops.append(nnode)
        self.curr_net = nnode
        self.net_depth = nnode.depth
        _, _outs, _ = self._collect_tensors(outs, nnode.name, nnode.label)
        for i in range(len(self.to_be_cloned)):
            cloned = self.to_be_cloned[i]
            linked = self.to_be_linked[i]
            if isinstance(cloned, NetNode):
                self.tensors.append(cloned)
                linked.add_child(cloned)
            else:
                t, k, u, n = cloned
                if n not in t.shared_names:
                    t.shared_names.append(n)
                if k not in t.outputs_by_depth.keys():
                    t.outputs_by_depth[k] = [u]
                else:
                    t.outputs_by_depth[k].append(u)
                linked[0].add_child(linked[1])
        net.__dict__.pop('_net_node')
        return _outs


    def _before_forward_new(self, net, in_args, in_kwargs):
        code = net.__name__ if hasattr(net, '__name__') else net.__class__.__name__
        # wrap input as tensors for arithmetic ops
        op_code = code.strip('_')
        if op_code.startswith('add') or op_code.startswith('iadd') or \
            op_code.startswith('sub') or op_code.startswith('isub') or \
            op_code.startswith('mul') or op_code.startswith('imul') or \
            op_code.startswith('div') or op_code.startswith('idiv'):
            list_in_args = []
            for ia in in_args:
                if not isinstance(ia, torch.Tensor):
                    list_in_args.append(torch.tensor(ia))
                else:
                    list_in_args.append(ia)
            in_args = tuple(list_in_args)
        # assign serial number to refrain from giving same names
        if code in self.op_sn.keys():
            self.op_sn[code] += 1
            name = code + '_%d' % self.op_sn[code]
            label = code + '_%d' % self.op_sn[code]
        else:
            self.op_sn[code] = 0
            name = code + '_0'
            label = code + '_0'
        self.net_depth = 0 # reset
        self.curr_stage = 'in'
        self.to_be_cloned = []
        self.to_be_linked = []
        self.curr_net = NetNode(name, label, '#FF3333', self.net_depth)
        # mark if this is a built-in module
        if self.fn_mng._belongs_to_builtin(net):
            self.curr_net.type = 'bim'
            if isinstance(net, FunctionModule):
                if net.__name__ in ('cat', 'stack'):
                    if len(in_args) > 1:
                        self.curr_net.attrs = ['Axis: %d'%(in_args[1])]
                    else:
                        self.curr_net.attrs = ['Axis: %d'%(in_kwargs['dim'])]
            else:
                self._record_builtin_attr(net)
            if not hasattr(self.curr_net, 'attrs'):
                self.curr_net.attrs = {'Attr: Null'}
        _, _in_args, m = self._collect_tensors(in_args, name, label)
        _, _in_kwargs, l = self._collect_tensors(in_kwargs, name, label, m)
        self.curr_net.depth = self.net_depth # input correct depth
        # hide it if this is a submodule of a built-in module
        # if name.startswith('rand'):
        #     import pdb;pdb.set_trace()
        is_visible = True
        tensor_buffer = []
        for i in range(len(self.to_be_cloned)):
            cloned = self.to_be_cloned[i]
            linked = self.to_be_linked[i]
            if isinstance(cloned, NetNode): # impossible to be hidden
                continue
            else:
                t, k, _, _ = cloned
                is_visible = t.inner_mark(self.curr_net, k)
                if not is_visible:
                    break
                tensor_buffer.append(t)
        if is_visible:
            for t in tensor_buffer:
                t.last_op = deepcopy(t.last_op_buffer)
                t.nest_sep = deepcopy(t.nest_sep_buffer)
            for i in range(len(self.to_be_cloned)):
                cloned = self.to_be_cloned[i]
                linked = self.to_be_linked[i]
                if isinstance(cloned, NetNode): # impossible to be hidden
                    # cloned.depth = self.net_depth
                    self.tensors.append(cloned)
                    cloned.add_child(linked)
                    cloned.inner_mark(self.curr_net, self.net_depth)
                else:
                    t, k, u, n = cloned
                    if n not in t.shared_names:
                        t.shared_names.append(n)
                    if k not in t.inputs_by_depth.keys():
                        t.inputs_by_depth[k] = [u]
                    else:
                        t.inputs_by_depth[k].append(u)
                    linked[0].add_child(linked[1])
        else:
            del self.curr_net
            return in_args, in_kwargs
        if m+l == 0: # no tensors
            return in_args, in_kwargs
        net.__dict__['_net_node'] = self.curr_net
        return _in_args, _in_kwargs

    def _after_forward_new(self, net, in_args, in_kwargs, outs):
        self.curr_stage = 'out'
        self.to_be_cloned = []
        self.to_be_linked = []
        nnode = net.__dict__.get('_net_node', None)
        if nnode is None:
            return outs
        if net is not self.main_body:
            self.ops.append(nnode)
        self.curr_net = nnode
        self.net_depth = nnode.depth
        _, _outs, _ = self._collect_tensors(outs, nnode.name, nnode.label)
        for i in range(len(self.to_be_cloned)):
            cloned = self.to_be_cloned[i]
            linked = self.to_be_linked[i]
            if isinstance(cloned, NetNode):
                self.tensors.append(cloned)
                linked.add_child(cloned)
            else:
                t, k, u, n = cloned
                if n not in t.shared_names:
                    t.shared_names.append(n)
                if k not in t.outputs_by_depth.keys():
                    t.outputs_by_depth[k] = [u]
                else:
                    t.outputs_by_depth[k].append(u)
                linked[0].add_child(linked[1])
        net.__dict__.pop('_net_node')
        return _outs



if __name__ == '__main__':
    class BasicConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, affine=False)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = F.pad(x, [0,1,0,1])
            x = self.conv(x)
            y = self.bn(x)
            y = F.relu(y, inplace=True)
            return y
        
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(28*28*2, 10)

        def forward(self, x):
            x = x.reshape(-1,)
            y = self.lin(x)
            return y

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(6, 8, 3, padding=1)
            self.mpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
            self.conv2 = BasicConv2d(8, 8, kernel_size=3, stride=2, padding=0)
            self.dense = Dense()

        def forward(self, x, scale=2):
            x -= scale
            z = x*scale
            y = torch.cat([x, z], dim=1)
            y = self.conv1(y)
            y = self.mpool1(y)
            y = self.conv2(y)
            y = self.dense(y)
            return y


    # ------------------ main ------------------- #
    net = NeuralNetwork()
    dummy_x = torch.randn(1, 3, 56, 56)
    dummy_s = 2
    ant = Inspector(('Image',), ('Helmet: 82%',))
    ant.dissect(net, dummy_x, scale=dummy_s)
    ant.draw(3)
