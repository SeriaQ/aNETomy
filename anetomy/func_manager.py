import torch
import types
import __future__


# exempted_funcs = (torch.zeros, torch.ones, torch.eye, torch.linspace, torch.normal, \
#                     torch.rand, torch.randn, torch.randint, torch.randperm, \
#                     torch.Tensor.new_zeros, torch.Tensor.new_ones, torch.Tensor.new_full)
prohibited_funcs = (torch.Tensor.__weakref__, torch.Tensor.__repr__, torch.Tensor.__format__, torch.Tensor.__getitem__, \
                    torch.Tensor.__gt__, torch.Tensor.__lt__, torch.Tensor.__ge__, torch.Tensor.__le__, \
                    torch.Tensor.__eq__, torch.Tensor.__ne__, torch.Tensor.tolist, torch.Tensor.item, \
                    torch.Tensor.unbind, torch.Tensor.has_names, torch.Tensor.clone, torch.Tensor.__setitem__, \
                    torch.Tensor.size, torch.Tensor.shape, torch.Tensor.dim)
exempted_funcs = ()

class FunctionModule():
    def __init__(self, name):
        self.__name__ = name
        self._record = {'attr': None, 'times': 0}


class FuncManager():
    def __init__(self, passed_func=()) -> None:
        self.passed_func = passed_func
        self._builtin_modules = self._get_builtin_modules()

    def _get_builtin_modules(self):
        module_packages = [
            (torch.nn, dir(torch.nn)),
        ]
        mod_info = []
        for pkg, mod_set in module_packages:
            for mod in mod_set:
                if mod.startswith('__'):
                    continue
                elif not mod[0].isupper():
                    continue
                elif mod == 'Module':
                    continue
                else:
                    mod_info.append(getattr(pkg, mod))
        return mod_info

    def _belongs_to_builtin(self, mod):
        if isinstance(mod, FunctionModule):
            return True
        for m in self._builtin_modules:
            if isinstance(mod, m):
                return True
        return False

    def _get_builtin_functions(self):
        func_info = []
        func_packages = [
            ('torch', torch, torch.__all__),# + dir(torch._C._VariableFunctions)),
            ('torch.functional', torch.functional, torch.functional.__all__),
            ('torch.nn.functional', torch.nn.functional, dir(torch.nn.functional)),
            ('torch.Tensor', torch.Tensor, dir(torch.Tensor)),
            ('torch.linalg', torch.linalg, dir(torch.linalg)),
            ('torch.fft', torch.fft, dir(torch.fft)),
        ]
        if hasattr(torch, 'special'):
            func_packages.append(('torch.special', torch.special, dir(torch.special)))
        for pkg_str, pkg, func_set in func_packages:
            pkg_str_wo_top = pkg_str[6:]
            for func_name in func_set:
                # ignore private functions or functions that are deleted in torch.__init__
                if pkg is not torch.Tensor:
                    if func_name.startswith('__'):
                        continue
                    elif func_name[0].isupper():
                        continue
                    elif func_name == 'unique_dim':
                        continue
                    elif 'clone' in func_name:
                        continue
                    elif 'identity' in func_name:
                        continue
                    # remove some nested functions that might be recorded many times
                    if pkg is torch:
                        if func_name.startswith('has_') or func_name.startswith('batch_norm') or \
                            func_name.startswith('instance_norm') or func_name.startswith('layer_norm') or \
                            func_name.startswith('group_norm') or func_name.startswith('max_pool') or \
                            func_name.startswith('adaptive_max_pool') or func_name.startswith('relu') or \
                            func_name.startswith('selu') or func_name.startswith('celu') or \
                            func_name.startswith('softmax') or func_name.startswith('log_softmax') or \
                            func_name.startswith('tanh') or func_name.startswith('sigmoid') or \
                            func_name.startswith('embedding') or func_name.startswith('ctc_loss') or \
                            func_name.startswith('poisson_nll_loss') or func_name.startswith('kl_div') or \
                            func_name.startswith('binary_cross_entropy_with_logits') or func_name.startswith('grid_sampler') or \
                            func_name.startswith('margin_ranking_loss') or func_name.startswith('cosine_embedding_loss') or \
                            func_name.startswith('triplet_margin_loss') or func_name.startswith('affine_grid_generator'):
                            continue
                else:
                    func = getattr(pkg, func_name)
                    if getattr(object, func_name, None) == func:
                        continue

                func = getattr(pkg, func_name)
                if pkg is torch.Tensor and getattr(object, func_name, None) == func:
                    continue
                # ignore re-exported modules
                if isinstance(func, types.ModuleType):
                    continue
                # ignore __future__ imports
                if isinstance(func, getattr(__future__, '_Feature')):
                    continue
                # ignore uncallable functions
                if not callable(func):
                    continue

                if func in prohibited_funcs:
                    continue
                # cannot be overriden by __torch_function__
                if func in torch.overrides.get_ignored_functions():
                    msg = (
                        '{}.{} is in the tuple returned by torch._overrides.get_ignored_functions '
                        'but still has an explicit override'
                    )
                    assert func not in torch.overrides.get_testing_overrides(), msg.format(
                        pkg, func.__name__
                    )
                    # exempt some tensor creator functions
                    if func in exempted_funcs:
                        pass
                    else:
                        continue
                if len(pkg_str_wo_top) == 0:
                    func_info.append(func_name)
                else:
                    func_info.append(f'{pkg_str_wo_top}.{func_name}')
        return func_info

    def _get_pkg_attr(self, pkg, path, sep='.'):
        ptoken = path.split(sep)
        assert len(ptoken) > 0
        atr = pkg
        for pt in ptoken:
            pkg = atr
            atr = getattr(pkg, pt)
        return pkg, pt, atr

    def decorate_funcs(self, func_hook):
        orig_funcs = {}
        for func_name in self._get_builtin_functions():
            direct_pkg, base_fname, func = self._get_pkg_attr(torch, func_name)
            orig_funcs[func_name] = func
            # if func_name.startswith('randn'):
            #     import pdb;pdb.set_trace()
            try:
                setattr(direct_pkg, base_fname, func_hook(func))
            except:
                setattr(direct_pkg, base_fname, func)
        self._orig_funcs = orig_funcs

    def undecorate_funcs(self):
        for k, v in self._orig_funcs.items():
            direct_pkg, base_fname, func = self._get_pkg_attr(torch, k)
            del func
            setattr(direct_pkg, base_fname, v)
