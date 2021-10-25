# import torch
# 
# # Util class to simplify delegation, written by Jeremy Howard
# # https://www.fast.ai/2019/08/06/delegation/
# class GetAttr:
#     "Base class for attr accesses in `self._xtra` passed down to `self.default`"
#     @property
#     def _xtra(self): return [o for o in dir(self.default) if not o.startswith('_')]
#     def __getattr__(self,k):
#         if k in self._xtra: return getattr(self.default, k)
#         raise AttributeError(k)
#     def __dir__(self): return dir(type(self)) + list(self.__dict__.keys()) + self._xtra
# 
# class GroupLassoOptimWrapper(GetAttr):
#     def __init__(self, wrapped_optim, group_size=16, alpha=0.001):
#         self.default = wrapped_optim
#         self.group_size = group_size
#         self.alpha = alpha
#         # overrides
#         self._step = self.default.step
#         self.default.step = self.step
# 
#     def step(self, closure):
#         for group in self.param_groups:
#             group_lasso_regularize(group['params'], self.group_size, self.alpha)
#         self.default.step(closure)
# 
# # For wrapping Trainer from Huggingface
# class GroupLassoTrainerWrapper(GetAttr):
#     def __init__(self, wrapped_trainer, group_size=16, alpha=0.001):
#         self.default = wrapped_trainer
#         self.group_size = group_size
#         self.alpha = alpha
#         # overrides
#         self._training_step = self.default.training_step
#         self.default.training_step = self.training_step
# 
#     def training_step(self, model, inputs):
#         loss = self._training_step(model, inputs)
#         group_lasso_regularize(model.parameters(), self.group_size, self.alpha)
#         return loss
# 
# def sparsity(param):
#     count_zero = param.numel() - param.nonzero().size(0)
#     return count_zero / param.numel()
# 
# def print_params_sparsity(params, pruning_threshold=0):
#     for key, param in params:
#         if pruning_threshold != 0:
#             param = torch.clone(param.data)
#             param[torch.abs(param) < pruning_threshold] = 0
#         print("Sparsity for {}{} is {}".format(key, list(param.shape), sparsity(param)))
# 
# def group_lasso_regularize(params, group_size, alpha):
#     for n, p in params:
#         if len(p.shape) > 1 and p.shape[0] % group_size[0] == 0 and p.shape[1] % group_size[1] == 0:
#             grad = p.grad
#             data = p.data
#             new_shape = [p.shape[0] // group_size[0], group_size[0], p.shape[1] // group_size[1], group_size[1]]
#             coeff = data.reshape(new_shape)
#             coeff = alpha / torch.norm(coeff, p = 2, dim = [1, 3])
#             coeff[torch.isinf(coeff)] = 0
#             coeff = coeff.repeat_interleave(group_size[0], dim=0).repeat_interleave(group_size[1], dim=-1)
#             grad.add_(data * coeff)
# 
# def print_params_block_sparsity(params, block_size, writer, global_step, pruning_threshold=0):
#     temp = []
#     sparsity_dict = {}
#     # alpha = min(max((global_step // 8420 - 0), 0) * 0.1, 0.75)
#     alpha = 0.75
#     #for key, param in params:
#     for i in range(len(params)):
#         key = params[i][0]
#         param = params[i][1]
#         if len(param.shape) > 1 and param.shape[0] % block_size[0] == 0 and param.shape[1] % block_size[1] == 0:
#             if pruning_threshold != 0:
#                 param = torch.clone(param.data)
#                 param[torch.abs(param) < pruning_threshold] = 0
#             shape = param.shape
#             new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
#             param = param.reshape(new_shape)
#             normed = torch.norm(param, p = 1, dim = [1, 3])
#             sparsity_per_layer = (normed.numel() - normed.nonzero().size(0)) / normed.numel()
#             sparsity_dict[key] = sparsity_per_layer
#             print("Sparsity with block size {} for {}{} is {}".format(block_size, key, list(normed.shape), sparsity_per_layer))
#             if sparsity_per_layer < alpha:
#                 temp.append(params[i])
#     # writer.add_scalars("sparsity for block size {}".format(block_size), sparsity_dict, global_step)
#     return temp
import torch

# Util class to simplify delegation, written by Jeremy Howard
# https://www.fast.ai/2019/08/06/delegation/
class GetAttr:
    "Base class for attr accesses in `self._xtra` passed down to `self.default`"
    @property
    def _xtra(self): return [o for o in dir(self.default) if not o.startswith('_')]
    def __getattr__(self,k):
        if k in self._xtra: return getattr(self.default, k)
        raise AttributeError(k)
    def __dir__(self): return dir(type(self)) + list(self.__dict__.keys()) + self._xtra

# For wrapping Optimizer from PyTorch
class GroupLassoOptimWrapper(GetAttr):
    def __init__(self, wrapped_optim, group_size=16, alpha=0.001):
        self.default = wrapped_optim
        self.group_size = group_size
        self.alpha = alpha
        # overrides
        self._step = self.default.step
        self.default.step = self.step

    def step(self, closure):
        for group in self.param_groups:
            group_lasso_regularize(group['params'], self.group_size, self.alpha)
        self.default.step(closure)

# For wrapping Trainer from Huggingface
class GroupLassoTrainerWrapper(GetAttr):
    def __init__(self, wrapped_trainer, group_size=16, alpha=0.001):
        self.default = wrapped_trainer
        self.group_size = group_size
        self.alpha = alpha
        # overrides
        self._training_step = self.default.training_step
        self.default.training_step = self.training_step

    def training_step(self, model, inputs):
        loss = self._training_step(model, inputs)
        group_lasso_regularize(model.parameters(), self.group_size, self.alpha)
        return loss

def sparsity(param):
    count_zero = param.numel() - param.nonzero().size(0)
    return count_zero / param.numel()

def print_params_sparsity(params, pruning_threshold=0):
    for key, param in params:
        if pruning_threshold != 0:
            param = torch.clone(param.data)
            param[torch.abs(param) < pruning_threshold] = 0
        print("Sparsity for {}{} is {}".format(key, list(param.shape), sparsity(param)))

def group_lasso_regularize(params, group_size, params_masks, alpha):
    for n, p in params:
        if n not in params_masks.keys() and len(p.shape) > 1 and p.shape[0] % group_size[0] == 0 and p.shape[1] % group_size[1] == 0:
            grad = p.grad
            data = p.data
            new_shape = [p.shape[0] // group_size[0], group_size[0], p.shape[1] // group_size[1], group_size[1]]
            coeff = data.reshape(new_shape)
            coeff = alpha / torch.norm(coeff, p = 2, dim = [1, 3])
            coeff[torch.isinf(coeff)] = 0
            coeff = coeff.repeat_interleave(group_size[0], dim=0).repeat_interleave(group_size[1], dim=-1)
            grad.add_(data * coeff)

def print_params_block_sparsity(params, block_size, global_step, pruning_threshold=0):
    temp = []
    sparsity_dict = {}
    alpha = 0.7
    #alpha = min(max((global_step // 8420 - 0), 0) * 0.1, 0.75)
    #for key, param in params:
    for i in range(len(params)):
        key = params[i][0]
        param = params[i][1]
        if len(param.shape) > 1 and param.shape[0] % block_size[0] == 0 and param.shape[1] % block_size[1] == 0:
            if pruning_threshold != 0:
                param = torch.clone(param.data)
                #param[torch.abs(param) < pruning_threshold] = 0
            shape = param.shape
            new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
            param = param.reshape(new_shape)
            normed = torch.norm(param, p = 1, dim = [1, 3])
            sparsity_per_layer = (normed.numel() - normed.nonzero().size(0)) / normed.numel()
            sparsity_dict[key] = sparsity_per_layer
            print("Sparsity with block size {} for {}{} is {}".format(block_size, key, list(normed.shape), sparsity_per_layer))
            if sparsity_per_layer < alpha:
                temp.append(params[i])
    # if global_step % 500 == 0:
    #     print(sparsity_dict)        
    #writer.add_scalars("sparsity for block size {}".format(block_size), sparsity_dict, global_step)
    return temp
