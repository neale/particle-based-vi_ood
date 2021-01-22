import torch
import torch.nn as nn
import numpy as np


# Exponential Moving Averager adapted from the Agent Learning framework
# https://github.com/HorizonRobotics/alf/blob/pytorch/alf/utils/averager.py
class EMA(torch.nn.Module):
    def __init__(self, shape, update_rate):
        super().__init__()
        self.shape = shape
        self.update_rate = update_rate
        var_id = [0]

        def create_variable(shape):
            var = torch.zeros(())
            self.register_buffer("_var%s" % var_id[0], var)
            var_id[0] += 1
            return var
        
        self.average = create_variable(shape).cuda()
        self.register_buffer("mass", torch.zeros((), dtype=torch.float64))#.cuda())

    def update(self, tensor):
        func = lambda average, t: average.add_(
            torch.as_tensor(self.update_rate, dtype=t.dtype) * (
                t.mean(0) - average))
        average = func(self.average, tensor.detach())
        self.mass.add_(torch.as_tensor(self.update_rate, dtype=torch.float64) * 
            (1 - self.mass))

    def get(self):
        func = lambda average: average / self.mass.clamp(min=self.update_rate).to(average.dtype)
        return func(self.average)

    def average(self, tensor):
        self.update(tensor)
        return self.get()
                

class Averager(EMA):
    def __init__(self, shape, speed=10):
        update_rate = torch.ones((), dtype=torch.float64)
        super().__init__(shape, update_rate)
        self.register_buffer('update_ema_rate', update_rate)
        self.register_buffer('total_steps', torch.as_tensor(speed).long())
        self.speed = speed

    def update(self, tensor):
        self.update_ema_rate.fill_(self.speed / self.total_steps.to(torch.float64))
        self.total_steps.add_(1)
        super().update(tensor)


def rbf_fn(x, y, width_averager=None):
    Nx = x.shape[0]
    Ny = y.shape[0]
    x = x.view(Nx, -1)
    y = y.view(Ny, -1)
    Dx = x.shape[1]
    Dy = y.shape[1]
    assert Dx == Dy
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, D]
    dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
    if width_averager is not None:
        h = width_averager(dist_sq.view(-1))
    else:
        if dist.ndim > 1:
            dist = torch.sum(dist, dim=-1)
            assert dist.ndim == 1, "dist must have dimension 1 or 2."
        width, _ = torch.median(dist, dim=0)
        h = width / np.log(len(dist))

    kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
    kappa_grad = torch.einsum('ij,ijk->ijk', kappa,
                              -2 * diff / h)  # [Nx, Ny, D]
    return kappa, kappa_grad

def rbf(X, sigma=None):
    GX = torch.dot(X, X.T)
    print (X.shape)
    print (GX.shape)
    print (GX) 
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def centering(K):
    """HKH are the same with KH,
       KH is the first centering,
       H(KH) do the second time,
       results are the sme with one time centering
    """
    n = K.shape[0]
    unit = torch.ones(n, n).cuda()
    I = torch.eye(n).cuda()
    H = I - unit / n

    return torch.dot(torch.dot(H, K), H)  
    # HK = torch.matmul(H, K)
    # HKH = torch.matmul(HK, H)
    # return HKH


def kernel_hsic(x, y, width_averager):
    x_ = x.detach().requires_grad_(True)
    y_ = y.detach().requires_grad_(True)
    center_x = centering(rbf(x)) # [N, N]
    center_y = centering(rbf(y)) # [N, N]
    return torch.sum(center_x * center_y)


def kernel_cka(x, y, width_averager=None):
    hsic = kernel_hsic(x, y, width_averager) # [N, N]
    var1 = kernel_hsic(x, x, width_averager).pow(.5) # [N, N]
    var2 = kernel_hsic(y, y, width_averager).pow(.5) # [N, N]
    return hsic / var1 * var2, None


def extract_parameters(models):
    params = []
    state_dict = {}
    for model in models:
        model_param = torch.tensor([]).cuda()
        for name, param in model.named_parameters():
            if param.requires_grad:# and 'bn' not in name:
                p = param.view(-1).clone().detach()
                start_idx = len(model_param)
                model_param = torch.cat((model_param, p), -1)
                end_idx = len(model_param)
                state_dict[name] = (param.shape, start_idx, end_idx)
        state_dict['param_len'] = len(model_param)
        params.append(model_param)
    params = torch.stack(params)
    return params, state_dict


def insert_items(models, item_list, state_dict, item_name='params'):
    for i, model in enumerate(models):
        for name, param in model.named_parameters():
            if param.requires_grad:# and 'bn' not in name:
                shape, start, end = state_dict[name]
                params_to_model = item_list[i, start:end].view(*shape)
                if item_name == 'params':
                    param.data = params_to_model
                elif item_name == 'grads':
                    param.grad = params_to_model
                else:
                    raise NotImplementedError


