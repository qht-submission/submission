import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1)
        return weight_q


class activation_quantize_fn(nn.Module):
    def __init__(self, nbits_a):
        super(activation_quantize_fn, self).__init__()
        assert nbits_a <= 8 or nbits_a == 32
        self.nbits_a = nbits_a
        self.uniform_q = uniform_quantize(k=nbits_a)

    def forward(self, x):
        if self.nbits_a == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
            # print(np.unique(activation_q.detach().numpy()))
        return activation_q


class Conv2dDoReFa(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4):
        super(Conv2dDoReFa, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = nbits_w
        self.act_q = activation_quantize_fn(nbits_a=nbits_a)
        self.quantize_fn = weight_quantize_fn(w_bit=nbits_w)

    def forward(self, input):
        input = self.act_q(input)
        weight_q = self.quantize_fn(self.weight)
        # print(np.unique(weight_q.detach().numpy()))
        return F.conv2d(input, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearDoReFa(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4):
        super(LinearDoReFa, self).__init__(in_features, out_features, bias)
        self.w_bit = nbits_w
        self.act_q = activation_quantize_fn(nbits_a=nbits_a)
        self.quantize_fn = weight_quantize_fn(w_bit=nbits_w)

    def forward(self, input):
        input = self.act_q(input)
        weight_q = self.quantize_fn(self.weight)
        # print(np.unique(weight_q.detach().numpy()))
        return F.linear(input, weight_q, self.bias)
