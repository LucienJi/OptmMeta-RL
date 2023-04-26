import numpy as np
import torch
import torch.nn as nn


def unflatten_grad(grads, shapes):
    unflatten_grad, idx = [], 0
    for shape in shapes:
        length = np.prod(shape)
        unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
        idx += length
    return unflatten_grad

def flatten_grad(grads):
    flattened = torch.cat([g.flatten() for g in grads])
    return flattened

def pack_grad(loss,model):
    gradients,shapes,has_grad = [],[],[]
    grads = torch.autograd.grad(loss,model.parameters())
    for grad,para in zip(grads,model.parameters()):
        shapes.append(grad.shape)
        gradients.append(grad.clone())
        has_grad.append(torch.ones_like(para).to(para.device))
    
    flattened_gradients = flatten_grad(gradients)
    flattened_has_grad = flatten_grad(has_grad)
    return flattened_gradients,shapes,flattened_has_grad