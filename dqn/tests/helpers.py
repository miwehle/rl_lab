import hashlib
import io

import torch
from torch import nn


def model_hash(model: nn.Module) -> str:
    buffer = io.BytesIO()
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in sorted(model.state_dict().items())
    }
    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

def ema(values, alpha=0.1):
    result = []
    avg = values[0]

    for value in values:
        avg = alpha * value + (1 - alpha) * avg
        result.append(avg)

    return result