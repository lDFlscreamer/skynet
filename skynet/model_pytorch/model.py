import torch
from torch import nn

from skynet.cuda import use_cuda
from skynet.model_pytorch.model_part import Input_part, Inner_part, Out_part


class BaseModel(nn.Module):
    def load(self, path):
        if use_cuda:
            params = torch.load(path)
        else:
            params = torch.load(path, map_location=lambda storage, loc: storage)

        state = self.state_dict()
        for name, val in params.items():
            if name in state:
                assert state[name].shape == val.shape, "%s size has changed from %s to %s" % \
                                                       (name, state[name].shape, val.shape)
                state[name].copy_(val)
            else:
                print("WARNING: %s not in model during model loading!" % name)

    def save(self, path):
        torch.save(self.state_dict(), path)


class Skynet_model_base(BaseModel):
    def __init__(self, name, inputs: list[Input_part], inner: Inner_part, outputs: list[Out_part]):
        super(Skynet_model_base, self).__init__()
        self.name = name
        self.inputs = nn.ModuleList(inputs)
        self.inner = inner
        self.outputs = nn.ModuleList(outputs)

    def forward(self, x, **kwargs):
        x = [input(x) for input in self.inputs]
        x = torch.stack(x)
        x = self.inner(x)
        x = (out(x) for out in self.outputs)
        return x
