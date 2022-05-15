from torch import nn


class Model_part(nn.Module):
    def __init__(self, name):
        super(Model_part, self).__init__()
        self.name = name
        self.layers = []

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def trainable(self) -> bool:
        result: bool
        result = True

        for parameter in self.parameters():
            result = result and parameter.requires_grad
        return result

    @trainable.setter
    def trainable(self, bool: bool):
        for parameter in self.parameters():
            parameter.requires_grad = bool


class Input_part(Model_part):
    def __init__(self, name):
        super(Input_part, self).__init__(name)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x


class Inner_part(Model_part):
    def __init__(self, name):
        super(Inner_part, self).__init__(name)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

class Out_part(Model_part):
    def __init__(self, name):
        super(Out_part, self).__init__(name)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x
