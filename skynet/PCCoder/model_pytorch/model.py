import numpy as np

from skynet.PCCoder.model_pytorch.model_part import *
from skynet.model_pytorch.model import Skynet_model_base


class PCCoder(Skynet_model_base):
    def __init__(self, name):
        inputs = [PCCoder_Input_part(f"{name}_Input")]
        inner = PCCoder_inner_part(f"{name}_Inner")
        outputs = [PCCoder_statement(f"{name}_statement"),
                   PCCoder_drophead(f"{name}_drophead"),
                   PCCoder_operator_head(f"{name}_operator_head")
                   ]
        super(PCCoder, self).__init__(name, inputs, inner, outputs)

    def forward(self, x, get_operator_head=True):
        num_batches = x.size()[0]
        x = self.inputs[0](x, num_batches=num_batches)
        x = self.inner(x, num_batches=num_batches)
        if get_operator_head:
            return self.outputs[0](x), self.outputs[1](x), self.outputs[2](x)
        else:
            return self.outputs[0](x), self.outputs[1](x)

    def predict(self, x):
        statement_pred, drop_pred, _ = self.forward(x)
        statement_probs = F.softmax(statement_pred, dim=1).data
        drop_indx = np.argmax(drop_pred.data.cpu().numpy(), axis=-1)
        return np.argsort(statement_probs.cpu().numpy()), statement_probs, drop_indx
