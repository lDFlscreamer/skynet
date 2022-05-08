import numpy as np

from skynet.DeepCoder.dsl.DeepCoder_value import NULLVALUE
from skynet.DeepCoder.params import deepCoder_params as params
from skynet.env.env import ProgramState_base


class ProgramState(ProgramState_base):

    def get_encoding(self):
        encoded_vars = [var.encoded for var in self._vars]
        if len(encoded_vars) < params.max_program_vars:
            encoded_vars.extend([NULLVALUE.encoded] * (params.max_program_vars - len(self._vars)))
        return np.array(encoded_vars + [self.output.encoded])
