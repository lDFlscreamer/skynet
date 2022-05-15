import torch
import torch.nn.functional as F
from torch import nn

from skynet.DeepCoder.env.operator import num_operators
from skynet.DeepCoder.env.statement import num_statements
from skynet.PCCoder.model.encoder import DenseBlock
from skynet.PCCoder.params import pcCoder_params as params
from skynet.model_pytorch.model_part import Input_part, Inner_part, Out_part


class PCCoder_Input_part(Input_part):
    def __init__(self, name):
        super(PCCoder_Input_part, self).__init__(name)
        self.name = name
        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)

    def forward(self, x, **kwargs):
        num_batches = kwargs['num_batches']
        assert num_batches is not None, f"num_batches not provide in {self.name}"
        x = self.embed_state(x, num_batches)
        x = F.selu(self.var_encoder(x))
        x = x.view(num_batches, params.num_examples, -1)
        return x

    def embed_state(self, x, num_batches):
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        assert values.size()[1] == params._num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        assert num_batches == x.size()[0], "Provided num_batches do not match"

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, params._num_examples, params.state_len, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1)


class PCCoder_inner_part(Inner_part):
    def __init__(self, name):
        super(PCCoder_inner_part, self).__init__(name)
        self.name = name
        self.dense = DenseBlock(10, params.dense_growth_size, params.var_encoder_size * params.state_len,
                                params.dense_output_size)

    def forward(self, x, **kwargs):
        num_batches = kwargs['num_batches']
        assert num_batches is not None, f"num_batches not provide in {self.name}"
        x = self.dense(x)
        x = x.mean(dim=1)
        return x.view(num_batches, -1)


class PCCoder_statement(Out_part):
    def __init__(self, name):
        super(PCCoder_statement, self).__init__(name)
        self.name = name
        self.statement_head = nn.Linear(params.dense_output_size, num_statements)
        self.layers.append(self.statement_head)

    def forward(self, x, **kwargs):
        return self.statement_head(x)


class PCCoder_drophead(Out_part):
    def __init__(self, name):
        super(PCCoder_drophead, self).__init__(name)
        self.name = name
        self.drop_head = nn.Linear(params.dense_output_size, params.max_program_vars)
        self.layers.append(self.drop_head)

    def forward(self, x, **kwargs):
        return torch.sigmoid(self.drop_head(x))


class PCCoder_operator_head(Out_part):
    def __init__(self, name):
        super(PCCoder_operator_head, self).__init__(name)
        self.name = name
        self.operator_head = nn.Linear(params.dense_output_size, num_operators)
        self.layers.append(self.operator_head)

    def forward(self, x, **kwargs):
        return self.operator_head(x)
