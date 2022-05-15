import numpy as np
import tensorflow as tf
from keras import Model
from keras.activations import selu, sigmoid
from keras.engine.keras_tensor import KerasTensor
from keras.layers import *
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy
from keras.optimizer_v2.adam import Adam

from skynet.DeepCoder.env.operator import num_operators
from skynet.DeepCoder.env.statement import num_statements
from skynet.PCCoder.params import pcCoder_params as params
from skynet.model_tf.model import Skynet_model_base
from skynet.model_tf.model_part import Input_part, Inner_part, Out_part


class PCCoder_Input_part(Input_part):

    def __init__(self, part_name) -> None:
        super().__init__(part_name, [], None, None)
        input_l = InputLayer((params.num_examples, params.state_len, params.max_list_len + params.type_vector_len))
        self.prepare_layer_name(input_l)
        self.layers[input_l.name] = input_l

        types = tf.slice(input_l.output, [0, 0, 0, 0], [-1, -1, -1, params.type_vector_len])
        types = tf.cast(types, tf.float32)
        values = tf.slice(input_l.output, [0, 0, 0, params.type_vector_len], [-1, -1, -1, -1])

        flatten = Flatten()
        self.prepare_layer_name(flatten)

        self.layers[flatten.name] = flatten
        values_embed = flatten(values)

        E = Embedding(params.integer_range + 1, params.embedding_size, input_length=1, name='Embedding')
        self.prepare_layer_name(E)
        self.layers[E.name] = E

        result_emb = E(values_embed)
        result_emb = tf.reshape(result_emb, [-1, params.num_examples, params.state_len,
                                             params.max_list_len * params.embedding_size])

        concat = Concatenate()
        self.prepare_layer_name(concat)
        self.layers[concat.name] = concat

        result = concat([result_emb, types])

        self.input_layer = input_l
        self.out_layer = concat
        self.out = result


class PCCoder_inner_part(Inner_part):

    def __init__(self, part_name, input_to_part_layer: KerasTensor) -> None:
        super().__init__(part_name, [], None, None)

        input_l = Dense(params.var_encoder_size, activation=selu, dtype=tf.float32)
        self.prepare_layer_name(input_l)
        self.layers[input_l.name] = input_l

        input_l(input_to_part_layer)
        inp_result = tf.reshape(input_l.output,
                                [-1, params.num_examples, input_l.output.shape[2] * input_l.output.shape[3]])
        input = [inp_result]

        dense = Dense(params.var_encoder_size, activation=selu)
        self.prepare_layer_name(dense)
        self.layers[dense.name] = dense
        dense_output = dense(inp_result)
        input.append(dense_output)
        for i in range(0, 9 - 1):
            dense = Dense(params.var_encoder_size, activation=selu)
            self.prepare_layer_name(dense)
            self.layers[dense.name] = dense

            concat = Concatenate(axis=-1, dtype=tf.float32)
            self.prepare_layer_name(concat)
            self.layers[concat.name] = concat

            dense_layer_input = concat(input)
            dense_output = dense(dense_layer_input)
            input.append(dense_output)
        dense = Dense(params.dense_output_size, activation=selu)
        self.prepare_layer_name(dense)
        self.layers[dense.name] = dense

        concat = Concatenate(axis=-1)
        self.prepare_layer_name(concat)
        self.layers[concat.name] = concat

        dense_layer_input = concat(input)
        dense_output = dense(dense_layer_input)
        input.append(dense_output)
        result = tf.math.reduce_mean(dense_output, axis=1)
        self.input_layer = input_l
        self.out_layer = dense
        self.out = result


class PCCoder_statement(Out_part):

    def __init__(self, part_name, input_to_part_layer: KerasTensor) -> None:
        super().__init__(part_name, [], None, None)

        input_l = Dense(num_statements, activation=None)
        self.prepare_layer_name(input_l)
        self.layers[input_l.name] = input_l

        result = input_l(input_to_part_layer)
        self.input_layer = input_l
        self.out_layer = input_l
        self.out = result


class PCCoder_drophead(Out_part):

    def __init__(self, part_name, input_to_part_layer: KerasTensor) -> None:
        super().__init__(part_name, [], None, None)

        input_l = Dense(params.max_program_vars, activation=sigmoid)
        self.prepare_layer_name(input_l)
        self.layers[input_l.name] = input_l

        result = input_l(input_to_part_layer)
        self.input_layer = input_l
        self.out_layer = input_l
        self.out = result


class PCCoder_operator_head(Out_part):

    def __init__(self, part_name, input_to_part_layer: KerasTensor) -> None:
        super().__init__(part_name, [], None, None)

        input_l = Dense(num_operators, activation=None)
        self.prepare_layer_name(input_l)
        self.layers[input_l.name] = input_l

        result = input_l(input_to_part_layer)
        self.input_layer = input_l
        self.out_layer = input_l
        self.out = result


class PCCoder(Skynet_model_base):

    def __init__(self, name: str, inputs: list[Input_part] = None, inner: Inner_part = None,
                 outputs: list[Out_part] = None,
                 model: Model = None, log_dir=None) -> None:
        if inputs is None:
            inputs = [PCCoder_Input_part(f"{name}_Input")]
        if inner is None:
            inner = PCCoder_inner_part(f"{name}_Inner", inputs[0].out)
        if outputs is None:
            outputs = [PCCoder_statement(f"{name}_statement", inner.out),
                       PCCoder_drophead(f"{name}_drophead", inner.out),
                       PCCoder_operator_head(f"{name}_operator_head", inner.out)
                       ]
        super().__init__(name, inputs, inner, outputs, model,log_dir)

    def create_model(self):
        super().create_model()
        opt = Adam(learning_rate=0.001)
        losses = {
            self.outputs[0].out_layer.name: CategoricalCrossentropy(from_logits=True),  # statement_criterion
            self.outputs[1].out_layer.name: BinaryCrossentropy(),  # drop_criterion
            self.outputs[2].out_layer.name: CategoricalCrossentropy(from_logits=True)  # operator_criterion
        }
        loss_weights = {
            self.outputs[0].out_layer.name: 1.0,  # statement_criterion
            self.outputs[1].out_layer.name: 1.0,  # drop_criterion
            self.outputs[2].out_layer.name: 1.0,  # operator_criterion
        }
        self.model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)

    def predict(self, x):
        statement_pred, drop_pred, _ = self(x)
        statement_probs = tf.keras.activations.softmax(statement_pred, axis=1).numpy()
        drop_indx = np.argmax(drop_pred, axis=-1)
        return np.argsort(statement_probs), statement_probs, drop_indx

    def __call__(self, x: KerasTensor, training=False):
        assert x.shape[1] == params._num_examples, "Invalid num of examples received!"
        assert x.shape[2] == params.state_len, "Example with invalid length received!"
        assert x.shape[3] == params.max_list_len + params.type_vector_len, "Example with invalid length received!"
        return self.model(x, training=training)
