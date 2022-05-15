import pickle

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from keras.models import load_model

from skynet.model_tf.model_part import *


class Skynet_model_base:
    model_name: str
    inputs: list[Input_part]
    inner: Inner_part
    outputs: list[Out_part]
    model: Model

    def __init__(self, name: str, inputs: list[Input_part], inner: Inner_part, outputs: list[Out_part],
                 model: Model = None, log_dir=None) -> None:
        super().__init__()
        self.model_name = name
        self.inputs = inputs
        self.inner = inner
        self.outputs = outputs
        if log_dir:
            self.set_tensorboard(log_dir=log_dir)
        if model is not None:
            self.model = model
        else:
            self.create_model()

    @property
    def loggable(self):
        return self.summary_writer is not None

    @property
    def input(self):
        return self.model.inputs

    @property
    def output(self):
        return self.model.outputs

    def get_config(self) -> dict:
        return {
            'model_name': self.model_name,
            'inputs': [input.get_config() for input in self.inputs],
            'inner': self.inner.get_config(),
            'outputs': [output.get_config() for output in self.outputs]
        }

    def set_tensorboard(self, log_dir: str):
        if log_dir:
            self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    def create_model(self):
        ''' need to declare model.compile'''
        self.model = Model([input.input for input in self.inputs], [output.out for output in self.outputs],
                           name=self.model_name)

    def predict(self, x):
        return self(x)

    def __call__(self, x: KerasTensor):
        return self.model(x)

    @classmethod
    def from_config_and_model(cls, config: dict, model: Model):
        model_name = config.get('model_name')
        inputs = config.get('inputs')
        inputs = [Input_part.from_config_and_model(input, model) for input in inputs]
        inner = Inner_part.from_config_and_model(config['inner'], model)
        outputs = config.get('outputs')
        outputs = [Out_part.from_config_and_model(output, model) for output in outputs]
        return cls(model_name, inputs, inner, outputs, model)

    def save(self, folderPath):
        with open(f'{folderPath}/config.pkl', 'ba+') as f:
            pickle.dump(self.get_config(), f)
        self.save_model(folderPath)

    def save_config(self, folderPath):
        with open(f'{folderPath}\\config.pkl', 'a+b') as f:
            pickle.dump(self.get_config(), f)

    def save_model(self, folderPath):
        self.model.save(f'{folderPath}\\model', save_format='h5')

    @classmethod
    def load_model(cls, folderPath):
        return load_model(f'{folderPath}\\model')

    @classmethod
    def load_config(cls, folderPath):
        with open(f'{folderPath}\\config.pkl', 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load(cls, folderPath):
        config = cls.load_config(folderPath)
        model = cls.load_model(folderPath)
        return cls.from_config_and_model(config, model)
