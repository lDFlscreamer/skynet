from keras import Model
from keras.layers import *


class NotObviousStructureError(Exception):
    pass


class Model_part:
    part_name: str
    input_layer: Layer
    out_layer: Layer
    layers: dict[str, Layer]

    def __init__(self, part_name, layers: list[Layer], part_input_layer=None, part_out_layer=None) -> None:
        super().__init__()
        self.part_name = part_name
        self.input_layer = part_input_layer
        self.out_layer = part_out_layer
        self._out = None
        self._input = None
        self.layers = {}
        if not len(layers) == 0:
            self.prepare_part(layers)

    def encode_layer_name(self, layer) -> str:
        name = f"{self.part_name}.{layer.__name__}_"
        index = len([value for key, value in self.layers.items() if name in key])
        return f"{name}{index}"

    def prepare_layer_name(self, layer: Layer):
        layer._name = self.encode_layer_name(layer.__class__)

    def decode_layers(self, layers) -> None:
        """decode Model_part args from list of layers

        Args:
            layers (list[Layer]):
        """
        filtered_layers = [layer for layer in layers if self.part_name in layer.name]
        if not len(filtered_layers) == 0:
            self.layers = {layer.name: layer for layer in filtered_layers}
            if self.out_layer is None:
                self.out_layer = filtered_layers[-1]
            if self.input_layer is None:
                self.input_layer = filtered_layers[0]
        else:
            raise NotObviousStructureError(f'cant found layer for {self.part_name} ')

    def prepare_part(self, layers: list[Layer]):
        for layer in layers:
            self.prepare_layer_name(layer)
            self.layers[layer.name] = layer

        if self.input_layer is None:
            self.input_layer = layers[0]
        if self.out_layer is None:
            self.out_layer = layers[-1]

    def decode_model(self, model) -> None:
        """decode Model_part args from list of layers

        Args:
            model (Model):
        """

        self.decode_layers(model.layers)

    def get_config(self):
        return {"part_name": self.part_name, "input": self.input_layer.name, "output": self.out_layer.name, }

    # def trainable_

    @property
    def trainable(self) -> bool:
        result: bool
        result = True
        for layer in self.layers.values():
            result = result and layer.trainable
        return result

    @trainable.setter
    def trainable(self, bool: bool):
        for layer in self.layers.values():
            layer.trainable = bool

    @property
    def out_shape(self) -> tuple:
        if self._out is None:
            return self.out_layer.output_shape if isinstance(self.out_layer, Layer) else self.out_layer.shape
        else:
            return self._out.shape

    @property
    def input_shape(self) -> tuple:
        if self._input is None:
            return self.input_layer.input_shape if isinstance(self.input_layer, Layer) else self.input_layer.shape
        else:
            return self._input.shape

    @property
    def out(self):
        if self._out is None:
            return self.out_layer.output
        else:
            return self._out

    @out.setter
    def out(self, out):
        self._out = out

    @property
    def input(self):
        if self._input is None:
            return self.input_layer.input
        else:
            return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @classmethod
    def from_config(cls, config: dict):
        obj = cls(part_name=config.get('part_name'), layers=[], part_input_layer=None, part_out_layer=None)
        return obj

    @classmethod
    def from_config_and_model(cls, config: dict, model: Model):
        obj = cls.from_config(config)
        obj.part_input_layer = model.get_layer(name=config.get('input'), )
        obj.part_out_layer = model.get_layer(name=config.get('output'))
        obj.decode_model(model=model)
        return obj


class Input_part(Model_part):

    @classmethod
    def only_input_layer(cls, part_name, shape: tuple):
        layers = [InputLayer(shape, name=part_name)]
        return cls(part_name, layers, None, None)


class Inner_part(Model_part):
    pass


class Out_part(Model_part):
    pass
