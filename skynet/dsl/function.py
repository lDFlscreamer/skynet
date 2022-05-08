from skynet.dsl.types import FunctionType
from skynet.dsl.value import Value_base


class OutputOutOfRangeError(Exception):
    pass


class NullInputError(Exception):
    pass


class Function_base(Value_base):
    def __init__(self, name, f, input_type, output_type):
        super(Function_base, self).__init__(f, FunctionType(input_type, output_type))
        self.name = name

    def __call__(self, *args):
        ''' function call '''
        pass

    @property
    def input_type(self):
        return self.type.input_type

    @property
    def output_type(self):
        return self.type.output_type

    def __eq__(self, other):
        if not isinstance(other, Function_base):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name
