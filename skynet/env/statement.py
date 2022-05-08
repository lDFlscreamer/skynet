class Statement(object):
    def __init__(self, function, args):
        self.function = function
        self.args = tuple(args)

        self.input_types = function.input_type
        self.output_type = self.function.output_type

        if not isinstance(self.input_types, tuple):
            self.input_types = (self.input_types,)

    def __repr__(self):
        return "<Statement: %s %s>" % (self.function, self.args)

    def __eq__(self, other):
        if not isinstance(other, Statement):
            return False
        return self.function == other.function and self.args == other.args

    def __hash__(self):
        return hash(str(self))
