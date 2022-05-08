def get_used_indices(program):
    used = set()
    for statement in program.statements:
        used |= set(statement.args)
    return used


def get_unused_indices(program):
    """Returns unused indices of variables/statements in program."""
    used = get_used_indices(program)
    all_indices = set(range(len(program.var_types) - 1))
    return all_indices - used


class Program_base(object):
    """
    Attributes:
        input_types: List of Type (INT,LIST) representing the inputs
        statements: List of statements that were done so far.
    """

    def __init__(self, input_types, statements):
        self.input_types = input_types
        self.statements = statements
        self.var_types = self.input_types + [statement.output_type for statement in self.statements]
        self._encoded = None

    def encode(self) -> str:
        '''encode program'''
        pass

    @property
    def encoded(self):
        if self._encoded is None:
            self._encoded = self.encode()
        return self._encoded

    def __str__(self):
        return self.encoded

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return self.encoded < other.encoded

    def __len__(self):
        return len(self.statements)

    def __hash__(self):
        return hash(self.encoded)

    @classmethod
    def parse(cls, encoding):
        """parse from encoding
        Attributes:
            encoding:encoded program
        Returns:
            decoded Program object
        """
        pass

    def __call__(self, *inputs):
        """ program evaluate method
        Attributes:
            inputs: input of Program
        Returns out result of Program """
        pass
