from skynet.env.statement import Statement

from skynet.DeepCoder.dsl.DeepCoder_types import INT, LIST
from skynet.DeepCoder.dsl.DeepCoder_value import NULLVALUE
from skynet.DeepCoder.dsl.impl import function_pool as f_pool
from skynet.dsl.program import Program_base


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


class Program(Program_base):
    """
    Attributes:
        input_types: List of Type (INT,LIST) representing the inputs
        statements: List of statements that were done so far.
    """

    def __init__(self, input_types, statements):
        super().__init__(input_types, statements)

    def encode(self):
        toks = [x.name for x in self.input_types]
        for statement in self.statements:
            parts = [x for x in [statement.function] + list(statement.args) if x is not None]
            tok = ','.join(map(str, parts))
            toks.append(tok)

        return '|'.join(toks)



    @classmethod
    def parse(cls, encoding):
        input_types = []
        statements = []

        def get_statement(term):
            args = []
            parts = term.split(',')

            func = f_pool.NAME2FUNC[parts[0]]

            for inner in parts[1:]:
                if inner.isdigit():
                    args.append(int(inner))
                else:
                    args.append(f_pool.NAME2FUNC[inner])

            return Statement(func, args)

        for tok in encoding.split('|'):
            if ',' in tok:
                statements.append(get_statement(tok))
            else:
                if tok == INT.name:
                    typ = INT
                elif tok == LIST.name:
                    typ = LIST
                else:
                    raise ValueError('invalid input type {}'.format(tok))
                input_types.append(typ)

        return Program(input_types, statements)

    def __call__(self, *inputs):
        if not self.statements:
            return NULLVALUE
        vals = list(inputs)
        for statement in self.statements:
            args = []
            for arg in statement.args:
                if isinstance(arg, int):
                    args.append(vals[arg])
                else:
                    args.append(arg)
            val = statement.function(*args)
            vals.append(val)
        return vals[-1]
