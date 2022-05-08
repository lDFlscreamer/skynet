import itertools

from skynet.DeepCoder.params import deepCoder_params as params
from skynet.DeepCoder.dsl.DeepCoder_types import INT, LIST
from skynet.DeepCoder.dsl.impl import function_pool as f_pool
from skynet.dsl.types import FunctionType
from skynet.env.statement import Statement


def build_statement_space():
    statements = []
    for func in f_pool.ALL_FUNCTIONS:
        input_type = func.input_type
        if not isinstance(input_type, tuple):
            input_type = (input_type,)

        argslists = []
        for type in input_type:
            if type in [LIST, INT]:
                argslists.append(range(params.max_program_vars))
            elif isinstance(type, FunctionType):
                argslists.append([x for x in f_pool.LAMBDAS if x.type == type])
            else:
                raise ValueError("Invalid input type encountered!")
        statements += [Statement(func, x) for x in list(itertools.product(*argslists))]

    return statements


statement_space = build_statement_space()
num_statements = len(statement_space)
index_to_statement = dict([(indx, statement) for indx, statement in enumerate(statement_space)])
statement_to_index = {v: k for k, v in index_to_statement.items()}
