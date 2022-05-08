from skynet.dsl.value import Value_base


# todo dsl_value input not forget
class Example(object):
    def __init__(self, inputs, output, dsl_value=Value_base):
        self.inputs = [dsl_value.construct(input) for input in inputs]
        self.output = dsl_value.construct(output)

    @classmethod
    def from_dict(cls, dict, dsl_value=Value_base):
        return Example(dict['inputs'], dict['output'], dsl_value)

    @classmethod
    def from_line(cls, line, dsl_value=Value_base):
        return [Example.from_dict(x, dsl_value) for x in (line['examples'])]
