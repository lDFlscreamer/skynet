from skynet.DeepCoder.params import deepCoder_params
from skynet.DeepCoder.dsl.DeepCoder_value import Value, IntValue, ListValue, NULLVALUE
from skynet.dsl.function import Function_base,NullInputError,OutputOutOfRangeError


def in_range(val, params=deepCoder_params):
    if isinstance(val, IntValue):
        val = ListValue([val.val])
    for x in val.val:
        if x < params.integer_min or x > params.integer_max:
            return False
    return True


class Function(Function_base):

    def __call__(self, *args):
        for arg in args:
            if arg == NULLVALUE:
                raise NullInputError('{}({})'.format(self.name, args))
        raw_args = [x.val for x in args]
        output_raw = self.val(*raw_args)
        output_val = Value.construct(output_raw, self.output_type)
        if output_val != NULLVALUE and not in_range(output_val):
            raise OutputOutOfRangeError('{}({})'.format(self.name, args))
        return output_val
