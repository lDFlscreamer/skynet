from skynet.DeepCoder.dsl.DeepCoder_function import Function
from skynet.dsl.function_pool import Function_pool
import copy


class DeepCoder_Function_pool(Function_pool):
    _command: dict[str, list[Function]]

    def __init__(self, lambdas: list[Function], higher_order_functions: list[Function],
                 first_order_functions: list[Function]):
        command = {"LAMBDAS": lambdas,
                   "HIGHER_ORDER_FUNCTIONS": higher_order_functions,
                   "FIRST_ORDER_FUNCTIONS": first_order_functions}
        super(DeepCoder_Function_pool, self).__init__(command)

    @property
    def LAMBDAS(self) -> list:
        return copy.deepcopy(self._command["LAMBDAS"])

    def add_LAMDAS(self, f: Function):
        self._command["LAMBDAS"].append(f)

    def add_LAMDAS(self, lf: list[Function]):
        self._command["LAMBDAS"].extend(lf)

    @property
    def HIGHER_ORDER_FUNCTIONS(self) -> list:
        return copy.deepcopy(self._command["HIGHER_ORDER_FUNCTIONS"])

    def add_HIGHER_ORDER_FUNCTIONS(self, f: Function):
        self._command["HIGHER_ORDER_FUNCTIONS"].append(f)

    def add_HIGHER_ORDER_FUNCTIONS(self, lf: list[Function]):
        self._command["HIGHER_ORDER_FUNCTIONS"].extend(lf)

    @property
    def FIRST_ORDER_FUNCTIONS(self) -> list:
        return copy.deepcopy(self._command["FIRST_ORDER_FUNCTIONS"])

    def add_FIRST_ORDER_FUNCTIONS(self, f: Function):
        self._command["FIRST_ORDER_FUNCTIONS"].append(f)

    def add_FIRST_ORDER_FUNCTIONS(self, lf: list[Function]):
        self._command["FIRST_ORDER_FUNCTIONS"].extend(lf)

    @property
    def ALL_FUNCTIONS(self) -> list:
        return copy.deepcopy(self.FIRST_ORDER_FUNCTIONS+self.HIGHER_ORDER_FUNCTIONS)

    @property
    def FUNCTIONS_AND_LAMBDAS(self) -> list:
        return self.ALL
