import copy


class Function_pool:
    def __init__(self, func_dict={}):
        self._command = func_dict

    @property
    def ALL(self) -> list:
        return copy.deepcopy([funct for group in self._command.values() for funct in group])

    @property
    def NAME2FUNC(self) -> dict:
        return {x.name: x for x in self.ALL}
