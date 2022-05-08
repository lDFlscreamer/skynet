from skynet.dsl.types import PrimitiveType


class Value_base(object):
    def __init__(self, val, typ: PrimitiveType):
        self.val = val
        self.type = typ
        self.name = str(self.val)

    def __eq__(self, other):
        if not isinstance(other, Value_base):
            return False
        return self.val == other.val and self.type == other.type

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @classmethod
    def construct(cls, val, typ=None):
        '''construct val and type here'''

    pass
