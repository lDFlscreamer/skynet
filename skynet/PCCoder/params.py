from skynet.default_params import Default_params


class PCCoder_params(Default_params):
    def __init__(self) -> None:
        super().__init__()

        self._integer_min = -256
        self._integer_max = 255

        self._max_list_len = 20

        self._type_vector_len = 2

        self._embedding_size = 20

        self._var_encoder_size = 56

        self._dense_output_size = 256
        self._dense_num_layers = 10
        self._dense_growth_size = 56

        self._dfs_max_width = 50

        self._cab_beam_size = 100
        self._cab_width = 10
        self._cab_width_growth = 10

    @property
    def integer_min(self):
        return self._integer_min

    @integer_min.setter
    def integer_min(self, value):
        self._integer_min = value

    @property
    def integer_max(self):
        return self._integer_max

    @integer_max.setter
    def integer_max(self, value):
        self._integer_max = value

    @property
    def integer_range(self):
        return self._integer_max - self._integer_min + 1

    @property
    def max_list_len(self):
        return self._max_list_len

    @max_list_len.setter
    def max_list_len(self, value):
        self._max_list_len = value

    # todo rewrite

    @property
    def type_vector_len(self):
        return self._type_vector_len

    # todo rewrite

    @type_vector_len.setter
    def type_vector_len(self, value):
        self._type_vector_len = value

    # todo rewrite

    @property
    def embedding_size(self):
        return self._embedding_size

    @embedding_size.setter
    def embedding_size(self, value):
        self._embedding_size = value

    @property
    def var_encoder_size(self):
        return self._var_encoder_size

    @var_encoder_size.setter
    def var_encoder_size(self, value):
        self._var_encoder_size = value

    @property
    def dense_output_size(self):
        return self._dense_output_size

    @dense_output_size.setter
    def dense_output_size(self, value):
        self._dense_output_size = value

    @property
    def dense_num_layers(self):
        return self._dense_num_layers

    @dense_num_layers.setter
    def dense_num_layers(self, value):
        self._dense_num_layers = value

    @property
    def dense_growth_size(self):
        return self._dense_growth_size

    @dense_growth_size.setter
    def dense_growth_size(self, value):
        self._dense_growth_size = value

    @property
    def dfs_max_width(self):
        return self._dfs_max_width

    @dfs_max_width.setter
    def dfs_max_width(self, value):
        self._dfs_max_width = value

    @property
    def cab_beam_size(self):
        return self._cab_beam_size

    @cab_beam_size.setter
    def cab_beam_size(self, value):
        self._cab_beam_size = value

    @property
    def cab_width(self):
        return self._cab_width

    @cab_width.setter
    def cab_width(self, value):
        self._cab_width = value

    @property
    def cab_width_growth(self):
        return self._cab_width_growth

    @cab_width_growth.setter
    def cab_width_growth(self, value):
        self._cab_width_growth = value


pcCoder_params = PCCoder_params()
