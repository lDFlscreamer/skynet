class Default_params:

    def __init__(self, num_inputs=3, num_examples=5, max_program_len=8) -> None:
        super().__init__()
        self._num_inputs = num_inputs
        self._num_examples = num_examples

        self._max_program_len = max_program_len

    @property
    def num_inputs(self):
        if self._num_inputs is None:
            self._num_inputs = 3
        return self._num_inputs

    @num_inputs.setter
    def num_inputs(self, value):
        self._num_inputs = value

    @property
    def num_examples(self):
        if self._num_examples is None:
            self._num_examples = 5
        return self._num_examples

    @num_examples.setter
    def num_examples(self, value):
        self._num_examples = value

    @property
    def max_program_len(self):
        if self._max_program_len is None:
            self._max_program_len = 8
        return self._max_program_len

    @max_program_len.setter
    def max_program_len(self, value):
        self._max_program_len = value

    @property
    def max_program_vars(self):
        return self._max_program_len + self._num_inputs

    @property
    def state_len(self):
        return self.max_program_vars + 1
