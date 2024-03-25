import warnings

class TrainerWarning(Warning):
    def __init__(self, message):
        super().__init__(message)


class InferenceAgentWarning(Warning):
    def __init__(self, message):
        super().__init__()


class DatasetWarning(Warning):
    def __init__(self, message):
        super().__init__()


class TrainerError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InferenceAgentError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class DatasetError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)