class NcairLmsError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(NcairLmsError):
    pass


class ModelUnavailableError(NcairLmsError):
    pass


class InvalidModelOutputError(NcairLmsError):
    pass


class RetrievalError(NcairLmsError):
    pass


class InvalidToolArgumentsError(NcairLmsError):
    pass
