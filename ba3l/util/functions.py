import inspect
from collections import OrderedDict


def get_default_kwargs_dict(f):
    sig = inspect.signature(f)
    return OrderedDict(
        [
            (p.name, p.default)
            for p in sig.parameters.values()
            if p.default != inspect._empty
        ]
    )
