from .Linear import Linear
from .NLinear import NLinear
from .DLinear import DLinear

# Automatically create a list of all classes imported in this file
import sys
import inspect
MODELS = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)]
print(f'{MODELS = }')