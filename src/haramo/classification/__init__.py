from ._instantiation import (
    instantiate_model,
    instantiate_pipeline,
)

from ._optimisation import (
    train,
    nested_crossval,
    magic_now,
)

__all__ = [
    "instantiate_model",
    "instantiate_pipeline",
    "train",
    "nested_crossval",
    "magic_now",
]
