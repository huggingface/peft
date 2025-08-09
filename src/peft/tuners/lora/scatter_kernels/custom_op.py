# Standard
from typing import Any, Callable, Iterable

# Third Party
import torch

try:
    # Third Party
    from torch.library import custom_op

    _IS_CUSTOM_OP_IN_PYTORCH = True
except:
    _IS_CUSTOM_OP_IN_PYTORCH = False


class _IdentityOp:
    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def register_fake(self, fn: Callable) -> Callable:
        return fn


def torch_custom_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: torch.device = None,
    schema: str | None = None,
) -> Callable | _IdentityOp:
    if _IS_CUSTOM_OP_IN_PYTORCH:
        op = custom_op(
            name,
            fn,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
        )
    else:
        op = _IdentityOp if fn is None else _IdentityOp(fn)

    return op