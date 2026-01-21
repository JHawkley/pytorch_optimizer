from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup, State


class KohyaHelper(BaseOptimizer):
    """A wrapper optimizer to make this library easier to use in Kohya_ss.

    Kohya_ss expects two positional arguments with an initializer signature similar to:
        `(params, lr, **kwargs) -> None`

    Because of this, using `pytorch_optimizer.create_optimizer` as an optimizer in Kohya_ss
    fails because its positional arguments are `(model, optimizer_name, lr, **kwargs)` instead.
    This class resolves that issue by accepting positional arguments as Kohya_ss expects them,
    and then passing them on to `create_optimizer` itself, wrapping the resulting optimizer.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        optimizer_name (str): Name of the optimizer; required.
        use_lookahead (bool): Wrap the optimizer with Lookahead.
        use_orthograd (bool): Wrap the optimizer with OrthoGrad.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        optimizer_name: Optional[str] = None,
        use_lookahead: bool = False,
        use_orthograd: bool = False,
        **kwargs,
    ) -> None:
        from pytorch_optimizer.optimizer import create_optimizer

        if optimizer_name is None:
            raise ValueError(f'optimizer_name must be provided')
        
        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        # If params is already an nn.Module, use it directly
        # Otherwise, wrap parameters in a dummy module to make them compatible with create_optimizer
        if isinstance(params, nn.Module):
            model = params
        else:
            # Wrap parameters in a dummy module
            class DummyModule(nn.Module):
                def __init__(self, params):
                    super().__init__()
                    # Check if params is already a list of parameter groups
                    # (each group is a dict with 'params' key)
                    if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
                        # It's already parameter groups, use as-is
                        self._param_groups = params
                    else:
                        # It's raw parameters, create a simple parameter group
                        # Convert to list if it's a generator/iterator
                        self._param_groups = [{'params': list(params)}]
                
                def parameters(self, recurse: bool = True):
                    # Return flattened parameters from all groups
                    for group in self._param_groups:
                        for p in group['params']:
                            yield p
                
                def named_parameters(self, prefix: str = '', recurse: bool = True):
                    # Generate names for parameters
                    for i, group in enumerate(self._param_groups):
                        for j, p in enumerate(group['params']):
                            name = f'{prefix}group_{i}.param_{j}'
                            yield name, p
                
                def named_modules(self, memo=None, prefix: str = '', remove_duplicate: bool = True):
                    # For compatibility with optimizers that expect named_modules
                    # Return self as the only module
                    yield prefix, self
            
            model = DummyModule(params)
        
        self.optimizer: Optimizer = create_optimizer(
            model, optimizer_name, lr,
            use_lookahead=use_lookahead,
            use_orthograd=use_orthograd,
            **kwargs
        )

    def __str__(self) -> str:
        return f'KohyaHelper[{type(self.optimizer).__name__}]'

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self) -> State:
        return self.optimizer.state

    @property
    def defaults(self) -> Defaults:
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, value: Defaults) -> None:
        self.optimizer.defaults = value

    def state_dict(self) -> State:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: State) -> None:
        self.optimizer.load_state_dict(state_dict)

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        # KohyaHelper doesn't need group initialization, but must implement
        # because BaseOptimizer requires it
        pass

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        return self.optimizer.step(closure)
