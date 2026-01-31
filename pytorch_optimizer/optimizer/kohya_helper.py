from typing import Callable, Dict, Optional, Union
from warnings import warn

import torch
import torch.nn as nn

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup, State
from pytorch_optimizer.optimizer.orthograd import OrthoGrad
from pytorch_optimizer.optimizer.lookahead import Lookahead
from pytorch_optimizer.optimizer.schedulefree import ScheduleFreeWrapper

# Some optimizers want to work with `nn.Module`, but Kohya only provides raw parameter groups.
KOHYA_INCOMPATIBLE = (
    'lomo',
    'adalomo',
    'adammini',
    'muon',
    'adamuon',
    'adago',
)

class KohyaHelper(BaseOptimizer):
    """A wrapper optimizer to make this library easier to use in Kohya_ss.

    Kohya_ss expects two positional arguments with an initializer signature similar to:
        `(params, lr, **kwargs) -> None`

    Because of this, using `pytorch_optimizer.create_optimizer` as an optimizer in Kohya_ss
    fails because its positional arguments are `(model, optimizer_name, lr, **kwargs)` instead.
    This class resolves that issue by accepting positional arguments as Kohya_ss expects them,
    and then passing them on to `create_optimizer` itself, wrapping the resulting optimizer.

    Args:
        params (Parameters|Module): A Torch module, an iterable of parameters to optimize, or
            dicts defining parameter groups.
        lr (float): Learning rate.
        optimizer_name (str): Name of the optimizer; required.
        use_lookahead (bool): Wrap the optimizer with Lookahead.
        use_orthograd (bool): Wrap the optimizer with OrthoGrad.
        use_schedulefree (bool): Wrap the optimizer with ScheduleFreeWrapper.
        schedulefree_momentum (float): The momentum to use if using ScheduleFree.
    """

    def __init__(
        self,
        params: Union[Parameters, nn.Module],
        lr: float = 1e-3,
        optimizer_name: Optional[str] = None,
        use_lookahead: bool = False,
        use_orthograd: bool = False,
        use_schedulefree: bool = False,
        schedulefree_momentum: float = 0.9,
        **kwargs,
    ) -> None:
        from pytorch_optimizer.optimizer import create_optimizer, load_optimizer, OPTIMIZER

        if optimizer_name is None:
            raise ValueError(f'optimizer_name must be provided')
        
        if isinstance(params, nn.Module):
            # Just in case someone does provide a module, we can defer to `create_optimizer`.
            optimizer = create_optimizer(
                params, optimizer_name, lr,
                use_lookahead=use_lookahead,
                use_orthograd=use_orthograd,
                **kwargs
            )
        else:
            # Unfortunately, we're reimplementing most of `create_optimizer` here.
            # It currently does not work well with the parameter groups Kohya provides.
            # In particular, `params[*]['lr']` is usually tossed away, breaking the feature
            # that allows the UNet and text encoder to have separate learning rates.

            if optimizer_name in KOHYA_INCOMPATIBLE:
                raise ValueError(f'optimizer {optimizer_name} is incompatible with KohyaHelper')

            self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
            self._optimizer_step_post_hooks: Dict[int, Callable] = {}

            optimizer_class: OPTIMIZER = load_optimizer(optimizer_name)

            if optimizer_name == 'alig':
                optimizer = optimizer_class(params, max_lr=lr, **kwargs)
            else:
                optimizer = optimizer_class(params, lr=lr, **kwargs)
            
            if use_schedulefree:
                # Make sure the explicit `momentum` is not accidentally overridden by `kwargs`.
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'momentum'}
                optimizer = ScheduleFreeWrapper(optimizer, momentum=schedulefree_momentum, **filtered_kwargs)

            if use_orthograd:
                optimizer = OrthoGrad(optimizer, **kwargs)

            if use_lookahead:
                if optimizer_name in ('ranger', 'ranger21', 'ranger25'):
                    warn(f'{optimizer} already has a Lookahead variant.', UserWarning, stacklevel=1)
                else:
                    optimizer = Lookahead(
                        optimizer,
                        k=kwargs.get('k', 5),
                        alpha=kwargs.get('alpha', 0.5),
                        pullback_momentum=kwargs.get('pullback_momentum', 'none'),
                    )

        self.optimizer = optimizer

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
