from pytorch_optimizer.optimizer.amos import Amos
from pytorch_optimizer.optimizer.orthograd import OrthoGrad
from pytorch_optimizer.optimizer.lookahead import Lookahead
from pytorch_optimizer.base.type import Parameters

class OrthoAmos(OrthoGrad):
    """
    Simple application of OrthoGrad to Amos to make using it in Kohya_ss easier.
    """

    def __init__(self, params: Parameters, lr: float = 1e-3, lookahead: bool = False, **kwargs) -> None:
        # Extract Lookahead-specific parameters from kwargs
        k = kwargs.pop('k', 5)
        alpha = kwargs.pop('alpha', 0.5)
        pullback_momentum = kwargs.pop('pullback_momentum', 'none')
        
        optimizer = Amos(params, lr, **kwargs)
        if lookahead:
            optimizer = Lookahead(optimizer, k=k, alpha=alpha, pullback_momentum=pullback_momentum)
        super().__init__(optimizer)

    def __str__(self) -> str:
        return 'OrthoAmos'