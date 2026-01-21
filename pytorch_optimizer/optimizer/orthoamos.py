from pytorch_optimizer.optimizer.amos import Amos
from pytorch_optimizer.optimizer.orthograd import OrthoGrad
from pytorch_optimizer.optimizer.lookahead import Lookahead
from pytorch_optimizer.base.type import Parameters

class OrthoAmos(OrthoGrad):
    """
    Simple application of OrthoGrad to Amos to make using it in Kohya_ss easier.
    """

    def __init__(self, params: Parameters, lr: float = 1e-3, use_lookahead: bool = False, **kwargs) -> None:
        optimizer = Amos(params, lr, **kwargs)
        if use_lookahead:
            optimizer = Lookahead(
                optimizer,
                k=kwargs.get('k', 5),
                alpha=kwargs.get('alpha', 0.5),
                pullback_momentum=kwargs.get('pullback_momentum', 'none'),
            )

        super().__init__(optimizer)

    def __str__(self) -> str:
        return 'OrthoAmos'