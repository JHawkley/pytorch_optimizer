from pytorch_optimizer.optimizer.amos import Amos
from pytorch_optimizer.optimizer.orthograd import OrthoGrad
from pytorch_optimizer.base.type import Parameters

class OrthoAmos(OrthoGrad):
    """
    Simple application of OrthoGrad to Amos to make using it in Kohya_ss easier.
    """

    def __init__(self, params: Parameters, lr: float = 1e-3, **kwargs) -> None:
        optimizer = Amos(params, lr, **kwargs)
        super().__init__(optimizer)

    def __str__(self) -> str:
        return 'OrthoAmos'