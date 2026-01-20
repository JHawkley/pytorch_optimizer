from pytorch_optimizer.optimizer import Amos, OrthoGrad

class OrthoAmos(OrthoGrad):
    """
    Simple application of OrthoGrad to Amos to make using it in Kohya_ss easier.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(Amos, **kwargs)

    def __str__(self) -> str:
        return 'OrthoAmos'