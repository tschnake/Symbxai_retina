from torch import Tensor


def gamma(
        gam: float = 0.0
) -> Tensor:

    def modify_parameters(parameters: Tensor):
        return parameters + (gam * parameters.clamp(min=0))

    return modify_parameters
