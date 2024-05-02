from torch import Tensor, zeros_like


def gamma(
        gam: float = 0.0,
        minimum: float = None,
        maximum: float = None,
        modify_bias: bool = True
) -> Tensor:

    def modify_parameters(parameters: Tensor, name: str = 'something'):
        if name == 'bias':
            raise NotImplementedError("this code produced relevance scores in the range of  10**20, I think we shouldn't use it")
            if not modify_bias:
                return zeros_like(parameters)
            else:
                return parameters + (gam * parameters.clamp(min=minimum, max=maximum))
        else:
            return parameters + (gam * parameters.clamp(min=minimum, max=maximum))

    return modify_parameters
