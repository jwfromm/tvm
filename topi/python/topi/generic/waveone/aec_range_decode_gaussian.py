import tvm
from ..vision import _default_schedule

@tvm.target.generic_func
def schedule_aec_range_decode_gaussian(outs):
    """Schedule for aec_range_decode_gaussian operator.

    Parameters
    ----------
    outs: Array of Tensor
      The indices that would sort an input array along
      the given axis.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs, False)
