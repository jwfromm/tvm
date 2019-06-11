import topi
from topi.util import get_const_int, get_const_tuple
from .. import op as reg
from ..op import OpPattern, schedule_injective

# aec_get_probs
@reg.register_schedule("waveone.aec_get_probs")
def schedule_aec_get_probs(_, outputs, target):
    with target:
        return topi.generic.schedule_aec_get_probs(outputs)

@reg.register_compute("waveone.aec_get_probs")
def compute_aec_get_probs(attrs, inputs, out_dtype, target):
    return [topi.waveone.aec_get_probs(inputs[0], inputs[1])]

reg.register_pattern("waveone.aec_get_probs", OpPattern.OPAQUE)
