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
    with target:
        return [topi.waveone.aec_get_probs(inputs[0], inputs[1])]

reg.register_pattern("waveone.aec_get_probs", OpPattern.OPAQUE)

# aec_encode
@reg.register_schedule("waveone.aec_encode")
def schedule_aec_encode(_, outputs, target):
    with target:
        return topi.generic.schedule_aec_encode(outputs)

@reg.register_compute("waveone.aec_encode")
def compute_aec_encode(attrs, inputs, out_dtype, target):
    with target:
        out = topi.waveone.aec_encode(inputs[0], inputs[1])
        out = out if isinstance(out, list) else [out]
        return out

reg.register_pattern("waveone.aec_encode", OpPattern.OPAQUE)


# aec_range_encode_gaussian
@reg.register_schedule("waveone.aec_range_encode_gaussian")
def schedule_aec_range_encode_gaussian(attrs, outputs, target):
    with target:
        return topi.generic.schedule_aec_range_encode_gaussian(outputs)

@reg.register_compute("waveone.aec_range_encode_gaussian")
def compute_aec_range_encode_gaussian(attrs, inputs, out_dtype, target):
    with target:
        out = topi.waveone.aec_range_encode_gaussian(inputs[0], inputs[1], inputs[2], serialize=attrs.serialize)
        out = out if isinstance(out, list) else [out]
        return out

reg.register_pattern("waveone.aec_range_encode_gaussian", OpPattern.OPAQUE)


# aec_merge
@reg.register_schedule("waveone.aec_merge")
def schedule_aec_merge(attrs, outputs, target):
    with target:
        return topi.generic.schedule_aec_merge(outputs)

@reg.register_compute("waveone.aec_merge")
def compute_aec_merge(attrs, inputs, out_dtype, target):
    with target:
        out = topi.waveone.aec_merge(inputs)
        out = out if isinstance(out, list) else [out]
        return out

reg.register_pattern("waveone.aec_merge", OpPattern.OPAQUE)
