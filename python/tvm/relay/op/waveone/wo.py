from . import _make
from ...expr import TupleWrapper

def aec_get_probs(bitplanes, feature_probs):
    return _make.aec_get_probs(bitplanes, feature_probs)

def aec_encode(bitplanes, aec_probs):
    out = _make.aec_encode(bitplanes, aec_probs)
    return TupleWrapper(out, 2)

def aec_range_encode_gaussian(quantized, anorm, lookup, serialize=False):
    out = _make.aec_range_encode_gaussian(quantized, anorm, lookup, serialize)
    return TupleWrapper(out, 2)

def aec_merge(data_tuple):
    out = _make.aec_merge(data_tuple)
    return TupleWrapper(out, 2)
