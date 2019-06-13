from . import _make

def aec_get_probs(bitplanes, feature_probs):
    return _make.aec_get_probs(bitplanes, feature_probs)

def aec_encode(bitplanes, aec_probs):
    return _make.aec_encode(bitplanes, aec_probs)
