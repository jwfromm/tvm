import tvm

@tvm.target.generic_func
def aec_get_probs(bitplanes, feature_probs):
    """Computes likelihood for each bit, needed for AEC encoding.

    Parameters
    ----------
    bitplanes : tvm.Tensor
        Quantized and decomposed inputs.

    feature_probs : tvm.Tensor
        Prior distribution of features.

    Returns
    -------
    aec_probs : tvm.Tensor
        Computed probabilities for each input bit.
    """
    aec_probs = tvm.extern(
        bitplanes.shape, [bitplanes, feature_probs],
        lambda ins, outs: tvm.call_packed('tvm.contrib.compression.aec_get_probs',
                                          ins[0], ins[1], outs[0]),
        dtype='int32',
        name='aec_probs')

    return aec_probs
