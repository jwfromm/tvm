import tvm

@tvm.target.generic_func
def aec_encode(bitplanes, aec_probs):
    """ Encodes incoming bit features.

    Parameters
    ----------
    bitplanes : tvm.Tensor
        Quantized and decomposed input features.

    aec_probs : tvm.Tensor
        Likelihood of each bit.

    Returns
    -------
    aec_encoded : tvm.Tensor
        Encoded values ready for transmission.

    aec_codelen : tvm.Tensor
        Length of encoded aec_encoded.
    """
    aec_encoded, aec_codelen = tvm.extern(
        [bitplanes.shape, [1]], [bitplanes, aec_probs],
        lambda ins, outs: tvm.call_packed('tvm.contrib.compression.aec_encode',
                                          ins[0], ins[1], outs[0], outs[1]),
        dtype=['uint8', 'int32'],
        name='aec_encode')

    return aec_encoded, aec_codelen
