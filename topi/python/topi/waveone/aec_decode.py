import tvm

@tvm.target.generic_func
def aec_decode(encoded, feature_probs):
    """ Decodes an incoming bitstream.

    Parameters
    ----------
    encoded : tvm.Tensor
        The incoming bitstream.

    feature_probs : tvm.Tensor
        Likelihood of each bit.

    Returns
    -------
    aec_decoded : tvm.Tensor
        The decoded tensor.
    """
    aec_decoded = tvm.extern(
        encoded.shape, [encoded, feature_probs],
        lambda ins, outs: tvm.call_packed('tvm.contrib.compression.aec_decode',
                                          ins[0], ins[1], outs[0]),
        dtype='uint8',
        name='aec_decoded')

    return aec_decoded
