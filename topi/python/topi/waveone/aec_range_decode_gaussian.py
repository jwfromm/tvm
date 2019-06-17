import tvm


@tvm.target.generic_func
def aec_range_decode_gaussian(gauss_encoded, anorm, div_anorm, lookup, serialize=False):
    """ Decodes an incoming bitstream using gaussian likelihoods.

    Parameters
    ----------
    gauss_encoded: tvm.Tensor
        Encoded bitstream.

    anorm: tvm.Tensor
        Likelihood of each bit.

    div_anorm: tvm.Tensor
        Normalized likelihood of each bit.

    lookup: tvm.Tensor
        Lookup table for gaussian CDF values.

    serialize: bool
        Whether to serialize output or not.

    Returns
    -------
    gauss_decoded : tvm.Tensor
        Decoded features.
    """
    gauss_decoded = tvm.extern(
        gauss_encoded.shape, [gauss_encoded, anorm, div_anorm, lookup],
        lambda ins, outs: tvm.call_packed('tvm.contrib.compression.aec_range_decode_gaussian',
                                          ins[0], ins[1], ins[2], ins[3], serialize, outs[0]),
        dtype='int32',
        name='gauss_decoded')

    return gauss_decoded
