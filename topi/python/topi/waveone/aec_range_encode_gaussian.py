import tvm


@tvm.target.generic_func
def aec_range_encode_gaussian(quantized, anorm, lookup, serialize=False):
    """ Encodes incoming bit features using gaussian likelihoods.

    Parameters
    ----------
    quantized: tvm.Tensor
        Quantized and decomposed input features.

    anorm: tvm.Tensor
        Likelihood of each bit.

    lookup: tvm.Tensor
        Lookup table for gaussian CDF values.

    Serialize: bool
        Whether to serialize output or not.

    Returns
    -------
    gauss_encoded : tvm.Tensor
        Encoded values ready for transmission.

    gauss_codelen : tvm.Tensor
        Length of encoded aec_encoded.
    """

    gauss_encoded, gauss_codelen = tvm.extern(
        [quantized.shape, [1]], [quantized, anorm, lookup],
        lambda ins, outs: tvm.call_packed(
            'tvm.contrib.compression.aec_range_encode_gaussian', ins[0], ins[
                1], ins[2], serialize, outs[0], outs[1]),
        dtype=['uint8', 'int32'],
        name='aec_range_encode_gaussian')

    return [gauss_encoded, gauss_codelen]
