import tvm
import numpy as np

def get_aec_merge_output_size(aec_inputs):
    N = aec_inputs[0].shape[0]
    output_size = 0
    for aec in aec_inputs:
        output_size += np.prod(aec.shape[1:])
        output_size += 4
    return [N, output_size]

@tvm.target.generic_func
def aec_merge(data_tuple):
    """ Encodes incoming bit features.

    Parameters
    ----------
    data_tuple: Tuple of tvm.Tensor
        Set of incoming aec features followed by corresponding codelengths.

    Returns
    -------
    merged_code : tvm.Tensor
        Merged encoded values ready for transmission.

    merged_codelen : tvm.Tensor
        Length of merged code.
    """
    num_aec = int(len(data_tuple) / 2)
    aec_tuple = data_tuple[0:num_aec]
    codelen_tuple = data_tuple[num_aec:]
    merged_output_shape = get_aec_merge_output_size(aec_tuple)
    merged_code, merged_codelen = tvm.extern(
        [merged_output_shape, [1]], [*aec_tuple, *codelen_tuple],
        lambda ins, outs: tvm.call_packed('tvm.contrib.compression.aec_merge',
                                          *ins, outs[0], outs[1]),
        dtype=['uint8', 'int32'],
        name='aec_merge')
    return [merged_code, merged_codelen]
