import tvm
import numpy as np

@tvm.target.generic_func
def aec_split(merged_code, merged_codelen, input_dims, aec_params, output_shapes):
    """ Splits the merged code from multiple AECs.

    Parameters
    ----------
    merged_code : tvm.Tensor
        Merged AEC output.

    merged_codelen : tvm.Tensor
        Length of merged code.

    input_dims : tvm.Tensor
        Height and width of the current input.

    aec_params : tvm.Tensor
        Information about outbound AECs.

    output_shapes : List of lists
        Output shape for each outbound AEC.

    Returns
    -------
    split_codes : Tuple of tvm.Tensor
        Split AEC codes with one entry per outbound AEC.
    """
    split_codes = tvm.extern(
        [*output_shapes], [merged_code, merged_codelen, input_dims, aec_params],
        lambda ins, outs: tvm.call_packed('tvm.contrib.compression.aec_split',
                                          ins[0], ins[1], ins[2], ins[3], *outs),
        dtype=['uint8'] * len(output_shapes),
        name='aec_split')

    return split_codes
