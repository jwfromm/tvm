from .keras import _check_data_format, _get_pad_pair, _convert_activation
from .. import op as _op
from .. import expr as _expr

def _convert_quantconv2d(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    weightList = keras_layer.get_weights()
    kernel_h, kernel_w, in_channels, n_filters = weightList[0].shape
    if etab.data_layout == 'NCHW':
        weight = weightList[0].transpose([3, 2, 0, 1])
        kernel_layout = 'OIHW'
    else:
        weight = weightList[0]
        kernel_layout = 'HWIO'

    if isinstance(keras_layer.dilation_rate, (list, tuple)):
        dilation = [keras_layer.dilation_rate[0], keras_layer.dilation_rate[1]]
    else:
        dilation = [keras_layer.dilation_rate, keras_layer.dilation_rate]
    dilated_kernel_h = (kernel_h - 1) * dilation[0] + 1
    dilated_kernel_w = (kernel_w - 1) * dilation[1] + 1
    stride_h, stride_w = keras_layer.strides

    # Quantize weights using ste sign.
    weight = (weight > 0).astype('int8')
    weight = _op.cast(etab.new_const(weight), 'int16')

    params = {
        'weight': weight,
        'kernel_size': [kernel_h, kernel_w],
        'strides': [stride_h, stride_w],
        'padding': [0, 0],
        'activation_bits': 1,
        'weight_bits': 1,
        'out_dtype': 'int16',
        'pack_dtype': 'uint32',
        'unipolar': False,
        'kernel_layout': kernel_layout,
        'data_layout': etab.data_layout,
        'channels': n_filters 
    }

    if keras_layer.padding == 'same':
        in_h = keras_layer.input_shape[1]
        in_w = keras_layer.input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        if pad_t == pad_b and pad_l == pad_r:
            params['padding'] = (pad_t, pad_l)
        else:
            if etab.data_layout == 'NCHW':
                inexpr = _op.nn.pad(data=inexpr, pad_width=(
                    (0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))
            else:
                inexpr = _op.nn.pad(data=inexpr, pad_width=(
                    (0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)))

    # Quantize input.
    inexpr = _op.cast(_op.greater(inexpr, _expr.const(0, dtype='float32')), 'int8')
    out = _op.nn.bitserial_conv2d(data=inexpr, **params)

    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        if etab.data_layout == 'NCHW':
            out = _op.nn.bias_add(out, bias)
        else:
            out = _op.nn.bias_add(out, bias, axis=1)
    # apply activation if specified
    act_type = keras_layer.activation.__name__
    if act_type != 'linear' and keras_layer.use_act:
        out = _convert_activation(out, act_type, etab)
    return out


def get_larq_convert_map():
    _larq_convert_map = {
        "QuantConv2D": _convert_quantconv2d,
    }
    return _larq_convert_map