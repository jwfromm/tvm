import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity

from .. import generic, tag
from ..util import traverse_inline, get_const_tuple


@autotvm.register_topi_compute("dense_octogemm.x86")
def dense_octogemm(cfg, data, weight, bias=None, out_dtype=None):
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    # Create a reduction axis
    k = te.reduce_axis((0, in_dim), name='k')
    # Define the core multiplication
    matmul = te.compute((batch, out_dim), \
                        lambda i, j: te.sum(data[i, k].astype(out_dtype) * \
                                            weight[j, k].astype(out_dtype), axis=k), \
                        name='T_dense', tag='octo_dense')
    # Compute the number of operations this function requires and add it to the config.
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    cfg.add_flop(2 * M * N * K * K)
    return matmul


# Define a schedule for our octogemm function.
# The inputs are cfg: a config file that keeps track of autotvm parameters,
# s: a schedule object, and
# C: the output of the above dense_octogemm function.
def _schedule_dense_octogemm(cfg, s, C):
    # Extract compute axes.
    y, x = s[C].op.axis
    # Extract reduction axes.
    kk, = s[C].op.reduce_axis
    # Define two tiling parameters for each compute axis.
    # Using cfg.define_split tells autotvm that this is a tunable
    # option that indicates how large tiles should be.
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    # Tile M and N using autotvm selected values.
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    # Separate tiles into better memory order.
    s[C].reorder(yo, xo, yi, xi)
    # Fuse outer axes and parallelize the result.
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyo)
    # Unroll the reduction axis
    s[C].unroll(kk)
    # Vectorize innermost loop
    s[C].vectorize(xi)
    return s


# Register our schedule and define a helper that schedules the correct op.
@autotvm.register_topi_schedule("dense_octogemm.x86")
def schedule_dense_octogemm(cfg, outs):
    """Create the schedule for dense_octogemm"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'octo_dense' in op.tag:
            _schedule_dense_octogemm(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s