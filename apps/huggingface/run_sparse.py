import os
import numpy as np
import tvm
import tensorflow as tf
import hf_extension
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import data_dep_optimization as ddo
import scipy.sparse as sp
import itertools

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import click

def _run_opt_pass(expr, opt_pass):
    """Helper function to run pass

    Parameters
    ----------
    expr : relay.Expr
        Expr will be optimized
    opt_pass : relay.Pass
        Optimization pass

    Returns
    -------
    ret: relay.Expr
        Optimized Expr by running opt_pass
    """
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    return mod["main"]

def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype="float32"):
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r+BS_R,c:c+BS_C] = np.random.uniform(-0.1, 0.1, (BS_R, BS_C))
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.data.size >= nnz
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s.todense()

def random_sparse_bert_params(func, params, density, BS_R=32, BS_C=1):
    def deepcopy(param_dic):
        ret = {}
        for k, v in param_dic.items():
            ret[k] = tvm.nd.array(v.asnumpy())
        return ret
    new_params = deepcopy(params)
    dense_weight_names = relay.analysis.sparse_dense._search_dense_op_weight(func)
    for item in dense_weight_names:
        name = str(item)
        shape = new_params[name].shape
        if shape[0] % BS_R == 0 and shape[1] % BS_C == 0:
            new_w = random_bsr_matrix(shape[0], shape[1], BS_R, BS_C, density)
            new_params[name] = tvm.nd.array(new_w)
    return new_params

def run_relay_graph(mod, params, shape_dict, target='llvm -mcpu=core-avx2', ctx=tvm.cpu()):
    # Build graph
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target=target, params=params)

    input_shape = shape_dict['input_1']
    rs = RandomState(MT19937(SeedSequence(123456789)))
    dummy_data = rs.randint(low=0, high=128, size=input_shape).astype('int32')

    m = graph_runtime.create(graph, lib, ctx)

    m.set_input(0, dummy_data)
    m.set_input(**params)
    m.run()

    tvm_output = m.get_output(0)

    ftimer = m.module.time_evaluator("run", ctx, min_repeat_ms=1000, repeat=3)
    prof_res = np.array(ftimer().results) * 1000
    print("%-20s %-19s (%s)" %
          ("bert", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
    return tvm_output

def import_graphdef(model_dir='frozen_models',
                    model_name='frozen_graph',
                    save_relay=False,
                    relay_file='model.txt',
                    param_file='model.params'):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, model_dir, model_name + '.pb')

    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Convert graph to relay
    shape_dict = {'input_1': (1, 128)}
    mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict)

    if save_relay:
        with open(os.path.join(abs_path, relay_file), 'w') as fo:
            fo.write(mod.astext())
        with open(os.path.join(abs_path, param_file), 'wb') as fo:
            fo.write(relay.save_param_dict(params))

    return mod, params, shape_dict

@click.command()
@click.option('--model_dir', default='frozen_models', required=True)
@click.option('--bs_r', default=16, type=int, required=True)
@click.option('--sparsity', default=0.85, type=float, required=True)
def run(model_dir, bs_r, sparsity):
    mod, params, shape_dict = import_graphdef(model_dir=model_dir)
    func = _run_opt_pass(mod["main"],
                      hf_extension.RemoveRedundantTrans())

    func = _run_opt_pass(func,
                      hf_extension.RemoveRedundantReshape())

    wt_expr, wt_params = ddo.simplify_fc_transpose.convert(func, params)

    bsr_params = random_sparse_bert_params(wt_expr, wt_params, BS_R=bs_r, BS_C=1, density=1-sparsity)
    print("Dense:")
    dense_output = run_relay_graph(wt_expr, bsr_params, shape_dict)

    print("BSR {sparsity}:".format(sparsity=sparsity))
    bsr_expr, bsr_params = ddo.bsr_dense.convert(wt_expr, bsr_params, (bs_r, 1), sparsity_threshold=0.8)
    sparse_output = run_relay_graph(bsr_expr, bsr_params, shape_dict)

    np.testing.assert_allclose(dense_output.asnumpy(), sparse_output.asnumpy(), atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    run()