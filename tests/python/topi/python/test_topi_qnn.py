# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test code for QNN operators."""
import numpy as np
import tvm
from tvm import topi, relay, te
from tvm.contrib import graph_runtime


def verify_simulated_quantize(
    data_shape, out_dtype, channels, axis
):
    # Create placeholder variables for all qnn inputs.
    A = te.placeholder(data_shape, name='value', dtype='float32')
    D = te.placeholder([1], name='dtype', dtype='int32')
    S = te.placeholder([te.size_var("scale_dim")], name="scale", dtype='float32')
    Z = te.placeholder([te.size_var("zp_dim")], name="zp", dtype='int32')
    SIM_Q = topi.nn.simulated_quantize(A, D, output_scale=S, output_zero_point=Z, axis=axis)

    # Create random numpy values to assign to inputs.
    a_np = np.random.uniform(size=data_shape).astype('float32')
    d_np = np.asarray([topi.nn.SQNN_DTYPE_TO_CODE[out_dtype]]).astype('int32')
    s_np = np.random.uniform(low=1e-4, high=.1, size=channels).astype('float32')
    z_np = np.random.uniform(low=-10, high=10, size=channels).astype('int32')
    q_np = np.zeros(shape=data_shape, dtype='float32')

    ctx = tvm.cpu()

    # Wrap the numpy arrays in nd arrays.
    a = tvm.nd.array(a_np, ctx)
    d = tvm.nd.array(d_np, ctx)
    s = tvm.nd.array(s_np, ctx)
    z = tvm.nd.array(z_np, ctx)
    q = tvm.nd.array(q_np, ctx)

    # Construct equivalent relay graph.
    per_channel = channels[0] != 1
    a_var = relay.var('a', shape=data_shape, dtype='float32')
    if per_channel:
        s_var = relay.const(s_np)
        z_var = relay.const(z_np)
    else:
        s_var = relay.const(s_np[0])
        z_var = relay.const(z_np[0])
    real_q_op = relay.qnn.op.quantize(a_var, s_var, z_var, axis=axis, out_dtype=out_dtype)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(tvm.IRModule.from_expr(real_q_op), target='llvm')

    # Get real qnn quantize output.
    m = graph_runtime.GraphModule(lib["default"](ctx))
    m.set_input('a', a_np)

    m.run()
    real_q_out = m.get_output(0)

    # Compile the simulated quantize function.
    sched = te.create_schedule([SIM_Q.op])
    func = tvm.build(sched, [A, D, S, Z, SIM_Q], 'llvm', name="sim_quantize")
    func(a, d, s, z, q)

    # Check correctness against the true qnn output.
    tvm.testing.assert_allclose(q.asnumpy(), real_q_out.asnumpy().astype('float32'))


def test_simulated_quantize():
    verify_simulated_quantize([1], 'int8', [1], -1)
    verify_simulated_quantize([2, 5], 'int8', [5], 1)
    verify_simulated_quantize([1, 32, 32, 32], 'int8', [32], -1)
    verify_simulated_quantize([1, 32, 32, 32], 'uint8', [32], -2)
    verify_simulated_quantize([2, 5], 'int32', [5], 1)

if __name__ == "__main__":
    test_simulated_quantize()

