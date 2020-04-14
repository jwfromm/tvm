import tvm
import numpy as np
from tvm import relay
from tvm import autotvm
from TidalWave import get_tidalwave_model


# Load the keras model
model, shape_dict = get_tidalwave_model()

# Run the model through the relay converter.
mod, params = relay.frontend.from_keras(model, shape_dict)

# Now lets compile the model
# Use LLVM to compile for our cpu
target = 'llvm'
# Tell TVM which device to run on
ctx = tvm.cpu()
# Compile the operators in the graph.
with autotvm.apply_history_best('tuning.log'):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target=target, params=params)

# Create a graph runtime and run our model
#from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime
# Dummy numpy data that we'll use for benchmarking
test_data = np.random.normal(size=shape_dict['input_1']).astype('float32')
# Create a runtime from our compile library and function graph.
m = graph_runtime.create(graph, lib, ctx)
# Set input and parameters
m.set_input('input_1', test_data)
m.set_input(**params)
# Run the model
m.run()
tvm_output = m.get_output(0)