import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from transformers import *

import tvm
from tvm import relay
from tvm import autotvm
#from tvm.contrib.debugger import debug_runtime as graph_runtime
from tvm.contrib import graph_runtime


seq_len = 128
batch_size = 1
tvm_target = "llvm"
tvm_ctx = tvm.cpu(0)

# Import bert model from transformers
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
dummy_input = tf.keras.Input(shape=[seq_len], batch_size=batch_size, dtype='int32')
dummy_out = model(dummy_input)

# Freeze to TF concrete function
model_func = tf.function(lambda x: model(x))
model_func = model_func.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
frozen_func = convert_variables_to_constants_v2(model_func)

# Convert graph to relay
shape_dict = {'input_1': (batch_size, seq_len)}
mod, params = relay.frontend.from_tensorflow(frozen_func.graph.as_graph_def(), shape=shape_dict)

# Build graph
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target=tvm_target, params=params)

# Get debug runtime stats out of this bad boy
m = graph_runtime.create(graph, lib, tvm_ctx)
dummy_data = np.random.uniform(low=0, high=128, size=shape_dict['input_1']).astype('int32')
m.set_input(0, dummy_data)
m.set_input(**params)
m.run()

print("START")
compiler = relay.vm.VMCompiler()
compiler.set_params(params)
print("LOWER")
compiler.lower(mod, target=tvm_target)
print("STOP")

# Now onto the bug...
#tasks = autotvm.task.extract_from_program(mod["main"], target=tvm_target,
#                                              params=params,
#                                              ops=(relay.op.get("nn.conv2d"),
#                                                   relay.op.get("nn.dense")))
#print(tasks)