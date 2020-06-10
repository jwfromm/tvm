import tvm
from tvm import relay
import os
import numpy as np
import transformers
import onnx
import tensorflow as tf

#model = onnx.load("/home/jwfromm/octoml/tvm/apps/huggingface/onnx/prunebert/bert.onnx")
#
#input_dict = {
#    'input_ids': [1, 128],
#    'attention_mask': [1, 128],
#    'token_type_ids': [1, 128]
#}
#
#mod, params = relay.frontend.from_onnx(model, input_dict)
#
#with open("prunebert.json", "w") as fo:
#    fo.write(tvm.ir.save_json(mod))
#with open("prunebert.params", "wb") as fo:
#    fo.write(relay.save_param_dict(params))

model = transformers.TFBertForSequenceClassification.from_pretrained('huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad')
input_dict = model._saved_model_inputs_spec
input_0 = list(input_dict.keys())[0]
print(input_dict[input_0])
print(dir(input_dict[input_0]))
#dummy_input = np.random.uniform(
#              size=[1, 128], low=0, high=128).astype('int32')
#model._set_inputs(dummy_input)
#import keras2onnx
#output_model_path = "bert-base-cased.onnx"
#onnx_model = keras2onnx.convert_keras(model, model.name)
#keras2onnx.save_model(onnx_model, output_model_path)

#model = onnx.load('bert-base-cased.onnx')
#shape_dict = {'input_1': [1, 128]}
#mod, params = relay.frontend.from_onnx(model, shape_dict)