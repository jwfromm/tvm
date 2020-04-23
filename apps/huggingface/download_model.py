import os
import numpy as np
import tensorflow as tf
import transformers
from transformers import *
import click

def load_keras_model(module, seq_len=128, batch_size=1, report_runtime=False):
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    model = module.from_pretrained('bert-base-cased')
    dummy_input = tf.keras.Input(shape=[seq_len], batch_size=batch_size, dtype='int32')
    dummy_out = model(dummy_input)
    if report_runtime:
        import time 
        np_input = np.random.uniform(size=[batch_size, seq_len], low=0, high=seq_len).astype('int32')
        start = time.time()
        repeats = 50
        for i in range(repeats):
            np_out = model(np_input)
        end = time.time()
        print("Keras Runtime: %f ms." % (1000 * ((end - start) / repeats)))
    return model


def convert_to_graphdef(model, model_dir='./frozen_models', model_name='frozen_graph'):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)
    # Try to convert model to concrete function
    model_func = tf.function(lambda x: model(x))
    model_func = model_func.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # freeze
    frozen_func = convert_variables_to_constants_v2(model_func)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=model_path,
        name="%s.pb" % model_name,
        as_text=False)


@click.command()
@click.option('--family', default='bert', required=True)
@click.option('--name', default='bert-base-uncased', required=True)
@click.option('--output', default='frozen_models')
def download_model(family, name, output):
    module = None
    if family == 'bert':
        module = getattr(transformers, "TFBertForSequenceClassification")
    else:
        raise NotImplementedError
    model = load_keras_model(module, report_runtime=False)
    convert_to_graphdef(model, model_dir=output)


if __name__ == '__main__':
    download_model()