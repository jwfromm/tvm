import os
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm.autotvm.tuner import XGBTuner
from TidalWave import get_tidalwave_model


# Load our tidal wave model and convert to the relay graph
def get_network():
    # Load the keras model
    model, shape_dict = get_tidalwave_model()

    # Run the model through the relay converter.
    mod, params = relay.frontend.from_keras(model, shape_dict)
    return mod, params 


# Define tuning parameters
# Target to compile to
target = "llvm"

# Measure option indicates that we should run on our local CPU.
tuning_option = {
    'log_filename': "tuning.log",
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=3, repeat=1,
                                   min_repeat_ms=10),
    ),
}


# Function for running a tuning jobs across tunable tasks
def tune_kernels(tasks,
                 measure_option,
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # Create an XGBoost tuner
        tuner_obj = XGBTuner(task, loss_type='rank', feature_type='knob')

        # Try out 128 different schedules and pick the best (normally you'd try many more)
        n_trial = 128
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])



def tune_and_evaluate(tuning_opt):
    mod, params = get_network()
    # extract workloads from relay program
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),
                                                   relay.op.get("nn.dense")))
    print(tasks)
    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)


if __name__ == "__main__":
    tune_and_evaluate(tuning_option)