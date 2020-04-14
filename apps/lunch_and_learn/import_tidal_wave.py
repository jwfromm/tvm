import tvm
from tvm import relay
from TidalWave import get_tidalwave_model

# Load the keras model
model, shape_dict = get_tidalwave_model()

# Run the model through the relay converter.
mod, params = relay.frontend.from_keras(model, shape_dict)

print(mod['main'])