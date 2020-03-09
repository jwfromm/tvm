import os
import numpy as np

with open(os.path.join('lib', 'output.bin'), 'rb') as fi:
    byte_array = fi.read()

output_data = np.frombuffer(byte_array, dtype='float32')
output_data = np.reshape(output_data, [12,])
print(output_data)