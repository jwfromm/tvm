import tensorflow as tf

class OctoGemm(tf.keras.layers.Layer):
    def __init__(self, units, legs):
        super(OctoGemm, self).__init__()
        self.units = units
        self.legs = legs
    
    def build(self, input_shape):
        self.weight = self.add_weight(
            'weight',
            shape=[input_shape[1], self.units]
        )
    
    def call(self, input):
        gemm_output = tf.linalg.matmul(input, self.weight)
        return gemm_output * self.legs

def get_tidalwave_model():
    TidalModel = tf.keras.Sequential()
    TidalModel.add(tf.keras.layers.InputLayer(input_shape=[32, 32, 3]))
    TidalModel.add(tf.keras.layers.Conv2D(64, 5, padding="SAME"))
    TidalModel.add(tf.keras.layers.MaxPool2D())
    TidalModel.add(tf.keras.layers.Conv2D(128, 3, padding="SAME"))
    TidalModel.add(tf.keras.layers.GlobalAveragePooling2D())
    TidalModel.add(OctoGemm(1000, 8))

    print(TidalModel.summary())
    
    return TidalModel, {'input_1': [1, 3, 32, 32]}

if __name__ == "__main__":
    get_tidalwave_model()