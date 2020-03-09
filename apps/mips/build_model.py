import os
import tvm
import pickle
import numpy as np
from tvm import relay
from PIL import Image

def main():
    model_dir = os.path.abspath("model")

    mod = relay.fromtext(open(os.path.join(model_dir, "model.txt"), "r").read())
    params = relay.load_param_dict(open(os.path.join(model_dir, "model.params"), "rb").read())

    #target="llvm -target=mipsel-linux-gnu -mcpu=mips32 --system-lib"
    target="llvm --system-lib"
    with relay.build_config(opt_level=3):
        graph, lib, params=relay.build(mod, target=target, params=params)    

    build_dir = os.path.abspath("lib")
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    lib.save(os.path.join(build_dir, 'model.o'))
    with open(os.path.join(build_dir, 'graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(build_dir, 'params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))

    # finally prepare an input image.
    data_dir = os.path.abspath("data")
    image_path = os.path.join(data_dir, "person.jpg")
    image = Image.open(image_path)
    # Convert to grayscale
    #image = image.convert("L")
    # Resize to WxH = 320x192
    #image = image.resize((320, 192))
    image = image.resize((608, 352))

    def transform_image(image):
        #image = np.expand_dims(np.array(image), axis=-1)
        image = np.array(image)
        image = image / 255
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(np.array(image), axis=0)
        return image

    x = transform_image(image)
    np.save(os.path.join(data_dir, "img.npy"), image)
    with open(os.path.join(build_dir, "img.bin"), "wb") as f_image:
        f_image.write(x.astype(np.float32).tobytes())


if __name__ == '__main__':
    main()