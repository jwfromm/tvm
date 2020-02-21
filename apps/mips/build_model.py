import os
import tvm
import pickle
from tvm import relay

def main():
    model_dir = os.path.abspath("model")

    mod = pickle.load(open(os.path.join(model_dir, "model.pkl"), "rb"))
    params = relay.load_param_dict(open(os.path.join(model_dir, "model.params"), "rb").read())

    target="llvm -target=mipsel-linux-gnu -mcpu=mips32 --system-lib"
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

if __name__ == '__main__':
    main()