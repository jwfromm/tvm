# HuggingFace TVM Demo

This application downloads a HuggingFace transformer model, imports it into Relay,
then benchmarks its performance on CPU or GPU for both dense and block sparse versions
of the model.

To get setup, follow the installation instructions [here](https://tvm.apache.org/docs/install/from_source.html).
Make sure to enable LLVM and CUDA in the build config so both platforms can be tested.

Once TVM is built and ready to go, we need to compile a few custom passes that are used
to optimize HF models. Do so by running `make` in the `hf_extension` directory.

Now you're ready to start benchmarking! You can try out the default configuration by
running

`python hf_demo.py`

This will benchmarking a `bert-base-uncased` model on CPU with `batch_size=1` and `seq_len=128`.

`hf_demo.py` supports many command line arguments. Here's a quick rundown of each.

* --family sets the type of transformer model that should be run, defaults to `bert`.
* --name sets the specific model to be downloaded and run, defaults to 'bert-base-uncased'.
* --batch_size sets the input batch size to be benchmarked, defaults to '1'.
* --seq_len sets the input sequence length, defaults to `128`.
* --platform determines what hardware to benchmark on, must be either `cpu` or `gpu`. Defaults to `cpu`.
* --extract_llvm_options is an optional argument that determines all features (such as AVX512) that are available on the host machine and compiles for them. Worth giving a shot to see if it improves performance over the default `llvm` CPU target.
* --run_sparse is a flag that benchmarks a block sparse version of the model when set. This only works when `platform = cpu`.
* --bs_r sets the block size for sparsity. Defaults to `16`.
* --sparsity determines the percentage of zero parameters to test. Defaults `0.85`. Higher sparsity will yield larger speedups.

# PruneBert Support
This demo now enables speedups on prepruned HuggingFace models out of the box! To run a pruned BERT model call the script as follows:
`python hf_demo.py --name huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad --run_sparse`
You should find that using TVM's sparse computation speeds up model execution by something like 2-4X. You can also try playing around
with the block size setting (`--bs_r`) as different block sizes may work well on different CPUs. Note that this is being run
with the **real bert weights**, not simulated sparse weights as in the unpruned models.