Step
----

0. build tvm

1. cd ```hf_extension```

2. run ```Make```

3. run ```python download_model.py --family bert --name bert-base-uncased```

4. run ```python run_sparse.py --bs_r 16 --sparsity 0.85```
