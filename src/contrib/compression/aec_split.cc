#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>

#include "wolib/wo/aec/split.h"

using namespace wo;

namespace tvm {
namespace contrib {

using namespace runtime;

inline int num_elements(DLTensor* input) {
    int nelems = 1;
    for (int i = 0; i < input->ndim; ++i) {
        nelems *= input->shape[i];
    }

    return nelems;
}

void aec_split(TVMArgs args, TVMRetValue* rv) {
    NDArray test_input = args[0];
    // Does work :O
    //int test = static_cast<int32_t*>(test_input->data)[0];
    NDArray test_input2 = args[1];
    std::vector<int64_t> shape(test_input->shape, test_input->shape + test_input->ndim);
    NDArray test = NDArray::Empty(shape, test_input->dtype, test_input->ctx);
    // Can also use push_back on top of vector to add one at a time.
    std::vector<NDArray> outputs{test_input, test_input2};
    //DLTensor* merged_code_tensor = args[0];
    //DLTensor* merged_codelen_tensor = args[1];
    //DLTensor* input_dim_tensor = args[2];
    //DLTensor* aec_params_tensor = args[3];

    //aec_api::Split aecSplit(
    //    static_cast<uint8_t *>(merged_code_tensor->data), num_elements(merged_code_tensor),
    //    static_cast<int32_t *>(merged_codelen_tensor->data), num_elements(merged_codelen_tensor),
    //    static_cast<int32_t *>(input_dim_tensor->data), num_elements(input_dim_tensor),
    //    static_cast<int32_t *>(aec_params_tensor->data), num_elements(aec_params_tensor)
    //);

    //size_t num_aec = aecSplit.get_num_aec();

    //DLTensor *test;
    //int ndim = 1;
    //int dtype_code = kDLFloat;
    //int dtype_bits = 32;
    //int dtype_lanes = 1;
    //int device_type = kDLCPU;
    //int device_id = 0;
    //tvm_index_t shape[1] = {10};
    //TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    //              device_type, device_id, &test);
    auto get_array = [outputs] (TVMArgs args, TVMRetValue* rv) {
      int index = args[0];
      *rv = outputs[index];
    };

    *rv = PackedFunc(get_array);
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_split")
.set_body(aec_split);

} // namespace contrib
} // namespace tvm
