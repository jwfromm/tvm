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

void simple_add(TVMArgs args, TVMRetValue* rv) {
    // automatically convert arguments to proper type.
    DLTensor* a = args[0];
    DLTensor* b = args[1];
    DLTensor* c = args[2];

    int num_elements = 1;
    for (int i = 0; i < a->ndim; ++i) {
        num_elements *= a->shape[i];
    }

    for (int i = 0; i < num_elements; i++) {
        *(static_cast<float *>(c->data) + i) = *(static_cast<float *>(a->data) + i) + *(static_cast<float *>(b->data) + i);
    }
}

int num_elements(DLTensor* input) {
    int nelems = 1;
    for (int i = 0; i < input->ndim; ++i) {
        nelems *= input->shape[i];
    }

    return nelems;
}

void aec_split(TVMArgs args, TVMRetValue* rv) {
    DLTensor* merged_code_tensor = args[0];
    DLTensor* merged_codelen_tensor = args[1];
    DLTensor* input_dim_tensor = args[2];
    DLTensor* aec_params_tensor = args[3];

    aec_api::Split aecSplit(
        static_cast<uint8_t *>(merged_code_tensor->data), num_elements(merged_code_tensor),
        static_cast<int32_t *>(merged_codelen_tensor->data), num_elements(merged_codelen_tensor),
        static_cast<int32_t *>(input_dim_tensor->data), num_elements(input_dim_tensor),
        static_cast<int32_t *>(aec_params_tensor->data), num_elements(aec_params_tensor)
    );

    size_t num_aec = aecSplit.get_num_aec();

    DLTensor* test;
    int ndim = 1;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    tvm_index_t shape[1] = {10};
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &test);

    *rv = test;
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_split")
.set_body(aec_split);

} // namespace contrib
} // namespace tvm
