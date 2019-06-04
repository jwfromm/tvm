#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>

#include "wo/aec/split.h"

namespace tvm {
namespace contrib {
namespace wo {

using namespace runtime;

void aec_split(TVMArgs args, TVMRetValue* rv) {
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

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_split")
.set_body(aec_split);

} // namespace wo
} // namespace contrib
} // namespace tvm
