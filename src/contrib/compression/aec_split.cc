#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <algorithm>
#include <vector>

#include "wolib/wo/aec/split.h"

using namespace wo;

namespace tvm {
namespace contrib {

using namespace runtime;

inline int num_elements(const DLTensor* input) {
  int nelems = 1;
  for (int i = 0; i < input->ndim; ++i) {
    nelems *= input->shape[i];
  }

  return nelems;
}

void aec_split(TVMArgs args, TVMRetValue* rv) {
  // Inputs
  const DLTensor* merged_code_tensor = args[0];
  const DLTensor* merged_codelen_tensor = args[1];
  const DLTensor* input_dim_tensor = args[2];
  const DLTensor* aec_params_tensor = args[3];
  // Extract and cast output arguments.
  int num_aec = args.num_args - 4;
  std::vector<DLTensor*> outputs;
  for (int i = 0; i < num_aec; ++i) {
    outputs.push_back((DLTensor*)args[i + 4]);
  }

  aec_api::Split aecSplit(
      static_cast<uint8_t*>(merged_code_tensor->data), num_elements(merged_code_tensor),
      static_cast<int32_t*>(merged_codelen_tensor->data), num_elements(merged_codelen_tensor),
      static_cast<int32_t*>(input_dim_tensor->data), num_elements(input_dim_tensor),
      static_cast<int32_t*>(aec_params_tensor->data), num_elements(aec_params_tensor));

  for (int i = 0; i < num_aec; ++i) {
    aecSplit.set_output_code_buffer(i, static_cast<uint8_t*>(outputs[i]->data),
                                    num_elements(outputs[i]));
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_split").set_body(aec_split);

}  // namespace contrib
}  // namespace tvm
