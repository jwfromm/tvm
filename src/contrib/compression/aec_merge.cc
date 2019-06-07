#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <algorithm>
#include <vector>

#include "wolib/wo/aec/merge.h"

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

void aec_merge(TVMArgs args, TVMRetValue* rv) {
  CHECK(args.num_args > 0);
  CHECK(args.num_args % 2 == 0);

  auto num_aec = (args.num_args - 2) / 2;

  auto N = ((DLTensor*)args[0])->shape[0];

  aec_api::Merge aecMerge(N, num_aec);

  int codelen_idx = num_aec;
  int i;
  for (i = 0; i < num_aec; ++i) {
    const DLTensor* code_tensor = args[i];
    const DLTensor* codelen_tensor = args[codelen_idx];
    ++codelen_idx;
    auto code_dims = code_tensor->ndim;

    CHECK((code_dims == 4) || (code_dims == 5));
    CHECK(N == code_tensor->shape[0]);

    // C, H, W, B
    uint16_t C = static_cast<uint16_t>(code_tensor->shape[1]);
    uint16_t H = static_cast<uint16_t>(code_tensor->shape[2]);
    uint16_t W = static_cast<uint16_t>(code_tensor->shape[3]);
    uint16_t B = (code_dims == 4) ? 1 : static_cast<uint16_t>(code_tensor->shape[4]);

    aecMerge.add_code_dims(C, H, W, B);
    aecMerge.set_code_input(static_cast<uint32_t>(i), static_cast<uint8_t*>(code_tensor->data),
                            static_cast<size_t>(num_elements(code_tensor)));

    auto codelen_data = static_cast<int32_t*>(codelen_tensor->data);
    for (int n = 0; n < N; ++n) {
      aecMerge.add_code_input_len(i, codelen_data[n]);
    }
  }
  DLTensor* merged_tensor = args[args.num_args - 2];
  DLTensor* merged_len_tensor = args[args.num_args - 1];

  aecMerge.generate_output(static_cast<uint8_t*>(merged_tensor->data), num_elements(merged_tensor));
  auto merged_len_data = static_cast<int32_t*>(merged_len_tensor->data);

  for (uint32_t n = 0; n < N; ++n) {
    merged_len_data[n] = aecMerge.get_output_codelen_size(n);
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_merge").set_body(aec_merge);

}  // namespace contrib
}  // namespace tvm
