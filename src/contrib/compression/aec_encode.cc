#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <algorithm>
#include <vector>

#include "wolib/wo/aec/aec_api.h"

using namespace wo;

namespace tvm {
namespace contrib {

using namespace runtime;

void aec_encode(TVMArgs args, TVMRetValue* rv) {
  // Inputs
  DLTensor *codelayers = args[0];
  DLTensor *probs = args[1];
  // Outputs
  DLTensor *encoded = args[2];
  DLTensor *sizes = args[3];

  int N = codelayers->shape[0], C = codelayers->shape[1], H = codelayers->shape[2],
      W = codelayers->shape[3], B = codelayers->shape[4];

  // Cast data tensors appropriately.
  const uint8_t* codelayers_data = static_cast<uint8_t*>(codelayers->data);
  const aec_api::wo_prob_t* probs_data = static_cast<aec_api::wo_prob_t*>(probs->data);
  uint8_t* encoded_data = static_cast<uint8_t*>(encoded->data);
  int32_t* sizes_data = static_cast<int32_t*>(sizes->data);

  aec_api::Codelayer cl(C, B, H, W);

  std::vector<aec_api::wo_block_t> work_buffer_;
  const int encoded_maxsize = C * H * W * B;
  uint32_t workbuff_min_size = cl.size() * 2;
  if (work_buffer_.size() < workbuff_min_size) {
    work_buffer_.resize(workbuff_min_size);
  }

  for (int i = 0; i < N; ++i) {
    cl.set_data(codelayers_data);

    sizes_data[i] = aec_api::encode(cl, probs_data, work_buffer_.data(), work_buffer_.size(),
                                    encoded_data, encoded_maxsize);

    CHECK(sizes_data[i] < encoded_maxsize); // running out of padding

    codelayers_data += cl.size();
    probs_data += cl.size();
    encoded_data += encoded_maxsize;
  }
}


TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_encode").set_body(aec_encode);

}  // namespace contrib
}  // namespace tvm
