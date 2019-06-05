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

void get_aec_probs_mask(TVMArgs args, TVMRetValue* rv) {
  DLTensor* codelayers = args[0];
  DLTensor* feature_probs = args[1];
  DLTensor* mask = args[2];
  DLTensor* probs = args[3];

  // Allocate output tensor.
  const uint8_t* codelayers_data = static_cast<uint8_t*>(codelayers->data);
  const aec_api::wo_prob_t* feature_probs_data =
      static_cast<aec_api::wo_prob_t*>(feature_probs->data);
  const uint8_t* mask_data = static_cast<uint8_t*>(mask->data);

  aec_api::wo_prob_t* probs_data = static_cast<aec_api::wo_prob_t*>(probs->data);

  int N = codelayers->shape[0], C = codelayers->shape[1], H = codelayers->shape[2],
      W = codelayers->shape[3], B = codelayers->shape[4];

  aec_api::Codelayer cl(C, B, H, W);

  for (int i = 0; i < N; ++i) {
    cl.set_data(codelayers_data);
    const size_t codelayer_size = cl.size();
    aec_api::get_code_probs(cl, feature_probs_data, probs_data, mask_data);
    codelayers_data += codelayer_size;
    probs_data += codelayer_size;
    mask_data += codelayer_size;
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.get_aec_probs_mask").set_body(get_aec_probs_mask);

}  // namespace contrib
}  // namespace tvm
