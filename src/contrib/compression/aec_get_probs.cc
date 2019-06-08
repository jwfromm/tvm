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

void aec_get_probs(TVMArgs args, TVMRetValue* rv) {
  // Inputs
  const DLTensor* codelayers = args[0];
  const DLTensor* feature_probs = args[1];
  // Outputs
  DLTensor* probs = args[2];

  const uint8_t* codelayers_data = static_cast<uint8_t*>(codelayers->data);
  const aec_api::wo_prob_t* feature_probs_data =
      static_cast<aec_api::wo_prob_t*>(feature_probs->data);
  aec_api::wo_prob_t* probs_data = static_cast<aec_api::wo_prob_t*>(probs->data);

  int N = codelayers->shape[0], C = codelayers->shape[1], H = codelayers->shape[2],
      W = codelayers->shape[3], B = codelayers->shape[4];

  aec_api::Codelayer cl(C, B, H, W);

  for (int i = 0; i < N; ++i) {
    cl.set_data(codelayers_data);

    aec_api::get_code_probs(cl, feature_probs_data, probs_data);

    codelayers_data += cl.size();
    probs_data += cl.size();
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_get_probs").set_body(aec_get_probs);

}  // namespace contrib
}  // namespace tvm
