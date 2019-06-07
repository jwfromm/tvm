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

void aec_decode(TVMArgs args, TVMRetValue* rv) {
  // Inputs
  const DLTensor *encoded = args[0];
  const DLTensor *probs = args[1];
  // Outputs
  DLTensor *codelayers = args[2];
  
  const aec_api::wo_prob_t *probs_data = static_cast<aec_api::wo_prob_t*>(probs->data);
  const uint8_t *encoded_data = static_cast<uint8_t*>(encoded->data);
  const uint8_t *codelayers_data = static_cast<uint8_t*>(codelayers->data);

  int N = codelayers->shape[0], C = codelayers->shape[1], H = codelayers->shape[2],
      W = codelayers->shape[3], B = codelayers->shape[4];

  const int encoded_maxsize = C * H * W * B;

  aec_api::Codelayer cl(C, B, H, W);

  for (int i = 0; i < N; ++i) {
    cl.set_data(codelayers_data);

    aec_api::decode(encoded_data, encoded_maxsize, cl, probs_data);

    codelayers_data += cl.size();
    encoded_data += encoded_maxsize;
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_decode").set_body(aec_decode);

}  // namespace contrib
}
