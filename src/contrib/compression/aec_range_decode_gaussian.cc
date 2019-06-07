#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <algorithm>
#include <vector>

#include "wolib/wo/aec/aec_gaussian.h"
#include "wolib/wo/aec/package.h"

using namespace wo;
using namespace aec_api;
using namespace aec::gaussian;

namespace tvm {
namespace contrib {

using namespace runtime;

void aec_range_decode_gaussian(TVMArgs args, TVMRetValue* rv) {
  // Inputs
  const DLTensor* encoded = args[0];
  const DLTensor* A_norm = args[1];
  const DLTensor* div_A_norm = args[2];
  const DLTensor* lookup = args[3];
  const bool serialize = args[4];
  // Outputs
  DLTensor *decoded = args[5];

  const int32_t* A_norm_data = static_cast<int32_t*>(A_norm->data);
  const int32_t* div_A_norm_data = static_cast<int32_t*>(div_A_norm->data);

  int N = decoded->shape[0], C = decoded->shape[1], H = decoded->shape[2], W = decoded->shape[3];

  GaussianLookupTables lookup_tables(static_cast<uint8_t*>(lookup->data));

  const uint8_t* encoded_data = static_cast<uint8_t*>(encoded->data);

  const int encoded_maxsize = C * H * W;

  int32_t* decoded_data = static_cast<int32_t*>(decoded->data);
  const int decoded_size = C * H * W;

  for (int i = 0; i < N; ++i) {
      if (serialize) {
          decode_range_gaussian_serial(encoded_data, encoded_maxsize,
              decoded_data, C, H * W, A_norm_data, div_A_norm_data, &lookup_tables);
      } else {
          decode_range_gaussian(encoded_data, encoded_maxsize, decoded_data,
              C, H * W, A_norm_data, div_A_norm_data, &lookup_tables);
      }

      A_norm_data += decoded_size;
      div_A_norm_data += decoded_size;
      decoded_data += decoded_size;
      encoded_data += encoded_maxsize;
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_range_decode_gaussian")
    .set_body(aec_range_decode_gaussian);

}  // namespace contrib
}  // namespace tvm
