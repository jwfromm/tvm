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

void aec_range_encode_gaussian(TVMArgs args, TVMRetValue* rv) {
  // Inputs
  DLTensor* quantized = args[0];
  DLTensor* div_A_norm = args[1];
  DLTensor* lookup = args[2];
  bool serialize = args[3];
  // Outputs
  DLTensor* encoded = args[4];
  DLTensor* sizes = args[5];

  int N = quantized->shape[0], C = quantized->shape[1], H = quantized->shape[2],
      W = quantized->shape[3];
  const int encoded_maxsize = C * H * W;
  const size_t quantized_size = C * H * W;

  // Cast data tensors appropriately.
  const int32_t* quantized_data = static_cast<int32_t*>(quantized->data);
  const int32_t* div_A_norm_data = static_cast<int32_t*>(div_A_norm->data);
  uint8_t* encoded_data = static_cast<uint8_t*>(encoded->data);
  int32_t* sizes_data = static_cast<int32_t*>(sizes->data);

  GaussianLookupTables lookup_tables(static_cast<uint8_t*>(lookup->data));

  int block_buffer_size = quantized_size * 2;
  int zero_buffer_size = quantized_size;
  int workbuff_min_size = block_buffer_size + zero_buffer_size;
  std::vector<wo_block_t> work_buffer_;
  if ((int)work_buffer_.size() < workbuff_min_size) {
    work_buffer_.resize(workbuff_min_size);
  }
  wo_block_t *block_buffer = work_buffer_.data();
  wo_code_t *zero_prob_buffer = (wo_code_t*)(block_buffer + block_buffer_size);

  for (int i = 0; i < N; ++i) {
    if (serialize) {
      sizes_data[i] = encode_range_gaussian_serial(
          quantized_data, C, H * W, div_A_norm_data, block_buffer, block_buffer_size,
          zero_prob_buffer, zero_buffer_size, encoded_data, encoded_maxsize, &lookup_tables);
    } else {
      sizes_data[i] = encode_range_gaussian(quantized_data, C, H * W, div_A_norm_data, block_buffer,
                                            block_buffer_size, zero_prob_buffer, zero_buffer_size,
                                            encoded_data, encoded_maxsize, &lookup_tables);
    }

    CHECK(sizes_data[i] < encoded_maxsize);

    quantized_data += quantized_size;
    div_A_norm_data += quantized_size;
    encoded_data += encoded_maxsize;
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.compression.aec_range_encode_gaussian")
    .set_body(aec_range_encode_gaussian);

}  // namespace contrib
}  // namespace tvm
