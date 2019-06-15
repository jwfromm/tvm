#include <tvm/relay/op.h>
#include <vector>
#include "woattrs.h"

namespace tvm {
namespace relay {

bool AecSplitRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  // types: {merged_code, merged_codelen, output}
  CHECK_EQ(types.size(), 3);
  const auto* merged_code = types[0].as<TensorTypeNode>();
  if (merged_code == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECSplit: Expect merged_code to be TensorType but get "
    << types[0];
    return false;
  }
  const auto* merged_codelen = types[1].as<TensorTypeNode>();
  if (merged_codelen == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
    << "AECSplit: Expect merged_codelen to be TensorType but get "
    << types[1];
    return false;
  }
  IndexExpr N = merged_code->shape[0];
  const AecSplitAttrs* param = attrs.as<AecSplitAttrs>();
  Array<IndexExpr> input_dims = param->input_dims;
  CHECK_EQ(input_dims.size(), 2);
  Array<IndexExpr> aec_params = param->aec_params;
  CHECK_EQ(aec_params.size() % 3, 0);

  Array<Type> output_types;
  IndexExpr H = input_dims[0];
  IndexExpr W = input_dims[1];
  int num_outputs = aec_params.size() / 3;
  for (int i = 0; i < num_outputs; ++i) {
    int aec_param_idx = 3 * i;
    IndexExpr ratio = aec_params[aec_param_idx];
    IndexExpr channels = aec_params[aec_param_idx + 1];
    IndexExpr bitplanes = aec_params[aec_param_idx + 2];
    Array<IndexExpr> output_shape({N, channels, H / ratio, W / ratio});
    output_types.push_back(TensorTypeNode::make(output_shape, UInt(8)));
    }
  TupleType output_tuple = TupleTypeNode::make(output_types);
  reporter->Assign(types[2], output_tuple);
  return true;
}

} // namespace relay
} // namespace tvm
