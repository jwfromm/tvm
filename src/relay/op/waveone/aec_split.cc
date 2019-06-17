#include <tvm/relay/op.h>
#include <vector>
#include "woattrs.h"

namespace tvm {
namespace relay {

bool AecSplitRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  // types: {merged_code, merged_codelen, input_dims, aec_params, output}
  CHECK_EQ(types.size(), 5);
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
  const auto* input_dims = types[2].as<TensorTypeNode>();
  if (input_dims == nullptr) {
    CHECK(types[2].as<IncompleteTypeNode>())
    << "AECSplit: Expect input_dims to be TensorType but get "
    << types[2];
    return false;
  }
  const auto* aec_params = types[3].as<TensorTypeNode>();
  if (aec_params == nullptr) {
    CHECK(types[3].as<IncompleteTypeNode>())
    << "AECSplit: Expect aec_params to be TensorType but get "
    << types[3];
    return false;
  }
  const AecSplitAttrs* param = attrs.as<AecSplitAttrs>();
  Array<Array<IndexExpr>> output_shapes = param->output_shapes;
  // Make sure each output shape has 5 dimensions.
  int num_aecs = output_shapes.size();

  Array<Type> output_types;
  for (uint16_t i = 0; i < num_aecs; ++i) {
    Array<IndexExpr> output_shape;
    for (uint16_t n = 0; n < output_shapes[i].size(); ++n) {
      output_shape.push_back(output_shapes[i][n]);
    }
    output_types.push_back(TensorTypeNode::make(output_shape, UInt(8)));
  }

  TupleType output_tuple = TupleTypeNode::make(output_types);
  reporter->Assign(types[4], output_tuple);
  return true;
}

Expr MakeAecSplit(Expr merged_code,
                  Expr merged_codelen,
                  Expr input_dims,
                  Expr aec_params,
                  Array<Array<IndexExpr>> output_shapes) {
  auto attrs = make_node<AecSplitAttrs>();
  attrs->output_shapes = output_shapes;
  static const Op& op = Op::Get("waveone.aec_split");
  return CallNode::make(op, {merged_code, merged_codelen, input_dims, aec_params}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.waveone._make.aec_split")
.set_body_typed(MakeAecSplit);

RELAY_REGISTER_OP("waveone.aec_split")
.describe(R"doc(Splits a merged code and codelen into components.)doc" TVM_ADD_FILELINE)
.set_num_inputs(4)
.add_argument("merged_code", "Tensor", "Incoming merged code.")
.add_argument("merged_codelen", "Tensor", "Length of merged code.")
.add_argument("input_dims", "Tensor", "Height and width of input image.")
.add_argument("aec_params", "Tensor", "Description of outbound AECs.")
.add_argument("output_shapes", "Array", "List of output shapes for each outbound AEC.")
.set_support_level(6)
.add_type_rel("AecSplit", AecSplitRel);

} // namespace relay
} // namespace tvm
