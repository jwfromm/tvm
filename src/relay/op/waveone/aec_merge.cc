#include <tvm/relay/op.h>
#include <vector>

namespace tvm {
namespace relay {

bool AecMergeRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  // types: [data_tuple, merged_output]
  CHECK_EQ(types.size(), 2);
  const auto* data_tuple = types[0].as<TupleTypeNode>();

  if (data_tuple == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "aec_merge: expect aec type to be TupleType but get "
    << types[0];
    return false;
  }

  const auto& first = Downcast<TensorType>(data_tuple->fields[0]);
  IndexExpr N = first->shape[0];
  Array<IndexExpr> merged_code_shape({N});
  Array<IndexExpr> merged_codelen_shape({1});

  int num_aecs = data_tuple->fields.size() / 2;
  IndexExpr output_length = 0;

  for (int i = 0; i < num_aecs; ++i) {
    const auto& e = Downcast<TensorType>(data_tuple->fields[i]);
    IndexExpr num_elements = 1;
    for (uint32_t n = 1; n < e->shape.size(); ++n) {
      num_elements *= e->shape[n];
    }
    num_elements += 4;
    output_length += num_elements;
  }

  merged_code_shape.push_back(output_length);

  auto merged_code_ty = TensorTypeNode::make(merged_code_shape, UInt(8));
  auto merged_codelen_ty = TensorTypeNode::make(merged_codelen_shape, Int(32));
  reporter->Assign(types[1], TupleTypeNode::make({merged_code_ty, merged_codelen_ty}));
  return true;
}

Expr MakeAecMerge(Expr data_tuple) {
  static const Op& op = Op::Get("waveone.aec_merge");
  return CallNode::make(op, {data_tuple}, Attrs(), {});
}

TVM_REGISTER_API("relay.op.waveone._make.aec_merge")
.set_body_typed(MakeAecMerge);

RELAY_REGISTER_OP("waveone.aec_merge")
.describe(R"doc(Combines multiple encoded tensors and their codelengths.)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data_tuple", "TensorTuple", "AEC encoded tensors and codelengths to be merged.")
.set_support_level(6)
.add_type_rel("AecMerge", AecMergeRel);

} // namespace relay
} // namespace tvm
