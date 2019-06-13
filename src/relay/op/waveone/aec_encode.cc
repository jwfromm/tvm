#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

bool AecEncodeRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {

  CHECK_EQ(types.size(), 4);
  const auto* bitplanes = types[0].as<TensorTypeNode>();
  const auto* aec_probs = types[1].as<TensorTypeNode>();
  if (bitplanes == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECEncode: Expect bitplanes to be TensorType but get "
    << types[0];
    return false;
  }
  if (aec_probs == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECEncode: Expect aec_probs to be TensorType but get "
    << types[0];
    return false;
  }
  reporter->Assign(types[2], TensorTypeNode::make(bitplanes->shape, UInt(8)));
  Array<IndexExpr> codelen_shape({1});
  reporter->Assign(types[3], TensorTypeNode::make(codelen_shape, Int(32)));
  return true;
}

Expr MakeAecEncode(Expr bitplanes,
                   Expr aec_probs) {
  static const Op& op = Op::Get("waveone.aec_encode");
  return CallNode::make(op, {bitplanes, aec_probs}, Attrs(), {});
}

TVM_REGISTER_API("relay.op.waveone._make.aec_encode")
.set_body_typed(MakeAecEncode);

RELAY_REGISTER_OP("waveone.aec_encode")
.describe(R"doc(Encodes a tensor of bits based on the
likelihood of each positional value.)doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("bitplanes", "Tensor", "Decomposed bit values of input features.")
.add_argument("aec_probs", "Tensor", "Generated confidence of each bit.")
.set_support_level(6)
.add_type_rel("AecEncode", AecEncodeRel);

}  // namespace relay
}
