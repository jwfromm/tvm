#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

bool AecDecodeRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  //types {encoded, feature_probs, decoded}
  CHECK_EQ(types.size(), 3);
  const auto* encoded = types[0].as<TensorTypeNode>();
  if (encoded == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECDecode: Expect encoded to be TensorType bug get "
    << types[0];
    return false;
  }
  const auto* feature_probs = types[1].as<TensorTypeNode>();
  if (feature_probs == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
    << "AECDecode: Expect feature_probs to be TensorType bug get "
    << types[1];
    return false;
  }
  reporter->Assign(types[2], TensorTypeNode::make(encoded->shape, UInt(8)));
  return true;
}

Expr MakeAecDecode(Expr encoded,
                   Expr feature_probs) {
  static const Op& op = Op::Get("waveone.aec_decode");
  return CallNode::make(op, {encoded, feature_probs}, Attrs(), {});
}

TVM_REGISTER_API("relay.op.waveone._make.aec_decode")
.set_body_typed(MakeAecDecode);

RELAY_REGISTER_OP("waveone.aec_decode")
.describe(R"doc(Decodes a tensor into bits based on the likelihood of each positional value.)doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("encoded", "Tensor", "Encoded bitstream.")
.add_argument("feature_probs", "Tensor", "Prior probabilities of each pixel.")
.set_support_level(6)
.add_type_rel("AecDecode", AecDecodeRel);

} // namespace relay
} // namespace tvm
