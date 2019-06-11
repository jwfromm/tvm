#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

bool AecGetProbsRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
                    const TypeReporter& reporter) {
  // 'types' contains: [bitplanes, feature_probs, result]
  CHECK_EQ(types.size(), 3);
  const auto* bitplanes = types[0].as<TensorTypeNode>();
  const auto* feature_probs = types[1].as<TensorTypeNode>();
  if (bitplanes == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECGetProbs: expect bitplanes to be TensorType but get "
    << types[0];
    return false;
  }
  if (feature_probs == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
    << "AECGetProbs: expect feature_probs to be TensorType but get "
    << types[1];
    return false;
  }
  reporter->Assign(types[2], TensorTypeNode::make(bitplanes->shape, Int(32)));
  return true;
}

Expr MakeAecGetProbs(Expr bitplanes,
                     Expr feature_probs) {
  static const Op& op = Op::Get("aec_get_probs");
  return CallNode::make(op, {bitplanes, feature_probs}, Attrs(), {});
}

TVM_REGISTER_API("relay.op._make.aec_get_probs")
.set_body_typed(MakeAecGetProbs);

RELAY_REGISTER_OP("aec_get_probs")
.describe(R"doc(Returns the likelihood of each bit
given incoming bitplanes and a prior distribution.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("bitplanes", "Tensor", "Decomposed bit values.")
.set_support_level(6)
.add_type_rel("AecGetProbs", AecGetProbsRel);
}  // namespace relay
}
