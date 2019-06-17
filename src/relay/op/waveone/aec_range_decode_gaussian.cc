#include <tvm/relay/op.h>
#include "woattrs.h"

namespace tvm {
namespace relay {

bool AecRangeDecodeGaussianRel(const Array<Type>& types,
                               int num_inputs,
                               const Attrs& attrs,
                               const TypeReporter& reporter) {
  // types: {gauss_encoded, anorm, div_anorm, lookup, gauss_decoded}
  CHECK_EQ(types.size(), 5);
  const auto* gauss_encoded = types[0].as<TensorTypeNode>();
  if (gauss_encoded == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECRangeDecodeGaussian: Expect gauss_encoded to be TensorType but get "
    << types[0];
    return false;
  }
  const auto* anorm = types[1].as<TensorTypeNode>();
  if (anorm == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECRangeDecodeGaussian: Expect anorm to be TensorType but get "
    << types[1];
    return false;
  }
  const auto* div_anorm = types[2].as<TensorTypeNode>();
  if (div_anorm == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECRangeDecodeGaussian: Expect div_anorm to be TensorType but get "
    << types[2];
    return false;
  }
  const auto* lookup = types[3].as<TensorTypeNode>();
  if (lookup == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECRangeDecodeGaussian: Expect lookup to be TensorType but get "
    << types[3];
    return false;
  }
  reporter->Assign(types[4], TensorTypeNode::make(gauss_encoded->shape, Int(32)));
  return true;
}

Expr MakeAecRangeDecodeGaussian(Expr gauss_encoded,
                                Expr anorm,
                                Expr div_anorm,
                                Expr lookup,
                                bool serialize) {
  auto attrs = make_node<AecRangeDecodeGaussianAttrs>();
  attrs->serialize = serialize;
  static const Op& op = Op::Get("waveone.aec_range_decode_gaussian");
  return CallNode::make(op, {gauss_encoded, anorm, div_anorm, lookup}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.waveone._make.aec_range_decode_gaussian")
.set_body_typed(MakeAecRangeDecodeGaussian);

RELAY_REGISTER_OP("waveone.aec_range_decode_gaussian")
.describe(R"doc("Decodes a tensor of bits using a gaussian
likelihood distribution.)doc" TVM_ADD_FILELINE)
.set_num_inputs(4)
.add_argument("gauss_encoded", "Tensor", "Incoming encoded bitstream.")
.add_argument("anorm", "Tensor", "Inferred likelihoods.")
.add_argument("div_anorm", "tensor", "Normalized likelhoods.")
.add_argument("lookup", "tensor", "Lookup table for Gaussian CDF.")
.set_support_level(6)
.add_type_rel("AecRangeDecodeGaussian", AecRangeDecodeGaussianRel);


} // namespace relay
} // namespace tvm
