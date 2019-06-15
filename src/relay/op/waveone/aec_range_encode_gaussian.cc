#include <tvm/relay/op.h>
#include "woattrs.h"

namespace tvm {
namespace relay {

bool AecRangeEncodeGaussianRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* quantized = types[0].as<TensorTypeNode>();
  const auto* anorm = types[1].as<TensorTypeNode>();
  const auto* lookup = types[2].as<TensorTypeNode>();
  if (quantized == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
    << "AECRangeEncodeGaussian: Expect quantized to be TensorType but get "
    << types[0];
    return false;
  }
  if (anorm == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
    << "AECRangeEncodeGaussian: Expect anorm to be TensorType but get "
    << types[1];
    return false;
  }
  if (lookup == nullptr) {
    CHECK(types[2].as<IncompleteTypeNode>())
    << "AECRangeEncodeGaussian: Expect lookup to be TensorType but get "
    << types[2];
    return false;
  }
  Array<IndexExpr> codelen_shape({1});
  auto gauss_encoded_ty = TensorTypeNode::make(quantized->shape, UInt(8));
  auto gauss_codelen_ty = TensorTypeNode::make(codelen_shape, Int(32));
  reporter->Assign(types[3], TupleTypeNode::make({gauss_encoded_ty, gauss_codelen_ty}));
  return true;
}

Expr MakeAecRangeEncodeGaussian(Expr quantized,
                                Expr anorm,
                                Expr lookup,
                                bool serialize) {
  auto attrs = make_node<AecRangeEncodeGaussianAttrs>();
  attrs->serialize = serialize;
  static const Op& op = Op::Get("waveone.aec_range_encode_gaussian");
  return CallNode::make(op, {quantized, anorm, lookup}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.waveone._make.aec_range_encode_gaussian")
.set_body_typed(MakeAecRangeEncodeGaussian);

RELAY_REGISTER_OP("waveone.aec_range_encode_gaussian")
.describe(R"doc(Encodes a tensor of bits using a gaussian
likelihood distribution.)doc" TVM_ADD_FILELINE)
.set_num_inputs(3)
.add_argument("quantized", "Tensor", "Incoming quantized values.")
.add_argument("anorm", "Tensor", "Normalized incoming features.")
.add_argument("lookup", "Tensor", "Lookup table of cdf values.")
.set_support_level(6)
.add_type_rel("AecRangeEncodeGaussian", AecRangeEncodeGaussianRel);

} // namespace relay
} // namespace tvm
