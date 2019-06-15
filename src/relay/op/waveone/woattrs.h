#ifndef TVM_RELAY_ATTRS_WO_H_
#define TVM_RELAY_ATTRS_WO_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Add range encode gaussian attributes. */
struct AecRangeEncodeGaussianAttrs : public tvm::AttrsNode<AecRangeEncodeGaussianAttrs> {
  bool serialize;

  TVM_DECLARE_ATTRS(AecRangeEncodeGaussianAttrs, "relay.attrs.AecRangeEncodeGaussianAttrs") {
    TVM_ATTR_FIELD(serialize)
      .set_default(false)
      .describe("Whether output should be serialized or not.");
  }
};

/*! \brief Add aec split attributes. */
struct AecSplitAttrs : public tvm::AttrsNode<AecRangeEncodeGaussianAttrs> {
  Array<IndexExpr> input_dims;
  Array<IndexExpr> aec_params;

  TVM_DECLARE_ATTRS(AecSplitAttrs, "relay.attrs.AecSplitAttrs") {
    TVM_ATTR_FIELD(input_dims)
      .set_default(nullptr)
      .describe("Height and width of input tensor.");
    TVM_ATTR_FIELD(aec_params)
      .set_default(nullptr)
      .describe("Information about codelayers.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_WO_H_
