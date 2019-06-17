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
struct AecSplitAttrs : public tvm::AttrsNode<AecSplitAttrs> {
  Array<Array<IndexExpr>> output_shapes;

  TVM_DECLARE_ATTRS(AecSplitAttrs, "relay.attrs.AecSplitAttrs") {
    TVM_ATTR_FIELD(output_shapes)
      .set_default(NullValue<Array<Array<IndexExpr> > >())
      .describe("List of output shape for each outbound AEC.");
  }
};

/*! \brief Add range encode gaussian attributes. */
struct AecRangeDecodeGaussianAttrs : public tvm::AttrsNode<AecRangeDecodeGaussianAttrs> {
  bool serialize;

  TVM_DECLARE_ATTRS(AecRangeDecodeGaussianAttrs, "relay.attrs.AecRangeDecodeGaussianAttrs") {
    TVM_ATTR_FIELD(serialize)
      .set_default(false)
      .describe("Whether output should be serialized or not.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_WO_H_
