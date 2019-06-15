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

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_WO_H_
