/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen.h
 * \brief Collection of Lowlevel IR pass to codegen.
 */
#ifndef TVM_CODEGEN_H_
#define TVM_CODEGEN_H_

#include <string>
#include "./base.h"
#include "./expr.h"
#include "./lowered_func.h"
#include "./api_registry.h"
#include "./runtime/packed_func.h"

namespace tvm {
/*! \brief namespace for lowlevel IR pass and codegen */
namespace codegen {
// use packed function from runtime.
using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

/*!
 * \brief Build a module from array of lowered function.
 * \param funcs The functions to be built.
 * \param target The target to be built.
 * \return The builded module.
 *
 * \note Calls global API function  "_codegen_build_" + target
 */
runtime::Module Build(const Array<LoweredFunc>& funcs,
                      const std::string& target);

/*!
 * \param target The target to be queried.
 * \return Whether target is enabled.
 */
bool TargetEnabled(const std::string& target);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_H_
