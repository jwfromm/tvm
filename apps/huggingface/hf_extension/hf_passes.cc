/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file convert_sparse_dense.cc
 *
 * \brief Mutate dense operator to sparse dense operator
 */
#include <tvm/ir/expr.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {

class RemoveRedundantTransMutator : public ExprRewriter {
 public:
  RemoveRedundantTransMutator()
      : trans_op_(Op::Get("transpose")), reshape_op_(Op::Get("reshape")) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (pre->op == reshape_op_) {
      const auto trans_op = pre->args[0].as<CallNode>();
      if (trans_op) {
        const auto arg = trans_op->args[0].as<VarNode>();
        if (arg) {
          const auto attrs = trans_op->attrs.as<TransposeAttrs>();
          if (attrs->axes.size() == 2 && attrs->axes[0] == 0 && attrs->axes[1] == 1) {
            // return new reshape
            const auto data = post.as<CallNode>()->args[0].as<CallNode>()->args[0];
            return Call(reshape_op_, {data}, pre->attrs);
          }
        }
      }
    }
    return post;
  }

 private:
  const Op& trans_op_;
  const Op& reshape_op_;

};  // class RemoveRedundantTransMutator


Expr RemoveRedundantTrans(const Expr& e) {
  auto rewriter = RemoveRedundantTransMutator();
  return PostOrderRewrite(e, &rewriter);
}


class RemoveRedundantReshapeMutator : public ExprRewriter {
 public:
  RemoveRedundantReshapeMutator()
      : trans_op_(Op::Get("transpose")), reshape_op_(Op::Get("reshape")) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (pre->op == trans_op_) {
      const auto reshape_op = pre->args[0].as<CallNode>();
      if (reshape_op) {
        const auto arg = reshape_op->args[0].as<VarNode>();
        if (arg) {
          const auto attrs = reshape_op->attrs.as<ReshapeAttrs>();
          const auto ty = reshape_op->checked_type_.as<TensorTypeNode>();
          if (attrs->newshape.value().size() == ty->shape.size()) {
            bool equal = true;
            for (size_t i = 0; i < attrs->newshape.value().size(); ++i) {
              if (ty->shape[i].as<IntImmNode>()->value != attrs->newshape.value()[i].as<IntImmNode>()->value) {
                equal = false;
                break;
              }
            }
            if (equal) {
              const auto data = post.as<CallNode>()->args[0].as<CallNode>()->args[0];
              return Call(trans_op_, {data}, pre->attrs);
            }

          }
        }
      }
    }
    return post;
  }

 private:
  const Op& trans_op_;
  const Op& reshape_op_;

};  // class RemoveRedundantTransMutator


Expr RemoveRedundantReshape(const Expr& e) {
  auto rewriter = RemoveRedundantReshapeMutator();
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass RemoveRedundantTrans() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // Remove FreeVar warnings
        auto f0 = Downcast<Function>(RemoveRedundantTrans(f));
        Array<Var> sparse_params = FreeVars(f0);
        auto f1 = Function(sparse_params, f0->body, f0->ret_type, f0->type_params, f0->attrs);
        Array<Var> params = FreeVars(f1);
        for (const auto& var : sparse_params) {
          params.push_back(var);
        }
        return Function(params, f1->body, f1->ret_type, f1->type_params, f1->attrs);
      };
  return CreateFunctionPass(pass_func, 4, "RemoveRedundantTrans", {"InferType"});
}

TVM_REGISTER_GLOBAL("hf_extension.RemoveRedundantTrans").set_body_typed(RemoveRedundantTrans);



Pass RemoveRedundantReshape() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // Remove FreeVar warnings
        auto f0 = Downcast<Function>(RemoveRedundantReshape(f));
        Array<Var> sparse_params = FreeVars(f0);
        auto f1 = Function(sparse_params, f0->body, f0->ret_type, f0->type_params, f0->attrs);
        Array<Var> params = FreeVars(f1);
        for (const auto& var : sparse_params) {
          params.push_back(var);
        }
        return Function(params, f1->body, f1->ret_type, f1->type_params, f1->attrs);
      };
  return CreateFunctionPass(pass_func, 4, "RemoveRedundantReshape", {"InferType"});
}

TVM_REGISTER_GLOBAL("hf_extension.RemoveRedundantReshape").set_body_typed(RemoveRedundantReshape);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
