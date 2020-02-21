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
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <memory>
#include <cstdio>
#include <iostream>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

extern unsigned char lib_graph_json[];
extern unsigned int lib_graph_json_len;
extern unsigned char lib_params_bin[];
extern unsigned int lib_params_bin_len;

void Verify(tvm::runtime::Module mod, std::string fname) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  CHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* x;
  DLTensor* y;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};
  std::cout << "Preallocate\n";
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &x);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &y);
  std::cout << "Postallocate\n";
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(x->data)[i] = i;
  }
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(x, y);
  std::cout << "Post Invoke\n";
  // Print out the output
  for (int i = 0; i < shape[0]; ++i) {
    CHECK_EQ(static_cast<float*>(y->data)[i], i + 1.0f);
  }
  std::cout << "Finish verification...\n";
}

void RunGraph(tvm::runtime::Module mod_syslib) {
  // Load graph json library.
  const std::string json_data(&lib_graph_json[0], &lib_graph_json[0] + lib_graph_json_len);
  std::cout << "Loaded graph JSON.\n";
  // Load parameter data
  TVMByteArray params;
  params.data = reinterpret_cast<const char *>(&lib_params_bin[0]);
  params.size = lib_params_bin_len;
  std::cout << "Loaded Params.\n";

  // Create graph runtime.
  int device_type = kDLCPU;
  int device_id = 0;
  tvm::runtime::Module mod =
    (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(
      json_data, mod_syslib, device_type, device_id
    );
  std::cout << "Graph Runtime Created.\n";
}

int main(void) {
  // For libraries that are directly packed as system lib and linked together with the app
  // We can directly use GetSystemLib to get the system wide library.
  std::cout << "Verify load function from system lib\n";
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  Verify(mod_syslib, "addonesys");
  RunGraph(mod_syslib);
  return 0;
}
