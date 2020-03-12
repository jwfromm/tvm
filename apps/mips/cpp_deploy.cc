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
#include <random>
#include <chrono>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

extern unsigned char lib_graph_json[];
extern unsigned int lib_graph_json_len;
extern unsigned char lib_params_bin[];
extern unsigned int lib_params_bin_len;
extern unsigned char lib_image_bin[];
extern unsigned int lib_image_bin_len;


tvm::runtime::Module PrepareRuntime(tvm::runtime::Module mod_syslib) {
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

  // Load parameters into the runtime.
  mod.GetFunction("load_params")(params);
  std::cout << "Graph Runtime Created.\n";
  return mod;
}

void RunGraph(tvm::runtime::Module mod) {
  // Input HxW = 352 X 608 (3 channels)
  //std::vector<int64_t> input_shape = {1, 3, 352, 608};
  //std::vector<float> input_storage(1 * 3 * 352 * 608);
  // Input HxW = 192 X 320 (1 channels)
  //std::vector<int64_t> input_shape = {1, 1, 192, 320};
  //std::vector<float> input_storage(1 * 1 * 192 * 320);
  // Input HxW = 96 X 160 (1 channel)
  std::vector<int64_t> input_shape = {1, 1, 96, 160};
  std::vector<float> input_storage(1 * 1 * 96 * 160);

  // Initialize and load input image.
  FILE * in_fp = fopen("lib/img.bin", "rb");
  size_t in_sz = fread(input_storage.data(), 1*1*96*160, 4, in_fp);
  //size_t in_sz = fread(input_storage.data(), 1*1*192*320, 4, in_fp);
  //size_t in_sz = fread(input_storage.data(), 1*3*352*608, 4, in_fp);
  fclose(in_fp);

  // Assign random numbers to each input value
  //std::mt19937 gen(0);
  //for (auto &e : input_storage) {
  //  e = std::uniform_real_distribution<float>(0.0, 1.0)(gen);
  //}


  DLTensor input;
  input.data = input_storage.data();
  input.ctx = DLContext{kDLCPU, 0};
  input.ndim = 4;
  input.dtype = DLDataType{kDLFloat, 32, 1};
  input.shape = input_shape.data();
  input.strides = nullptr;
  input.byte_offset = 0;
  
  // Output shapes for input HxW = 352 X 608 (3 channel)
  //std::vector<int64_t> output_shape = {1, 24, 44, 76};
  //std::vector<float> output_storage(1 * 24 * 44 * 76);
  // Output shapes for input HxW = 192 X 320 (1 channel)
  //std::vector<int64_t> output_shape = {1, 18, 12, 20};
  //std::vector<float> output_storage(1 * 18 * 12 * 20);
  // Output shapes for input HxW = 96 X 160 (1 channel)
  std::vector<int64_t> output_shape = {1, 18, 6, 10};
  std::vector<float> output_storage(1 * 18 * 6 * 10);

  DLTensor output;
  output.data = output_storage.data();
  output.ctx = DLContext{kDLCPU, 0};
  output.ndim = 4;
  output.dtype = DLDataType{kDLFloat, 32, 1};
  output.shape = output_shape.data();
  output.strides = nullptr;
  output.byte_offset = 0;

  // Time model run.
  auto start = std::chrono::high_resolution_clock::now();
  // Assign Input
  mod.GetFunction("set_input")("data", &input);
  std::cout << "Input loaded\n";

  // Perform inference.
  mod.GetFunction("run")();
  std::cout << "Inference Complete\n";
  // Get the output.
  mod.GetFunction("get_output")(0, &output);
  std::cout << "Output Extracted\n";

  // Write output to file
  //FILE * out_fp = fopen("lib/output.bin", "wb");
  //size_t out_sz = fwrite(output_storage.data(), 1 * 24 * 44 * 76, 4, out_fp);
  //size_t out_sz = fwrite(output_storage.data(), 1 * 18 * 12 * 20, 4, out_fp);
  //fclose(out_fp);

  // Calculating total time taken by the program. 
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  std::cout << "Time taken by program is : " << microseconds;
  std::cout << " us\n";

  // Now loop and measure a few runs.
  //time_t start, end;
  //std::time(&start);
  //for (int i = 0; i < 5; i++) {
  //  mod.GetFunction("run")();
  //  mod.GetFunction("get_output")(0, &output);
  //}
  //std::time(&end); 
  //// Calculating total time taken by the program. 
  //float time_taken = float(end - start); 
  //std::cout << "Time taken by program is : " << time_taken;
  //std::cout << " sec\n";
}

int main(void) {
  // For libraries that are directly packed as system lib and linked together with the app
  // We can directly use GetSystemLib to get the system wide library.
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  tvm::runtime::Module mod = PrepareRuntime(mod_syslib);
  // Perform first run to initialize.
  RunGraph(mod);
  return 0;
}
