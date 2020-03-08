# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Makefile Example to deploy TVM modules.

#CXX=clang++ -static -target mipsel-linux-gnu -march=mips32 --sysroot=${Ingenic_sdk_path}/crossToolchain/mips-gcc472-glibc216-32bit/mips-linux-gnu/libc 
CXX = g++

TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++11 -O2\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-DTVM_NOTHREADS\

#PKG_CFLAGS += ${Ingenic_CFLAGS} ${Ingenic_INC} -stdlib=libc++

#PKG_LDFLAGS = ${Ingenic_LIBS}

PKG_LDFLAGS += -L${TVM_ROOT}/build -pthread -ldl -lc++ -lrt

model_dir := model

.PHONY: clean all

all: lib/cpp_deploy_packed

# Serialize img.bin file
lib/img.bin.cc: lib/img.bin
	xxd -i $^ > $@

# Serialize graph.json file
lib/graph.json.cc: lib/graph.json
	xxd -i $^ > $@

# Serialize params.bin file
lib/params.bin.cc: lib/params.bin
	xxd -i $^ > $@

# Compile relay model and export artifacts.
lib/model.o lib/graph.json lib/params.bin: build_model.py
	python $<

# The code library built by TVM
lib/test_addone_sys.o: prepare_test_libs.py
	@mkdir -p $(@D)
	python3 prepare_test_libs.py

# Deploy using pre-built libtvm_runtime.so
lib/cpp_deploy_packed: cpp_deploy.cc lib/test_addone_sys.o tvm_runtime_pack.cc lib/model.o lib/graph.json.cc lib/params.bin.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)
clean:
	rm -rf lib
