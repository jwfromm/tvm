#-------------------------------------------------------------------------------
#  Template configuration for compiling
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  ./build.sh
#
#-------------------------------------------------------------------------------
APP_ABI = arm64-v8a

APP_PLATFORM = android-24

# whether enable OpenCL during compile
USE_OPENCL = 0

USE_VULKAN = 0

USE_SORT = 0

USE_DEBUG_RUNTIME = 1

# the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
ifeq ($(USE_OPENCL), 1)
    ADD_C_INCLUDES += $(HOME)/Android/adrenosdk/Development/Inc
    ADD_C_INCLUDES += $(HOME)/OpenCL-Headers/

    # the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
    ADD_LDLIBS += $(HOME)/Android/galaxy/libOpenCL.so
endif

# Include directories for waveone API
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/wo/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/wo/framework/default/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/wo/backend/armv8/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/thirdparty/

