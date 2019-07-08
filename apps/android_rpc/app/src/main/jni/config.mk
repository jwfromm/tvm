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

APP_PLATFORM = android-28

# whether enable OpenCL during compile
USE_OPENCL = 1

USE_VULKAN = 1

USE_SORT = 1

# the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
ADD_C_INCLUDES += /home/jwfromm/Android/galaxy/adrenosdk/Development/Inc
ADD_C_INCLUDES += /home/jwfromm/OpenCL-Headers/ 

# the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
ADD_LDLIBS = /home/jwfromm/Android/galaxy/libOpenCL.so

# Include directories for waveone API
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/wo/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/wo/framework/default/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/wo/backend/armv8/
ADD_C_INCLUDES += ${WO_ROOT}/code/cc/src/thirdparty/

