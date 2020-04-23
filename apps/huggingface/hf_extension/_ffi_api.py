import ctypes
import tvm._ffi

lib = ctypes.CDLL("./hf_extension/libhf_passes.so", ctypes.RTLD_GLOBAL)

tvm._ffi._init_api("hf_extension", __name__)

