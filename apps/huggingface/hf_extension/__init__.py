from . import _ffi_api

def RemoveRedundantTrans():
    return _ffi_api.RemoveRedundantTrans()

def RemoveRedundantReshape():
    return _ffi_api.RemoveRedundantReshape()