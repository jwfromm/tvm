if (USE_WAVEONE)
  set (WO_ROOT src/contrib/waveone/wolib)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DWO_LOG_LEVEL=3")
  message(STATUS "Build with waveone support.")
  include_directories(
	${WO_ROOT}
	${WO_ROOT}/thirdparty
	${WO_ROOT}/wo/framework/default
	${WO_ROOT}/wo/backend/x86	
  )
  file(GLOB WAVEONE_CONTRIB_SRC 
  	src/contrib/waveone/*.cc
	${WO_ROOT}/thirdparty/**/*.cc
	${WO_ROOT}/wo/common/*.cc
	${WO_ROOT}/wo/aec/*.cc
	${WO_ROOT}/wo/md5/*.cc
	${WO_ROOT}/wo/ops/*.cc
	${WO_ROOT}/wo/thirdparty/**/*.cc
  )
  list(APPEND RUNTIME_SRCS ${WAVEONE_CONTRIB_SRC})
  add_subdirectory(${WO_ROOT})
endif(USE_WAVEONE)
