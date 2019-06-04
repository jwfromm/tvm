if (USE_COMPRESSION)
  set (WO_ROOT src/contrib/compression/wolib)
  message(STATUS "Build with waveone compression.")
  include_directories(
	${WO_ROOT}
	${WO_ROOT}/thirdparty
	${WO_ROOT}/wo/framework/default
	${WO_ROOT}/wo/backend/x86	
  )
  file(GLOB COMPRESSION_CONTRIB_SRC 
  	src/contrib/compression/*.cc
	${WO_ROOT}/thirdparty/**/*.cc
	${WO_ROOT}/wo/common/*.cc
	${WO_ROOT}/wo/aec/*.cc
	${WO_ROOT}/wo/md5/*.cc
	${WO_ROOT}/wo/ops/*.cc
	${WO_ROOT}/wo/thirdparty/**/*.cc
  )
  list(APPEND RUNTIME_SRCS ${COMPRESSION_CONTRIB_SRC})
  add_subdirectory(${WO_ROOT})
endif(USE_COMPRESSION)
