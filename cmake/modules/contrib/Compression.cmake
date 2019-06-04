if (USE_COMPRESSION)
  message(STATUS "Build with waveone compression.")
  file(GLOB COMPRESSION_CONTRIB_SRC src/contrib/compression/*.cc)
  list(APPEND RUNTIME_SRCS ${COMPRESSION_CONTRIB_SRC})
  add_subdirectory(src/contrib/compression/wolib)
endif(USE_COMPRESSION)
