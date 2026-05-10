add_library(examples_shared STATIC npy_writer.c)
target_include_directories(examples_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
