add_library(examples_shared STATIC npy_writer.c npy_dump_sink.c)
target_include_directories(examples_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(examples_shared PRIVATE
        Layer
        Tensor
        Quantization
        Rounding
        Common
)
