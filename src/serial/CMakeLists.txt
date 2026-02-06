add_library(Serialize Serialize.c)
target_include_directories(Serialize PUBLIC include)
target_link_libraries(Serialize PRIVATE
        Common
        Tensor
        Quantization
        Rounding

        Layer
)

add_library(Deserialize Deserialize.c)
target_include_directories(Deserialize PUBLIC include)
target_link_libraries(Deserialize PRIVATE
        Common
        Tensor
        Quantization
        Rounding

        Layer
)