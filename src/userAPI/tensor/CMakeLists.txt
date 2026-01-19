add_library(QuantizationAPI QuantizationAPI.c)
target_include_directories(QuantizationAPI PUBLIC include)
target_link_libraries(QuantizationAPI PRIVATE
        Quantization
        Rounding
        Common
        StorageAPI
)

add_library(TensorAPI TensorAPI.c)
target_include_directories(TensorAPI PUBLIC include)
target_link_libraries(TensorAPI PRIVATE
        Tensor
        Rounding
        StorageAPI
        Common
        TensorConversion
)