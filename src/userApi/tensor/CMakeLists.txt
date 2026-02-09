add_library(QuantizationApi QuantizationApi.c)
target_include_directories(QuantizationApi PUBLIC include)
target_link_libraries(QuantizationApi PRIVATE
        Quantization
        Rounding
        Common
        StorageApi
)

add_library(TensorApi TensorApi.c)
target_include_directories(TensorApi PUBLIC include)
target_link_libraries(TensorApi PRIVATE
        Tensor
        Rounding
        StorageApi
        Common
        TensorConversion
        Distributions
)