add_library(Conv1dApi Conv1dApi.c)
target_include_directories(Conv1dApi PUBLIC include)
target_link_libraries(Conv1dApi PRIVATE
        Common
        Conv1d
        Distributions
        Kernel
        Layer
        LayerCommon
        LayerQuant
        Quantization
        QuantizationApi
        Rounding
        StorageApi
        Tensor
        TensorApi
)

add_library(LinearApi LinearApi.c)
target_include_directories(LinearApi PUBLIC include)
target_link_libraries(LinearApi PRIVATE
        Tensor
        TensorApi
        QuantizationApi
        Rounding
        Layer
        Linear
        LayerCommon
        LayerQuant
        Distributions
        Common
        StorageApi
)

add_library(ReluApi ReluApi.c)
target_include_directories(ReluApi PUBLIC include)
target_link_libraries(ReluApi PRIVATE
        Tensor
        Rounding
        Layer
        Relu
        LayerQuant
        Quantization
        Common
        StorageApi
)

add_library(SoftmaxApi SoftmaxApi.c)
target_include_directories(SoftmaxApi PUBLIC include)
target_link_libraries(SoftmaxApi PRIVATE
        Common
        Layer
        LayerQuant
        Quantization
        Rounding
        Softmax
        StorageApi
        Tensor
)

add_library(FlattenApi FlattenApi.c)
target_include_directories(FlattenApi PUBLIC include)
target_link_libraries(FlattenApi PRIVATE
        Tensor
        Rounding
        Layer
        Common
        StorageApi
)

add_library(Conv1dTransposedApi Conv1dTransposedApi.c)
target_include_directories(Conv1dTransposedApi PUBLIC include)
target_link_libraries(Conv1dTransposedApi PRIVATE
        Common
        Conv1dTransposed
        Distributions
        Kernel
        Layer
        LayerCommon
        LayerQuant
        Quantization
        QuantizationApi
        Rounding
        StorageApi
        Tensor
        TensorApi
)

add_library(Pool1dApi Pool1dApi.c)
target_include_directories(Pool1dApi PUBLIC include)
target_link_libraries(Pool1dApi PRIVATE
        AvgPool1d
        Common
        Kernel
        Layer
        LayerQuant
        MaxPool1d
        Quantization
        QuantizationApi
        Rounding
        StorageApi
        Tensor
        TensorApi
)