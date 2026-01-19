add_library(LinearAPI LinearAPI.c)
target_include_directories(LinearAPI PUBLIC include)
target_link_libraries(LinearAPI PRIVATE
        Tensor
        Rounding
        Layer
        Linear
        Common
        StorageAPI
)

add_library(ReluAPI ReluAPI.c)
target_include_directories(ReluAPI PUBLIC include)
target_link_libraries(ReluAPI PRIVATE
        Tensor
        Rounding
        Layer
        Relu
        Common
        StorageAPI
)

add_library(SoftmaxAPI SoftmaxAPI.c)
target_include_directories(SoftmaxAPI PUBLIC include)
target_link_libraries(SoftmaxAPI PRIVATE
        Layer
        Tensor
        Rounding
        Common
        StorageAPI
)