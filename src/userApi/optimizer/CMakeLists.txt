add_library(OptimizerApi OptimizerApi.c)
target_include_directories(OptimizerApi PUBLIC include)
target_link_libraries(OptimizerApi PRIVATE
        Sgd
        Optimizer
        Tensor
        TensorApi
        Rounding
        Layer
        Conv1d
        Conv1dTransposed
        GroupNorm
        LayerNorm
        Linear
        Common
        StorageApi
)

add_library(SgdApi SgdApi.c)
target_include_directories(SgdApi PUBLIC include)
target_link_libraries(SgdApi PRIVATE
        TensorApi
        Tensor
        Rounding
        Layer
        Conv1d
        Conv1dTransposed
        LayerNorm
        Sgd
        Optimizer
        OptimizerApi
        Common
        StorageApi
)