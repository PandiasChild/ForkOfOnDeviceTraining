add_library(OptimizerAPI OptimizerAPI.c)
target_include_directories(OptimizerAPI PUBLIC include)
target_link_libraries(OptimizerAPI PRIVATE
        Sgd
        Optimizer
        Tensor
        Rounding
        Layer
        Common
        StorageAPI
)

add_library(SgdAPI SgdAPI.c)
target_include_directories(SgdAPI PUBLIC include)
target_link_libraries(SgdAPI PRIVATE
        TensorAPI
        Tensor
        Rounding
        Layer
        Sgd
        Optimizer
        Common
        StorageAPI
)