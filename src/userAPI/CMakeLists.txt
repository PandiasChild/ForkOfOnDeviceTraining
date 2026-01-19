add_subdirectory(data_loader)
add_subdirectory(layer)
add_subdirectory(optimizer)
add_subdirectory(tensor)


add_library(InferenceAPI InferenceAPI.c)
target_include_directories(InferenceAPI PUBLIC include)
target_link_libraries(InferenceAPI PRIVATE
        TensorAPI
        Tensor
        Rounding
        Layer
        LossFunction
        Common
)

add_library(StorageAPI StorageAPI.c)
target_include_directories(StorageAPI PUBLIC include)
target_link_libraries(StorageAPI PRIVATE
        Tensor
        Rounding
        Common
)

add_library(TrainingAPI TrainingAPI.c)
target_include_directories(TrainingAPI PUBLIC include)
target_link_libraries(TrainingAPI PRIVATE
        TensorAPI
        Tensor
        Rounding
        Layer
        MSE
        Optimizer
        Common
        LossFunction
)
