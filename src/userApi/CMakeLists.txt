add_subdirectory(data_loader)
add_subdirectory(layer)
add_subdirectory(optimizer)
add_subdirectory(tensor)


add_library(InferenceApi InferenceApi.c)
target_include_directories(InferenceApi PUBLIC include)
target_link_libraries(InferenceApi PRIVATE
        TensorApi
        Tensor
        Rounding
        Layer
        LossFunction
        Common
        DataLoader
)

add_library(StorageApi StorageApi.c)
target_include_directories(StorageApi PUBLIC include)
target_link_libraries(StorageApi PRIVATE
        Tensor
        Rounding
        Common
)

add_library(TrainingApi TrainingApi.c)
target_include_directories(TrainingApi PUBLIC include)
target_link_libraries(TrainingApi PRIVATE
        TensorApi
        Tensor
        Rounding
        Layer
        MSE
        Optimizer
        Common
        LossFunction
        DataLoader
)
