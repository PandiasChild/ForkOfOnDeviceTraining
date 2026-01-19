add_library(DataLoaderAPI DataLoaderAPI.c)
target_include_directories(DataLoaderAPI PUBLIC include)
target_link_libraries(DataLoaderAPI PRIVATE
        DataLoader
        Common
        StorageAPI
        Tensor
        Rounding
)

add_library(DatasetAPI DatasetAPI.c)
target_include_directories(DatasetAPI PUBLIC include)
target_link_libraries(DatasetAPI PRIVATE
        QuantizationAPI
        Quantization
        Rounding
        DataLoader
        Common
)

add_library(MnistNPYLoaderAPI MnistNPYLoaderAPI.c)
target_include_directories(MnistNPYLoaderAPI PUBLIC include)
target_link_libraries(MnistNPYLoaderAPI PRIVATE
        DataLoader
        TensorAPI
        Tensor
        Rounding
        Common
        StorageAPI
)