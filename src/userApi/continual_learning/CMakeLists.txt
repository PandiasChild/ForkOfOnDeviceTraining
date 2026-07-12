add_library(PpcaReplayApi PpcaReplayApi.c)
target_include_directories(PpcaReplayApi PUBLIC include)
target_link_libraries(PpcaReplayApi PRIVATE
        Common
        PpcaReplay
        Tensor
        TensorApi
        Quantization
        QuantizationApi
        ArithmeticType
        Rounding
        RNG
        DataLoader
        StorageApi
)
