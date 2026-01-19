add_library(DataLoader DataLoader.c)
target_include_directories(DataLoader PUBLIC include)
target_link_libraries(DataLoader PRIVATE
        Tensor
        Rounding
        Common
        RNG
)

add_library(MnistNPYLoader MnistNPYLoader.c)
target_include_directories(MnistNPYLoader PUBLIC include)
target_link_libraries(MnistNPYLoader PRIVATE
        Tensor
        Rounding
        Common
)