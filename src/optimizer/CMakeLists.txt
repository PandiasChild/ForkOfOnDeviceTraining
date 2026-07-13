add_library(Optimizer Optimizer.c)
target_include_directories(Optimizer PUBLIC include)
target_link_libraries(Optimizer PRIVATE
        Quantization
        Rounding
        Layer
        Conv1d
        Conv1dTransposed
        Linear
        Tensor
        Common
)

add_library(Sgd Sgd.c)
target_include_directories(Sgd PUBLIC include)
target_link_libraries(Sgd PRIVATE
        DTypes
        Tensor
        Arithmetic
        ArithmeticType
        ExecuteOp
        Layer
        Common
)

add_library(LrScheduler LrScheduler.c)
target_include_directories(LrScheduler PUBLIC include)
target_link_libraries(LrScheduler PRIVATE
        Optimizer
        # #327: PRIVATE-everywhere convention (docs/conventions) - Optimizer.h
        # transitively includes Layer.h/Tensor.h/Quantization.h/Rounding.h;
        # Optimizer's own links to those are PRIVATE, so consumers must list
        # the header-owning targets too (same precedent as Sgd's own deps).
        Layer
        Tensor
        Arithmetic
        Common
        m
)
