add_library(PpcaReplay PpcaReplay.c)
target_include_directories(PpcaReplay PUBLIC include)
target_link_libraries(PpcaReplay PRIVATE
        m
        Common
        Tensor
        DTypes
        Quantization
        Rounding
        TensorConversion
        ExecuteOp
        ArithmeticType
        Matmul
        Reduce
        Axpby
        RowBroadcast
        JacobiEig
        Sub
        Add
        Mul
        MinMax
        Distributions
        RNG
)
