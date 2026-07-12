add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        PpcaReplay
        MORE_LIBS
        m
        PpcaReplayApi
        Tensor
        TensorApi
        QuantizationApi
        Quantization
        Rounding
        TensorConversion
        ExecuteOp
        Common
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
        StorageApi
        DataLoader
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        PpcaReplaySerialize
        MORE_LIBS
        m
        PpcaReplay
        PpcaReplayApi
        Serialize
        Deserialize
        Tensor
        TensorApi
        TensorConversion
        ExecuteOp
        ArithmeticType
        Quantization
        QuantizationApi
        Rounding
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
        StorageApi
        DataLoader
        Common
        odt_test_death
)
target_compile_definitions(UnitTestPpcaReplaySerialize PRIVATE
        PPCA_SERIALIZE_TEST_FILE_PATH="${CMAKE_CURRENT_BINARY_DIR}/PpcaReplayTestFile.bin"
)

# add_elastic_ai_unit_test derives NAME = "UnitTest" + LIB_UNDER_TEST and always
# compiles exactly ${NAME}.c against that one library; it has no MORE_SOURCES
# hook to redirect the source file. ReplayDataLoader isn't a library of its own
# (the wrapper lives inside PpcaReplayApi), so the helper can't express this
# target — register it manually with the same effect (executable + unity link
# + all_unit_tests dependency + labeled test).
add_executable(UnitTestReplayDataLoader UnitTestReplayDataLoader.c)
target_link_libraries(UnitTestReplayDataLoader PRIVATE
        PpcaReplayApi
        PpcaReplay
        DataLoaderApi
        DataLoader
        Tensor
        TensorApi
        Quantization
        QuantizationApi
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
        StorageApi
        Common
        m
        unity
        odt_test_death)
add_test(UnitTestReplayDataLoader UnitTestReplayDataLoader)
set_property(TEST UnitTestReplayDataLoader PROPERTY LABELS unit)
if(TARGET all_unit_tests)
    add_dependencies(all_unit_tests UnitTestReplayDataLoader)
endif()
