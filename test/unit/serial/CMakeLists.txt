add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Deserialize
        MORE_LIBS
        TensorApi
        Tensor
        Rounding
        Quantization
        Serialize

        QuantizationApi
        Layer
        Linear
        LinearApi
        LayerQuant
        ArithmeticType
        Relu
        ReluApi
        Softmax
        SoftmaxApi

        StorageApi
)

# Absolute scratch-file path so the test runs from any working directory
# (host build dir vs. Docker /work, see Phase 2 LSan migration).
target_compile_definitions(UnitTestDeserialize PRIVATE
        SERIALIZE_TEST_FILE_PATH="${CMAKE_CURRENT_BINARY_DIR}/SerializeTestFile.bin"
)