add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Linear
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        LinearApi
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Relu
        MORE_LIBS
        CommonLayerLibs
        ReluApi
        TensorApi
        QuantizationApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Softmax
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        SoftmaxApi
        QuantizationApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Flatten
        MORE_LIBS
        CommonLayerLibs
        FlattenApi
        TensorApi
        QuantizationApi
)