add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        StorageApi
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LayerQuant
        MORE_LIBS
        QuantizationApi
        Quantization
        Rounding
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LayerWeightsApi
        MORE_LIBS
        LinearApi
        QuantizationApi
        LayerQuant
        LayerCommon
        Quantization
        Rounding
        Linear
        Tensor
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        StateDictApi
        MORE_LIBS
        LinearApi
        ReluApi
        LayerQuant
        QuantizationApi
        LayerWeightsApi
        LayerCommon
        Quantization
        Rounding
        Linear
        Relu
        Tensor
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        InferenceApi
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        LinearApi
        ReluApi
        QuantizationApi
        LossFunction
        DataLoader
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        TrainingLoopApi
        MORE_LIBS
        TrainingEpochDefault
        TrainingBatchDefault
        CalculateGradsSequential
        CommonLayerLibs
        TensorApi
        LinearApi
        ReluApi
        SgdApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)

add_executable(UnitTestMultiLayerTraining UnitTestMultiLayerTraining.c)
target_link_libraries(UnitTestMultiLayerTraining PRIVATE
        unity
        TrainingLoopApi
        TrainingEpochDefault
        TrainingBatchDefault
        CalculateGradsSequential
        CommonLayerLibs
        TensorApi
        LinearApi
        ReluApi
        SoftmaxApi
        SgdApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)
__register_target_as_unit_test(UnitTestMultiLayerTraining)

add_executable(UnitTestMnistSmoke UnitTestMnistSmoke.c)
target_link_libraries(UnitTestMnistSmoke PRIVATE
        unity
        TrainingLoopApi
        TrainingEpochDefault
        TrainingBatchDefault
        CalculateGradsSequential
        CommonLayerLibs
        TensorApi
        LinearApi
        ReluApi
        SoftmaxApi
        SgdApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)
__register_target_as_unit_test(UnitTestMnistSmoke)

add_executable(UnitTestFlattenIntegration UnitTestFlattenIntegration.c)
target_link_libraries(UnitTestFlattenIntegration PRIVATE
        unity
        TrainingLoopApi
        CalculateGradsSequential
        CommonLayerLibs
        FlattenApi
        LinearApi
        SoftmaxApi
        SgdApi
        TensorApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        StorageApi
)
__register_target_as_unit_test(UnitTestFlattenIntegration)

add_executable(UnitTestOptimizerScaling UnitTestOptimizerScaling.c)
target_link_libraries(UnitTestOptimizerScaling PRIVATE
        unity
        OptimizerApi
        SgdApi
        Sgd
        Optimizer
        LinearApi
        CommonLayerLibs
        TensorApi
        QuantizationApi
        StorageApi
)
__register_target_as_unit_test(UnitTestOptimizerScaling)

add_executable(UnitTestQuantizationApiBitsControl UnitTestQuantizationApiBitsControl.c)
target_link_libraries(UnitTestQuantizationApiBitsControl PRIVATE
        unity
        QuantizationApi
        Quantization
        Rounding
)
__register_target_as_unit_test(UnitTestQuantizationApiBitsControl)