add_executable(train_c_mixed_width_mlp train_c.c)

target_link_libraries(train_c_mixed_width_mlp PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        QuantLayerApi
        QuantizationLayer

        SoftmaxApi
        Softmax

        ArithmeticType

        QuantizationApi
        Quantization

        TensorApi
        Tensor
        TensorConversion
        Rounding

        TrainingLoopApi
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        Optimizer

        LossFunction
        CrossEntropy

        Sgd
        SgdApi

        InferenceApi

        LayerQuant
        LayerCommon
        Distributions

        Common
        StorageApi
        RNG
)
