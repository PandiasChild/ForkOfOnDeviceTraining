add_executable(train_c_har_classifier train_c.c)

target_link_libraries(train_c_har_classifier PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        MaxPool1d
        AvgPool1d

        QuantizationApi
        Quantization

        TensorApi
        Tensor
        Rounding

        TrainingLoopApi
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        Optimizer

        LossFunction
        CrossEntropy

        SoftmaxApi
        Softmax

        Sgd
        SgdApi

        InferenceApi

        Common
        StorageApi
        RNG

        examples_shared
)

add_executable(train_c_binary_har_classifier binary_classifier_train_c.c)

target_link_libraries(train_c_binary_har_classifier PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        MaxPool1d
        AvgPool1d

        QuantizationApi
        Quantization

        TensorApi
        Tensor
        Rounding

        TrainingLoopApi
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        Optimizer

        LossFunction
        CrossEntropy

        SoftmaxApi
        Softmax

        Sgd
        SgdApi

        InferenceApi

        Common
        StorageApi
        RNG

        examples_shared
)
