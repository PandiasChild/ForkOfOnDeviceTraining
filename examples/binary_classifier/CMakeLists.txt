add_executable(train_c_binary_classifier_speech_commands train_SPEECH_COMMANDS_c.c)
link_libraries(train_c_binary_classifier_speech_commands PRIVATE
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

add_executable(train_c_binary_classifier train_YES_NO_c.c)
target_link_libraries(train_c_binary_classifier PRIVATE
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
