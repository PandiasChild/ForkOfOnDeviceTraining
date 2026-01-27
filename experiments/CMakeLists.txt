add_executable(MnistExperiment MnistExperiment.c)
target_link_libraries(MnistExperiment PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        LinearApi
        Linear

        ReluApi
        Relu

        QuantizationApi
        Quantization

        ReluApi
        Relu

        TensorApi
        Tensor
        Rounding

        TrainingApi
        Optimizer

        LossFunction
        CrossEntropy
        SoftmaxApi
        Softmax

        Sgd
        SgdApi

        InferenceApi

        CSVHelper

        Common

)