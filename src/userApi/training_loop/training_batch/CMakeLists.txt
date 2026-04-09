add_library(TrainingBatchDefault TrainingBatchDefault.c)
target_include_directories(TrainingBatchDefault PUBLIC include)
target_link_libraries(TrainingBatchDefault PRIVATE
        TrainingLoopApi
        Tensor
        Rounding
        Layer
        Optimizer
        LossFunction
        InferenceApi
        DataLoader
        Common
)
