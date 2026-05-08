find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GEN_CONV1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d.py
                --out ${GEN_CONV1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d.py
        COMMENT "Generating expected_conv1d.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d
        DEPENDS ${GEN_CONV1D_HEADER})

set(GEN_CONV1D_TRANSPOSED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d_transposed.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_TRANSPOSED_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_transposed.py
                --out ${GEN_CONV1D_TRANSPOSED_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_transposed.py
        COMMENT "Generating expected_conv1d_transposed.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d_transposed
        DEPENDS ${GEN_CONV1D_TRANSPOSED_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1d
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        Conv1dApi
        Kernel
        QuantizationApi
        StorageApi
)
add_dependencies(UnitTestConv1d generate_expected_conv1d)
target_include_directories(UnitTestConv1d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Linear
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        LinearApi
        StorageApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Relu
        MORE_LIBS
        CommonLayerLibs
        ReluApi
        TensorApi
        QuantizationApi
        StorageApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Softmax
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        SoftmaxApi
        QuantizationApi
        StorageApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Flatten
        MORE_LIBS
        CommonLayerLibs
        FlattenApi
        TensorApi
        QuantizationApi
        StorageApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1dTransposed
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        Kernel
        StorageApi
)
add_dependencies(UnitTestConv1dTransposed generate_expected_conv1d_transposed)
target_include_directories(UnitTestConv1dTransposed PRIVATE ${CMAKE_CURRENT_BINARY_DIR})