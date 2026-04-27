add_library(Common__hdrs INTERFACE)
target_include_directories(Common__hdrs INTERFACE
        include
)
add_library(Common INTERFACE)
target_link_libraries(Common INTERFACE Common__hdrs)

# DEBUG_MODE_ERROR is the default — every ODT consumer gets PRINT_ERROR output
# unless explicitly silenced (no opt-out preset today; revisit when MCU presets land).
target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_ERROR=true)

if(DEBUG_MODE_INFO)
    target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_INFO=true)
endif()

if(DEBUG_MODE_DEBUG)
    target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_DEBUG=true)
endif()

