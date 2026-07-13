# Cross toolchain for the compile-only ARM CI job. Cortex-M4/thumb is a
# representative 32-bit MCU target (ILP32: 32-bit size_t/pointers, newlib),
# NOT a product pin -- the job exists to surface the bug classes that are
# structurally invisible on LP64 hosts (e.g. 1<<32 shift UB, PR #297).
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)

# No OS, no linkable runtime: compile-only. Static-library try-compile keeps
# CMake's compiler check from demanding _exit/_sbrk stubs at configure time.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-m4 -mthumb")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
