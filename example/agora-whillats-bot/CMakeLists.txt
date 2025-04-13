cmake_minimum_required(VERSION 3.14)
project(
     AgoraTransceiver
)

# Set C++14 as the required standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable debug information
set(CMAKE_BUILD_TYPE Debug)

# Add compiler warnings
add_compile_options(-Wno-inconsistent-missing-override)

# Add include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/agora_rtc_sdk/agora_sdk/include
    ${PROJECT_SOURCE_DIR}/agora_rtc_sdk/example
    ${PROJECT_SOURCE_DIR}/agora_rtc_sdk/example/common
    ${PROJECT_SOURCE_DIR}/../../src
)

# Define the path to whillats libraries
set(WHILLATS_LIB_DIR ${PROJECT_SOURCE_DIR}/whillats/build/bin)

# Define path to the extracted Agora SDK (from parent scope)
# Check if the variable is set - it should be by FetchContent_MakeAvailable
if(NOT DEFINED agora_sdk_SOURCE_DIR)
    message(FATAL_ERROR "Agora SDK source directory (agora_sdk_SOURCE_DIR) is not defined. Ensure FetchContent_MakeAvailable(agora_sdk) was called in the parent CMakeLists.txt")
endif()

set(AGORA_SDK_INCLUDE_DIR ${agora_sdk_SOURCE_DIR}/agora_sdk/include)
set(AGORA_SDK_LIBS_DIR ${agora_sdk_SOURCE_DIR}/agora_sdk)
# Assuming common example files are still relative to SDK root, adjust if needed:
set(AGORA_SDK_COMMON_EXAMPLE_DIR ${agora_sdk_SOURCE_DIR}/example/common)

# Check if Agora SDK directories exist
if(NOT IS_DIRECTORY ${AGORA_SDK_INCLUDE_DIR})
    message(WARNING "Agora SDK include directory not found at ${AGORA_SDK_INCLUDE_DIR}")
endif()
if(NOT IS_DIRECTORY ${AGORA_SDK_LIBS_DIR})
    message(WARNING "Agora SDK libs directory not found at ${AGORA_SDK_LIBS_DIR}")
endif()
if(NOT IS_DIRECTORY ${AGORA_SDK_COMMON_EXAMPLE_DIR})
    message(WARNING "Agora SDK common example directory not found at ${AGORA_SDK_COMMON_EXAMPLE_DIR}")
endif()

# Build sample_receive_yuv_pcm
file(GLOB TRANSCEIVER_YUV_PCM_CPP_FILES
     "${PROJECT_SOURCE_DIR}/transceiver_yuv_pcm.cpp"
     # Use variable for common source files path
     "${AGORA_SDK_COMMON_EXAMPLE_DIR}/log.cpp"
     "${AGORA_SDK_COMMON_EXAMPLE_DIR}/opt_parser.cpp"
     "${AGORA_SDK_COMMON_EXAMPLE_DIR}/sample_common.cpp"
     "${AGORA_SDK_COMMON_EXAMPLE_DIR}/sample_connection_observer.cpp"
     "${AGORA_SDK_COMMON_EXAMPLE_DIR}/sample_event.cpp"
     )
add_executable(transceiver_yuv_pcm ${TRANSCEIVER_YUV_PCM_CPP_FILES}) 

# Add include directories for the target
target_include_directories(transceiver_yuv_pcm PRIVATE
    ${AGORA_SDK_INCLUDE_DIR} # For agora includes <IAgora...>
    ${agora_sdk_SOURCE_DIR}/example # For example includes "common/helper.h"
    # Include directories from whillats should be inherited via linking
)

# Define the expected path for the copied library
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}) # From parent scope
set(AGORA_SDK_COPIED_LIB_PATH ${EXECUTABLE_OUTPUT_PATH}/libagora_rtc_sdk.so)

# Link against the absolute path of the *copied* library and whillats target
target_link_libraries(transceiver_yuv_pcm PRIVATE
    ${AGORA_SDK_COPIED_LIB_PATH} # Link the path where the library will be copied
    whillats                   # Link whillats target from parent project
    # Dependencies like whisper, llama, ggml, pthread, dl, m are handled by whillats
)

# Find the original library path for the copy command
# This still needs to run before the copy target uses it.
find_library(AGORA_SDK_ORIGINAL_LIBRARY agora_rtc_sdk
    HINTS ${AGORA_SDK_LIBS_DIR}
    NO_DEFAULT_PATH
)

# Create a custom target to copy the Agora SDK library
if(AGORA_SDK_ORIGINAL_LIBRARY)
    add_custom_target(copy_agora_sdk_for_example ALL # Add ALL so it runs by default
        COMMENT "Copying Agora SDK library (${AGORA_SDK_ORIGINAL_LIBRARY}) for transceiver_yuv_pcm"
        COMMAND cp "${AGORA_SDK_ORIGINAL_LIBRARY}" "${EXECUTABLE_OUTPUT_PATH}/" # Use system cp, note trailing slash on dest
        VERBATIM # Use VERBATIM for system commands to avoid potential escaping issues
    )
    # Ensure the copy happens before the executable is linked/used
    add_dependencies(transceiver_yuv_pcm copy_agora_sdk_for_example)
else()
    message(FATAL_ERROR "Original Agora SDK library not found in ${AGORA_SDK_LIBS_DIR}. Cannot create copy target.") # Changed to FATAL
endif()

# Optional: Set RPATH so the executable can find the shared libraries at runtime
# Point to the main build output directory (where libwhillats.so is) and executable's own dir.
set_target_properties(transceiver_yuv_pcm PROPERTIES
    INSTALL_RPATH "$ORIGIN:${EXECUTABLE_OUTPUT_PATH}" 
    BUILD_WITH_INSTALL_RPATH TRUE
)
