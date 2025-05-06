cmake_minimum_required(VERSION 3.16)
project(whillats VERSION 1.0.0 LANGUAGES CXX C OBJCXX)

# Configure Metal backend for ggml on macOS (enable for non-Release, disable for Release)
if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Disabling GGML Metal backend for Release build on macOS to avoid compiler errors.")
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    else()
        set(GGML_METAL ON CACHE BOOL "" FORCE)
    endif()
endif()

# Ensure Position Independent Code is used for all targets
# Necessary when linking static libs into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(ExternalProject) # Needed for ExternalProject_Add

# Configure outputs per-configuration (Debug/Release) using generator expressions
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_FRAMEWORK_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Frameworks/$<CONFIG>)

# Additional compiler and linker flags
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wno-incompatible-pointer-types-discards-qualifiers)
add_compile_options(-Wno-exit-time-destructors -Wno-global-constructors)

# --- Dependency Declarations ---
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
file(MAKE_DIRECTORY ${THIRD_PARTY_DIR}) # Ensure third_party exists

# Make llava headers (clip.h, llava.h) available to all targets
include_directories(${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        v1.7.5            # pin to a known release
    SOURCE_DIR     ${THIRD_PARTY_DIR}/whisper.cpp
    GIT_SHALLOW    TRUE              # Faster clone
)
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
    SOURCE_DIR ${THIRD_PARTY_DIR}/llama.cpp
    GIT_SHALLOW TRUE # Faster clone
)

# --- iOS Framework Build ---
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(STATUS "Configuring for iOS framework build")
    # Build whisper for iOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static whisper build" FORCE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_STATIC ON CACHE BOOL "Force static whisper build" FORCE)
    add_subdirectory(${THIRD_PARTY_DIR}/whisper.cpp ${CMAKE_BINARY_DIR}/_deps/whisper-build EXCLUDE_FROM_ALL)
    # Build llama for iOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static llama build" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(llama)
    # Build llava from source (no FetchContent) as object and static library
    add_library(llava_obj OBJECT
        "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/llava.cpp"
        "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/clip.cpp"
    )
    target_include_directories(llava_obj PRIVATE
        "${THIRD_PARTY_DIR}/llama.cpp/include"
        "${THIRD_PARTY_DIR}/llama.cpp/common"
        "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"
    )
    set_target_properties(llava_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Define whillats.framework
    add_library(${PROJECT_NAME} SHARED
        src/whisper_transcription.cc
        src/llama_device_base.cc
        src/whillats.cc
        src/whillats_ios.mm
        src/whillats_synth.mm
        src/whillats_utils.cc
    )
    set_target_properties(${PROJECT_NAME} PROPERTIES
        FRAMEWORK ON
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER ai.webrtcsays.whillats
        XCODE_ATTRIBUTE_DEFINES_MODULE YES
        XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES YES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE TTS_PLATFORMS=0 WEBRTC_IOS)
    target_link_libraries(${PROJECT_NAME} PRIVATE whisper llama $<TARGET_OBJECTS:llava_obj>
        "-framework AVFoundation" "-framework Foundation" "-framework Accelerate"
        "-framework Metal" "-framework MetalKit" "-framework AudioToolbox"
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${THIRD_PARTY_DIR}/whisper.cpp/include
        ${THIRD_PARTY_DIR}/whisper.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/include
        ${THIRD_PARTY_DIR}/llama.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
    )
else()
    # --- Static Linux & macOS Build ---
    message(STATUS "Configuring for non-iOS static build (Linux & macOS)")

    # Build llava from source (llava.cpp & clip.cpp)
    add_library(llava_obj OBJECT
        "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/llava.cpp"
        "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/clip.cpp"
    )
    target_include_directories(llava_obj PRIVATE
        "${THIRD_PARTY_DIR}/llama.cpp/include"
        "${THIRD_PARTY_DIR}/llama.cpp/common"
        "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"
    )
    set_target_properties(llava_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Define whillats.framework
    add_library(${PROJECT_NAME} SHARED
        src/whisper_transcription.cc
        src/llama_device_base.cc
        src/whillats.cc
        src/whillats_ios.mm    # iOS speech synth implementation
        src/whillats_utils.cc
    )
    set_target_properties(${PROJECT_NAME} PROPERTIES
        FRAMEWORK ON
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER ai.webrtcsays.whillats
        XCODE_ATTRIBUTE_DEFINES_MODULE YES
        XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES YES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE TTS_PLATFORMS=0 WEBRTC_IOS)
    target_link_libraries(${PROJECT_NAME} PRIVATE whisper llama $<TARGET_OBJECTS:llava_obj>
        "-framework AVFoundation" "-framework Foundation" "-framework Accelerate"
        "-framework Metal" "-framework MetalKit" "-framework AudioToolbox"
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${THIRD_PARTY_DIR}/whisper.cpp/include
        ${THIRD_PARTY_DIR}/whisper.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/include
        ${THIRD_PARTY_DIR}/llama.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
    )
endif()

# --- Common Post-Build Steps (If any) ---

message(STATUS "CMake Project Name: ${PROJECT_NAME}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Apple Target: ${CMAKE_SYSTEM_NAME} STREQUAL \"iOS\"")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Library Output Directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "Framework Output Directory: ${CMAKE_FRAMEWORK_OUTPUT_DIRECTORY}")

# Example Installation (adjust paths and components as needed)
# install(TARGETS ${PROJECT_NAME}
#     EXPORT ${PROJECT_NAME}Targets
#     RUNTIME DESTINATION bin COMPONENT Runtime
#     LIBRARY DESTINATION lib COMPONENT Runtime
#     ARCHIVE DESTINATION lib/archive COMPONENT Development
#     FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}/Frameworks COMPONENT Runtime # Install framework
#     PUBLIC_HEADER DESTINATION include COMPONENT Development # Install public headers
# )
# install(EXPORT ${PROJECT_NAME}Targets
#     FILE ${PROJECT_NAME}Targets.cmake
#     NAMESPACE ${PROJECT_NAME}::
#     DESTINATION lib/cmake/${PROJECT_NAME}
# )

# if(NOT APPLE)
#     # Install espeak-ng data only for non-Apple
#     install(DIRECTORY ${ESPEAK_DATA_DEST_DIR}/
#         DESTINATION bin/espeak-ng-data COMPONENT Runtime
#     )
# endif()

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.m"
    PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
)