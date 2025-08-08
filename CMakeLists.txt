cmake_minimum_required(VERSION 3.16)

set(LLAMA_TAG "b6119")
set(WHISPER_TAG "v1.7.6")

# Set project languages conditionally based on platform
if(APPLE)
    project(whillats VERSION 1.0.0 LANGUAGES CXX C OBJC OBJCXX)
else()
    project(whillats VERSION 1.0.0 LANGUAGES CXX C)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
file(MAKE_DIRECTORY ${THIRD_PARTY_DIR})

# Ensure per-config output dirs use lowercase names
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<LOWER_CASE:$<CONFIG>>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<LOWER_CASE:$<CONFIG>>)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<LOWER_CASE:$<CONFIG>>)

# Make mtmd headers available
include_directories(${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        ${WHISPER_TAG}
    SOURCE_DIR     ${THIRD_PARTY_DIR}/whisper.cpp
    GIT_SHALLOW    TRUE              # Faster clone
)
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
    GIT_TAG        ${LLAMA_TAG}
    SOURCE_DIR ${THIRD_PARTY_DIR}/llama.cpp
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama)

# Patch mtmd-audio.cpp template param if needed (after llama is fetched)
set(MTMD_AUDIO_CPP "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp")
if(EXISTS ${MTMD_AUDIO_CPP})
    file(READ ${MTMD_AUDIO_CPP} MTMD_AUDIO_SRC)
    string(REPLACE "std::vector data(" "std::vector<float> data(" MTMD_AUDIO_FIXED "${MTMD_AUDIO_SRC}")
    if(NOT "${MTMD_AUDIO_SRC}" STREQUAL "${MTMD_AUDIO_FIXED}")
        file(WRITE ${MTMD_AUDIO_CPP} "${MTMD_AUDIO_FIXED}")
        message(STATUS "Patched mtmd-audio.cpp to use std::vector<float>")
    endif()
endif()

# whisper.cpp only for non-iOS
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(WHISPER_STATIC ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        whisper
        GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
        GIT_TAG        master
        SOURCE_DIR     ${THIRD_PARTY_DIR}/whisper.cpp
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(whisper)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # Build mtmd objects
    add_library(mtmd_obj OBJECT
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd.cpp
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-helper.cpp
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/clip.cpp
    )
    target_include_directories(mtmd_obj PRIVATE
        ${THIRD_PARTY_DIR}/llama.cpp/include
        ${THIRD_PARTY_DIR}/llama.cpp/common
        ${THIRD_PARTY_DIR}/llama.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
        ${THIRD_PARTY_DIR}/llama.cpp/vendor)
    set_target_properties(mtmd_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(${PROJECT_NAME} SHARED
        src/llama_device_base.cc
        src/whillats.cc
        src/whillats_ios.mm
        src/whillats_synth.mm
        src/whillats_utils.cc)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        FRAMEWORK ON
        FRAMEWORK_VERSION A
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_WEAK YES)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WEBRTC_IOS)
    target_link_libraries(${PROJECT_NAME} PRIVATE llama $<TARGET_OBJECTS:mtmd_obj>
        "-framework AVFoundation" "-framework Foundation" "-framework Accelerate"
        "-framework Metal" "-framework MetalKit" "-framework AudioToolbox")

    install(TARGETS ${PROJECT_NAME} FRAMEWORK DESTINATION bin)
else()
    # Non-iOS shared library
    set(WHILLATS_SOURCES
        src/llama_device_base.cc
        src/whillats.cc
        src/whillats_utils.cc
        src/whisper_transcription.cc)
    if(UNIX AND NOT APPLE)
        list(APPEND WHILLATS_SOURCES src/espeak_tts.cc)
    endif()
    if(APPLE)
        list(APPEND WHILLATS_SOURCES src/whillats_synth.mm src/whillats_ios.mm src/synthesis.cc)
        # Link mtmd sources on macOS for multimodal support
        list(APPEND WHILLATS_SOURCES
            ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd.cpp
            ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-helper.cpp
            # ensure patched std::vector<float> line below is applied before compiling
            ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp
            ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/clip.cpp)
    endif()

    add_library(${PROJECT_NAME} SHARED ${WHILLATS_SOURCES})
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${THIRD_PARTY_DIR}/llama.cpp/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
        ${THIRD_PARTY_DIR}/llama.cpp/vendor)
    target_link_libraries(${PROJECT_NAME} PRIVATE llama)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        target_link_libraries(${PROJECT_NAME} PRIVATE whisper)
    endif()
    if(APPLE)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "-framework Accelerate" "-framework Metal" "-framework MetalKit"
            "-framework CoreAudio" "-framework AudioToolbox" "-framework AVFoundation")
    elseif(UNIX)
        target_link_libraries(${PROJECT_NAME} PRIVATE stdc++)
    endif()

    install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin)

    add_executable(test_whillats
        test/test_whillats.cc
        test/test_utils.cc)
    target_include_directories(test_whillats PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_whillats PRIVATE ${PROJECT_NAME})
    install(TARGETS test_whillats RUNTIME DESTINATION bin)
endif()
