cmake_minimum_required(VERSION 3.14)
project(whillats VERSION 1.0.0 LANGUAGES CXX)

# Ensure Debug build type for whillats and subdirectories (llama, whisper)
set(CMAKE_BUILD_TYPE Debug)

# Set output directories - all to bin (MOVE THIS TO THE TOP)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Additional compiler and linker flags
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wno-incompatible-pointer-types-discards-qualifiers)
add_compile_options(-Wno-exit-time-destructors -Wno-global-constructors)

# Add whisper.cpp
add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

# Add llama.cpp
set(LLAMA_STATIC ON)
set(LLAMA_BUILD_EXAMPLES OFF)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# Add espeak-ng and its components
add_library(espeak-ng STATIC IMPORTED)
set_target_properties(espeak-ng PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/src/include
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/build/src/libespeak-ng/libespeak-ng.a
)

add_library(speechPlayer STATIC IMPORTED)
set_target_properties(speechPlayer PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/build/src/speechPlayer/libspeechPlayer.a
)

add_library(ucd STATIC IMPORTED)
set_target_properties(ucd PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/build/src/ucd-tools/libucd.a
)

# Add pcaudio
add_library(pcaudio STATIC IMPORTED)
set_target_properties(pcaudio PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcaudiolib/src/.libs/libpcaudio.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcaudiolib/src/include
)

# Create library target
add_library(${PROJECT_NAME} SHARED
    src/whisper_transcription.cc
    src/llama_device_base.cc
    src/espeak_tts.cc
    src/whillats.cc
)

# Add additional include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcaudiolib/src/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# Add library directories
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/build/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build/bin
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/build/src/libespeak-ng
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/build/src/speechPlayer
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng/build/src/ucd-tools
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcaudiolib/src
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        whisper
        llama
        ggml
        ggml-base
        ggml-cpu
        espeak-ng
        speechPlayer
        ucd
        pcaudio
)

# Add after project() declaration, before other settings
if(APPLE)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif()

# Modify the target compile definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
        $<$<PLATFORM_ID:Darwin>:PLATFORM_DARWIN>
        WHILLATS_BUILDING_DLL
    PUBLIC
        $<$<PLATFORM_ID:Darwin>:WHILLATS_DLL>
)

if(APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_DARWIN)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Accelerate"
        "-framework Metal"
        "-framework MetalKit"
        "-framework CoreAudio"
        "-framework AudioToolbox"
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        "-Wl,-rpath,@loader_path/third_party/whisper.cpp/build/src"
        "-Wl,-rpath,@loader_path/third_party/whisper.cpp/build/ggml/src"
        "-Wl,-rpath,@loader_path/third_party/llama.cpp/build/bin"
        "-Wl,-rpath,@loader_path/third_party/llama.cpp/build/ggml/src"
        "-Wl,-rpath,@loader_path/third_party/pcaudiolib/src"
    )
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_LINUX)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        gomp
        stdc++
    )
    target_link_directories(${PROJECT_NAME} PRIVATE
        /usr/lib/x86_64-linux-gnu
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        -Wl,--allow-shlib-undefined
    )
endif()

# Set output directories - all to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add test executable
add_executable(test_whillats
    test/test_whillats.cc
    test/test_utils.cc
)

target_link_libraries(test_whillats
    PRIVATE
        ${PROJECT_NAME}
)

# Set rpath for the test executable
if(APPLE)
    set_target_properties(test_whillats PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@executable_path"
    )
endif()

# Install targets to bin
install(TARGETS ${PROJECT_NAME} test_whillats
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
)
target_include_directories(test_whillats
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add rpath for test executable
if(APPLE)
    set_target_properties(test_whillats PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@executable_path"
    )
endif()