cmake_minimum_required(VERSION 3.16)
project(whillats VERSION 1.0.0 LANGUAGES CXX C OBJC OBJCXX)

# Configure Metal backend for ggml on macOS (enable for non-Release, disable for Release)
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

# Ensure Position Independent Code is used for all targets
# Necessary when linking static libs into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(ExternalProject) # Needed for ExternalProject_Add

# Configure outputs per-configuration using generator expressions
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

# Make mtmd headers (clip.h, mtmd.h) available to all targets
include_directories(${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        master
    SOURCE_DIR     ${THIRD_PARTY_DIR}/whisper.cpp
    GIT_SHALLOW    TRUE              # Faster clone
)
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
    SOURCE_DIR ${THIRD_PARTY_DIR}/llama.cpp
    GIT_SHALLOW TRUE # Faster clone
)
FetchContent_Declare(
    espeak-ng-proj # Rename to avoid conflict with potential target name
    GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng
    SOURCE_DIR ${THIRD_PARTY_DIR}/espeak-ng
    GIT_SHALLOW TRUE # Faster clone
)
FetchContent_Declare(
    pcaudiolib-proj # Rename to avoid conflict
    GIT_REPOSITORY https://github.com/espeak-ng/pcaudiolib
    SOURCE_DIR ${THIRD_PARTY_DIR}/pcaudiolib
    GIT_SHALLOW TRUE # Faster clone
)

# Declare Agora SDK (Only for Linux)
if(UNIX AND NOT APPLE)
    FetchContent_Declare(
        agora_sdk
        URL https://download.agora.io/rtsasdk/release/Agora-RTC-x86_64-linux-gnu-v4.4.30-20241024_101940-398537.tgz
        # Specify the subdirectory inside the archive that contains the SDK
        SOURCE_SUBDIR Agora-RTC-x86_64-linux-gnu-v4.4.30-20241024_101940-398537 
        # CMake should automatically handle extraction for .tgz
    )
endif()

# --- Populate and Build Dependencies ---

# llama.cpp (CMake)
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "iOS") AND NOT (CMAKE_OSX_SYSROOT MATCHES ".*iPhoneOS.*"))
    set(LLAMA_STATIC ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(UNIX AND NOT APPLE) # Linux specific
        set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
    else()
        set(GGML_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
    endif()
    FetchContent_MakeAvailable(llama)
endif()

# whisper.cpp (CMake)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libs for whisper" FORCE)
set(WHISPER_STATIC ON CACHE BOOL "Build static libs for whisper" FORCE) # Explicitly static
if(UNIX AND NOT APPLE) # Linux specific
    set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
else()
    set(GGML_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
endif()
# Disable Metal backend on macOS and iOS
set(GGML_METAL OFF CACHE BOOL "Disable ggml Metal backend" FORCE)
FetchContent_MakeAvailable(whisper)

# Build pcaudiolib and espeak-ng only on Linux (skip on macOS and iOS)
if(UNIX AND NOT APPLE)
    # pcaudiolib (Autotools)
    FetchContent_GetProperties(pcaudiolib-proj)
    if(NOT pcaudiolib-proj_POPULATED)
        cmake_policy(SET CMP0169 OLD)
        FetchContent_Populate(pcaudiolib-proj)
        cmake_policy(SET CMP0169 NEW)
        message(STATUS "pcaudiolib source downloaded to ${pcaudiolib-proj_SOURCE_DIR}")
    endif()
    
    set(PCAUDIOLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/pcaudiolib-build)
    set(PCAUDIOLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/pcaudiolib-install)
    set(PCAUDIOLIB_LIB_PATH ${PCAUDIOLIB_BUILD_DIR}/src/.libs/libpcaudio.a)
    
    execute_process(
        COMMAND sh ./autogen.sh
        WORKING_DIRECTORY ${pcaudiolib-proj_SOURCE_DIR}
        RESULT_VARIABLE _autogen_res
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _autogen_res EQUAL 0)
        message(FATAL_ERROR "autogen.sh for pcaudiolib failed: ${_autogen_res}")
    endif()

    ExternalProject_Add(pcaudiolib_external
        SOURCE_DIR ${pcaudiolib-proj_SOURCE_DIR}
        BINARY_DIR ${PCAUDIOLIB_BUILD_DIR}
        CONFIGURE_COMMAND bash -c "export CFLAGS='-fPIC -I${pcaudiolib-proj_SOURCE_DIR}/src/include' && export CXXFLAGS='-fPIC -I${pcaudiolib-proj_SOURCE_DIR}/src/include' && exec <SOURCE_DIR>/configure --with-pic --prefix='${PCAUDIOLIB_INSTALL_DIR}'"
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${PCAUDIOLIB_LIB_PATH}
        LOG_CONFIGURE YES
        LOG_BUILD YES
    )

    add_library(pcaudio STATIC IMPORTED)
    set_target_properties(pcaudio PROPERTIES
        IMPORTED_LOCATION ${PCAUDIOLIB_LIB_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${pcaudiolib-proj_SOURCE_DIR}/src/include
    )
    add_dependencies(pcaudio pcaudiolib_external)

    # espeak-ng (CMake)
    set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "C compiler" FORCE)
    list(APPEND CMAKE_PROGRAM_PATH /usr/bin /bin)
    set(CMAKE_PROGRAM_PATH ${CMAKE_PROGRAM_PATH} CACHE STRING "Path for CMake to search for programs" FORCE)
    FetchContent_MakeAvailable(espeak-ng-proj)

    # Define paths for espeak-ng data
    set(ESPEAK_DATA_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/espeak-ng-proj-build/espeak-ng-data)
    set(ESPEAK_DATA_DEST_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/espeak-ng-data)

    add_custom_command(
        OUTPUT ${ESPEAK_DATA_DEST_DIR}/phontab
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${ESPEAK_DATA_SOURCE_DIR} ${ESPEAK_DATA_DEST_DIR}
        DEPENDS espeak-ng-proj
        COMMENT "Copying espeak-ng-data to ${ESPEAK_DATA_DEST_DIR}"
    )
    add_custom_target(copy_espeak_data ALL DEPENDS ${ESPEAK_DATA_DEST_DIR}/phontab)
endif()  # end if(UNIX AND NOT APPLE)

# --- mtmd example shared library ---
find_package(Threads REQUIRED)

add_library(mtmd OBJECT
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd.h
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.h
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-helper.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd.h
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/clip.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/clip.h
    )

target_link_libraries(mtmd PRIVATE ggml llama Threads::Threads)

target_include_directories(mtmd PUBLIC
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
    ${THIRD_PARTY_DIR}/llama.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/include
    ${THIRD_PARTY_DIR}/llama.cpp/common
    ${THIRD_PARTY_DIR}/llama.cpp/ggml/include # Added for ggml.h
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
    ${THIRD_PARTY_DIR}/llama.cpp/vendor
)

target_compile_features(mtmd PRIVATE cxx_std_17)

add_library(mtmd_shared STATIC $<TARGET_OBJECTS:mtmd>)
set_target_properties(mtmd_shared PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(mtmd_shared PRIVATE LLAMA_BUILD)
target_link_libraries(mtmd_shared PRIVATE ggml llama Threads::Threads)
install(TARGETS mtmd_shared ARCHIVE)


# --- Main Library Target ---
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_compile_options(-fobjc-arc)
    add_library(${PROJECT_NAME} SHARED
        src/whisper_transcription.cc
        src/llama_device_base.cc
        src/synthesis.cc
        src/whillats.cc
        src/whillats_utils.cc
        src/whillats_synth.mm
        src/whillats_ios.mm
    )

    # Ensure espeak data is copied before building the main library (Linux only)
    if(UNIX AND NOT APPLE)
        add_dependencies(${PROJECT_NAME} copy_espeak_data)
    endif()

    # Add compile definition for espeak data path
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        ESPEAK_DATA_PATH="\"${ESPEAK_DATA_DEST_DIR}\""
    )

    # Add additional include directories
    target_include_directories(${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/include
            ${THIRD_PARTY_DIR}/llama.cpp/include  # Needed for llama.h
            ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd  # Added for clip.h
    )

    # Link core libraries (MetalOmitted: llama is pulled in by mtmd_shared)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            whisper
            mtmd_shared
    )

    # Link espeak-ng and pcaudio only on Linux
    if(UNIX AND NOT APPLE)
        target_link_libraries(${PROJECT_NAME}
            PRIVATE
                espeak-ng
                pcaudio
        )
    endif()

    # Platform-specific link options
    if(APPLE)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "-framework Accelerate"
            "-framework Metal"
            "-framework MetalKit"
            "-framework CoreAudio"
            "-framework AudioToolbox"
            "-framework AVFoundation"
        )
    elseif(UNIX)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            gomp
            stdc++
        )
    endif()

endif()  # end if(NOT iOS)

# --- iOS Framework Build ---
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(STATUS "Configuring for iOS framework build")
    # Build whisper for iOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static whisper build" FORCE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_STATIC ON CACHE BOOL "Force static whisper build" FORCE)
    # Skip building whisper here to avoid binary dir conflict
    # add_subdirectory(${THIRD_PARTY_DIR}/whisper.cpp ${CMAKE_BINARY_DIR}/_deps/whisper-build EXCLUDE_FROM_ALL)
    # Build llama for iOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static llama build" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    # Build llama inline for iOS
    add_subdirectory(${THIRD_PARTY_DIR}/llama.cpp ${CMAKE_BINARY_DIR}/_deps/llama-ios-build EXCLUDE_FROM_ALL)
    # Build mtmd from source (no FetchContent) as object and static library
    add_library(mtmd_obj OBJECT
        "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd.cpp"
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-helper.cpp
        "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/clip.cpp"
    )
    target_include_directories(mtmd_obj PRIVATE
        "${THIRD_PARTY_DIR}/llama.cpp/include"
        "${THIRD_PARTY_DIR}/llama.cpp/common"
        "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"
    )
    set_target_properties(mtmd_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
    target_link_libraries(${PROJECT_NAME} PRIVATE whisper llama $<TARGET_OBJECTS:mtmd_obj>
        "-framework AVFoundation" "-framework Foundation" "-framework Accelerate"
        "-framework Metal" "-framework MetalKit" "-framework AudioToolbox"
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd # Added for clip.h
    )
endif()  # end if(iOS)

# Enable debug symbols for whillats main library in Debug builds
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
)
target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g>
)

# Add after project() declaration, before other settings
if(APPLE)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif()

# Modify the target compile definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
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
        "-framework AVFoundation"
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

# Add test executable
add_executable(test_whillats
    test/test_whillats.cc
    test/test_utils.cc
)

# Add compile definition for espeak data path to test executable as well
target_include_directories(test_whillats PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(test_whillats PRIVATE
    ESPEAK_DATA_PATH="\"${ESPEAK_DATA_DEST_DIR}\"")

target_link_libraries(test_whillats
    PRIVATE
        ${PROJECT_NAME}
)

target_compile_options(test_whillats PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
)
target_link_options(test_whillats PRIVATE
    $<$<CONFIG:Debug>:-g>
)

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
# Add synthesis process executable
add_executable(synthesis
    src/synthesis.mm
)
# Link Objective-C++ frameworks for synthesis
target_link_libraries(synthesis
    PRIVATE
        "-framework AppKit"
        "-framework Foundation"
        "-framework AVFoundation"
)
set_target_properties(synthesis PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
)

# Keep macOS RPATH settings
    # Ensure test executable loads libwhillats.dylib from ../lib and ../lib/<config>
    set_target_properties(test_whillats PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path/../../lib;@loader_path/../../lib/$<CONFIG>"
    )
endif()

# Final install block: framework-only on iOS, full on others
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    install(TARGETS ${PROJECT_NAME}
        FRAMEWORK DESTINATION bin
    )
else()
    install(TARGETS ${PROJECT_NAME} test_whillats
        FRAMEWORK DESTINATION bin
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
    )
    install(TARGETS ${PROJECT_NAME} synthesis
        FRAMEWORK DESTINATION bin
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

# Add compile definition for espeak data path to test executable as well
target_include_directories(test_whillats
    PRIVATE
        ${THIRD_PARTY_DIR}/whisper.cpp/include
        ${THIRD_PARTY_DIR}/whisper.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/include
        ${THIRD_PARTY_DIR}/llama.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
)