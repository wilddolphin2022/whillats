cmake_minimum_required(VERSION 3.14)
project(whillats VERSION 1.0.0 LANGUAGES CXX C)

# Ensure Position Independent Code is used for all targets
# Necessary when linking static libs into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(ExternalProject) # Needed for ExternalProject_Add

# Set output directories - all to bin (MOVE THIS TO THE TOP)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Additional compiler and linker flags
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wno-incompatible-pointer-types-discards-qualifiers)
add_compile_options(-Wno-exit-time-destructors -Wno-global-constructors)

# --- Dependency Declarations ---
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
file(MAKE_DIRECTORY ${THIRD_PARTY_DIR}) # Ensure third_party exists

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp
    SOURCE_DIR ${THIRD_PARTY_DIR}/whisper.cpp
    GIT_SHALLOW TRUE # Faster clone
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
FetchContent_MakeAvailable(whisper)

# llama.cpp (CMake)
set(LLAMA_STATIC ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(UNIX AND NOT APPLE) # Linux specific
    set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
else()
    set(GGML_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
endif()
FetchContent_MakeAvailable(llama)

# espeak-ng (CMake)
# Note: espeak-ng requires pcaudiolib. We need to ensure pcaudiolib is built first
# or handle the dependency within espeak-ng's CMake.
# Let's build pcaudiolib first.

# pcaudiolib (Autotools)
FetchContent_GetProperties(pcaudiolib-proj)
if(NOT pcaudiolib-proj_POPULATED)
    # Set policy to suppress FetchContent_Populate deprecation warning
    cmake_policy(SET CMP0169 OLD)
    FetchContent_Populate(pcaudiolib-proj) # Uses source dir: ${THIRD_PARTY_DIR}/pcaudiolib
    cmake_policy(SET CMP0169 NEW) # Revert policy if needed elsewhere, though likely not
    message(STATUS "pcaudiolib source downloaded to ${pcaudiolib-proj_SOURCE_DIR}")
endif()

set(PCAUDIOLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/pcaudiolib-build)
set(PCAUDIOLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/pcaudiolib-install) # Define install dir for prefix
set(PCAUDIOLIB_LIB_PATH ${PCAUDIOLIB_BUILD_DIR}/src/.libs/libpcaudio.a)

# Use execute_process to run autogen.sh from the source directory first,
# as ExternalProject_Add CONFIGURE_COMMAND runs in BINARY_DIR.
# This ensures ./configure can find necessary files created by autogen.sh.
message(STATUS "Running autogen.sh for pcaudiolib...")
execute_process(
    COMMAND sh ./autogen.sh
    WORKING_DIRECTORY ${pcaudiolib-proj_SOURCE_DIR}
    RESULT_VARIABLE _autogen_res
    OUTPUT_QUIET ERROR_QUIET # Be less verbose
)
if(NOT _autogen_res EQUAL 0)
    message(FATAL_ERROR "autogen.sh for pcaudiolib failed: ${_autogen_res}")
endif()
message(STATUS "autogen.sh for pcaudiolib completed.")


ExternalProject_Add(pcaudiolib_external
    SOURCE_DIR ${pcaudiolib-proj_SOURCE_DIR}
    BINARY_DIR ${PCAUDIOLIB_BUILD_DIR}
    # CONFIGURE_COMMAND runs ./configure from the source dir in the build dir
    CONFIGURE_COMMAND bash -c "export CFLAGS='-fPIC -I${pcaudiolib-proj_SOURCE_DIR}/src/include' && export CXXFLAGS='-fPIC -I${pcaudiolib-proj_SOURCE_DIR}/src/include' && exec <SOURCE_DIR>/configure --with-pic --prefix='${PCAUDIOLIB_INSTALL_DIR}'"
    BUILD_COMMAND $(MAKE) # Use CMake's make variable
    INSTALL_COMMAND "" # Don't install system-wide
    BUILD_BYPRODUCTS ${PCAUDIOLIB_LIB_PATH} # Tell CMake about the output file
    LOG_CONFIGURE YES # Log steps for debugging
    LOG_BUILD YES
)

# Create an IMPORTED target for pcaudio
add_library(pcaudio STATIC IMPORTED)
set_target_properties(pcaudio PROPERTIES
    IMPORTED_LOCATION ${PCAUDIOLIB_LIB_PATH}
    INTERFACE_INCLUDE_DIRECTORIES ${pcaudiolib-proj_SOURCE_DIR}/src/include
)
# Add dependency: pcaudio depends on the external project completing
add_dependencies(pcaudio pcaudiolib_external)


# espeak-ng (CMake) - Now build espeak-ng after pcaudiolib external project is defined
# Set CMake options for espeak-ng if necessary, e.g., point to pcaudio include/lib
# We might need to pass PCAUDIO_INCLUDE_DIR and PCAUDIO_LIBRARY to espeak-ng's CMake.
# Let's assume espeak-ng CMake can find pcaudio if built first, or adjust later if needed.

# Ensure espeak-ng uses the correct C compiler found by the parent project
set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "C compiler" FORCE)

# Explicitly add standard paths to CMake's search path for programs
list(APPEND CMAKE_PROGRAM_PATH /usr/bin /bin)
set(CMAKE_PROGRAM_PATH ${CMAKE_PROGRAM_PATH} CACHE STRING "Path for CMake to search for programs" FORCE)

# Set options and make espeak-ng available (builds it)
FetchContent_MakeAvailable(espeak-ng-proj) # Uses source dir: ${THIRD_PARTY_DIR}/espeak-ng

# Define paths for espeak-ng data
set(ESPEAK_DATA_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/espeak-ng-proj-build/espeak-ng-data)
set(ESPEAK_DATA_DEST_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/espeak-ng-data)

# Custom target to copy espeak-ng data after it's built
add_custom_command(
    OUTPUT ${ESPEAK_DATA_DEST_DIR}/phontab # Check for a representative file
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${ESPEAK_DATA_SOURCE_DIR} ${ESPEAK_DATA_DEST_DIR}
    DEPENDS espeak-ng # Ensure espeak-ng (and its data) is built first
    COMMENT "Copying espeak-ng-data to ${ESPEAK_DATA_DEST_DIR}"
)
add_custom_target(copy_espeak_data ALL DEPENDS ${ESPEAK_DATA_DEST_DIR}/phontab)

# Create library target
add_library(${PROJECT_NAME} SHARED
    src/whisper_transcription.cc
    src/llama_device_base.cc
    src/espeak_tts.cc
    src/whillats.cc
)

# Ensure espeak data is copied before building the main library
add_dependencies(${PROJECT_NAME} copy_espeak_data)

# Add compile definition for espeak data path
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ESPEAK_DATA_PATH="\"${ESPEAK_DATA_DEST_DIR}\"")

# Add additional include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE # Changed from PUBLIC to potentially help include path resolution
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}/_deps/llama-build/include # Use build path for llama includes
        ${CMAKE_BINARY_DIR}/_deps/whisper-build/include # Use build path for whisper includes
        ${CMAKE_BINARY_DIR}/_deps/whisper-build/ggml/include # Use build path for ggml includes
        # Use espeak-ng source paths for its includes
        ${espeak-ng-proj_SOURCE_DIR}/src/include 
        ${espeak-ng-proj_SOURCE_DIR}/src/ucd-tools/src/include
        ${pcaudiolib-proj_SOURCE_DIR}/src/include # pcaudiolib source path for includes
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# Link directories no longer needed

target_link_libraries(${PROJECT_NAME}
    # Link PRIVATE to prevent system libs from being preferred transitively
    PRIVATE
        whisper      # Link against main whisper target (should be static)
        llama        # Link against main llama target (should be static)
        espeak-ng    # Link against espeak-ng target (built static by default)
        # speechPlayer # Only if needed
        # ucd          # Only if needed
        pcaudio      # Link against our imported pcaudio target (static)
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

# Add compile definition for espeak data path to test executable as well
target_compile_definitions(test_whillats PRIVATE
    ESPEAK_DATA_PATH="\"${ESPEAK_DATA_DEST_DIR}\"")

target_link_libraries(test_whillats
    PRIVATE
        ${PROJECT_NAME}
)

# Keep macOS RPATH settings
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

# Make Agora SDK available (downloads and extracts) (Only for Linux)
if(UNIX AND NOT APPLE)
    FetchContent_MakeAvailable(agora_sdk)
    message(STATUS "Agora SDK source dir: ${agora_sdk_SOURCE_DIR}")
endif()

# Ensure examples are processed after main library and dependencies (Only for Linux)
if(UNIX AND NOT APPLE)
    add_subdirectory(example/agora-whillats-bot)
endif()

# --- Add patch for espeak-ng compat header ---
FetchContent_GetProperties(espeak-ng-proj)
if(NOT espeak-ng-proj_POPULATED)
    message(STATUS "Populating espeak-ng-proj to allow patching...")
    FetchContent_Populate(espeak-ng-proj)
endif()
set(ESPEAK_WCTYPE_H_PATH ${espeak-ng-proj_SOURCE_DIR}/src/include/compat/wctype.h)
if(EXISTS ${ESPEAK_WCTYPE_H_PATH})
    message(STATUS "Patching ${ESPEAK_WCTYPE_H_PATH} to prevent C++ macro pollution...")
    file(READ ${ESPEAK_WCTYPE_H_PATH} ESPEAK_WCTYPE_H_CONTENT)
    # Wrap the conflicting defines in #ifndef __cplusplus
    string(REPLACE "#define tolower ucd_tolower" "#ifndef __cplusplus\n#define tolower ucd_tolower\n#endif" ESPEAK_WCTYPE_H_CONTENT_PATCHED "${ESPEAK_WCTYPE_H_CONTENT}")
    string(REPLACE "#define toupper udc_toupper" "#ifndef __cplusplus\n#define toupper udc_toupper\n#endif" ESPEAK_WCTYPE_H_CONTENT_PATCHED "${ESPEAK_WCTYPE_H_CONTENT_PATCHED}")
    file(WRITE ${ESPEAK_WCTYPE_H_PATH} "${ESPEAK_WCTYPE_H_CONTENT_PATCHED}")
    message(STATUS "Patching wctype.h done.")
else()
    message(WARNING "Could not find ${ESPEAK_WCTYPE_H_PATH} to patch.")
endif()