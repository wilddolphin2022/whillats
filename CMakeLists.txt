cmake_minimum_required(VERSION 3.16)
project(whillats VERSION 1.0.0 LANGUAGES CXX C OBJCXX)

# Ensure Position Independent Code is used for all targets
# Necessary when linking static libs into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(ExternalProject) # Needed for ExternalProject_Add

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Changed to lib for consistency
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Changed to lib for consistency
set(CMAKE_FRAMEWORK_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Frameworks) # Output for frameworks

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

# Only declare non-Apple dependencies if not building for Apple
if(NOT APPLE)
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
endif() # NOT APPLE

# --- Populate Common Dependencies (Headers needed even on iOS) ---
FetchContent_GetProperties(whisper)
if(NOT whisper_POPULATED)
    FetchContent_Populate(whisper)
    message(STATUS "Whisper source downloaded to ${whisper_SOURCE_DIR}")
endif()

FetchContent_GetProperties(llama)
if(NOT llama_POPULATED)
    FetchContent_Populate(llama)
    message(STATUS "Llama source downloaded to ${llama_SOURCE_DIR}")
endif()

# --- Build Dependencies and Target (Non-Apple Platforms) ---
if(NOT APPLE)
    message(STATUS "Configuring for non-Apple platform")

    # --- Populate and Build Non-Apple Dependencies ---

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

    # espeak-ng & pcaudiolib (CMake & Autotools)
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

    message(STATUS "Running autogen.sh for pcaudiolib...")
    execute_process(
        COMMAND sh ./autogen.sh
        WORKING_DIRECTORY ${pcaudiolib-proj_SOURCE_DIR}
        RESULT_VARIABLE _autogen_res
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _autogen_res EQUAL 0)
        message(FATAL_ERROR "autogen.sh for pcaudiolib failed: ${_autogen_res}")
    endif()
    message(STATUS "autogen.sh for pcaudiolib completed.")

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

    # Ensure espeak-ng uses the correct C compiler
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
        DEPENDS espeak-ng
        COMMENT "Copying espeak-ng-data to ${ESPEAK_DATA_DEST_DIR}"
    )
    add_custom_target(copy_espeak_data ALL DEPENDS ${ESPEAK_DATA_DEST_DIR}/phontab)

    # --- Define whillats Target (Non-Apple) ---
    add_library(${PROJECT_NAME} SHARED
        src/whisper_transcription.cc
        src/llama_device_base.cc
        src/espeak_tts.cc # Only include for non-Apple
        src/whillats.cc
    )

    add_dependencies(${PROJECT_NAME} copy_espeak_data)

    target_compile_definitions(${PROJECT_NAME} PRIVATE
        ESPEAK_DATA_PATH="\"${ESPEAK_DATA_DEST_DIR}\""
        WHILLATS_BUILDING_DLL # For Windows DLL export/import if needed later
    )

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_BINARY_DIR}/include # Use build path for llama includes
        ${whisper_BINARY_DIR}/include # Use build path for whisper includes
        ${whisper_BINARY_DIR}/ggml/include # Use build path for ggml includes
        ${espeak-ng-proj_SOURCE_DIR}/src/include
        ${espeak-ng-proj_SOURCE_DIR}/src/ucd-tools/src/include
        ${pcaudiolib-proj_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE
        whisper      # Link against main whisper target (static)
        llama        # Link against main llama target (static)
        espeak-ng    # Link against espeak-ng target (static)
        pcaudio      # Link against imported pcaudio target (static)
    )

    if(UNIX AND NOT APPLE)
        target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_LINUX)
        target_link_libraries(${PROJECT_NAME} PRIVATE gomp stdc++)
    endif()

# --- Build Configuration for Apple Platforms (macOS, iOS) ---
else() # APPLE
    message(STATUS "Configuring for Apple platform: ${CMAKE_SYSTEM_NAME}")

    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

    # Assume XCFrameworks are pre-built in standard locations
    set(LLAMA_XCFRAMEWORK_PATH ${THIRD_PARTY_DIR}/llama.cpp/build-apple/llama.xcframework)
    set(WHISPER_XCFRAMEWORK_PATH ${THIRD_PARTY_DIR}/whisper.cpp/build-apple/whisper.xcframework)

    if(NOT EXISTS ${LLAMA_XCFRAMEWORK_PATH})
        message(WARNING "Llama XCFramework not found at ${LLAMA_XCFRAMEWORK_PATH}. Please build it using third_party/llama.cpp/build-xcframework.sh")
    endif()
    if(NOT EXISTS ${WHISPER_XCFRAMEWORK_PATH})
        message(WARNING "Whisper XCFramework not found at ${WHISPER_XCFRAMEWORK_PATH}. Please build it using third_party/whisper.cpp/build-xcframework.sh")
    endif()

    # --- Define whillats Target (Apple Framework) ---
    add_library(${PROJECT_NAME} SHARED
        src/whisper_transcription.cc
        src/llama_device_base.cc
        src/whillats.cc
        src/whillats_synth.mm
        src/whillats_ios.mm
    )

    set_target_properties(${PROJECT_NAME} PROPERTIES
        FRAMEWORK ON
        FRAMEWORK_VERSION A # Standard practice
        MACOSX_FRAMEWORK_IDENTIFIER ai.webrtcsays.whillats
        # Define PUBLIC_HEADER to list headers exposed by the framework
        # PUBLIC_HEADER "src/whillats.h;src/whillats_ios.h"
        # Remove headers from RESOURCES
        # RESOURCE "src/whillats.h;src/whillats_ios.h"
        XCODE_ATTRIBUTE_DEFINES_MODULE YES # Enable module map generation
        XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES YES
        XCODE_ATTRIBUTE_APPLICATION_EXTENSION_API_ONLY NO # Or YES if needed
        # Ensure ObjC ARC is enabled if needed (often default, but good to be explicit)
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        # Set deployment target (should ideally come from toolchain file)
        # CMAKE_OSX_DEPLOYMENT_TARGET "16.4" # Example for iOS
    )

    # Remove the previous set(FRAMEWORK_OUTPUT_PATH ...) if it exists
    # set(FRAMEWORK_OUTPUT_PATH $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}.framework)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        # Use direct CMake commands and variable substitution
        COMMAND ${CMAKE_COMMAND} -E echo "POST_BUILD: CMAKE_LIBRARY_OUTPUT_DIRECTORY = ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        COMMAND ${CMAKE_COMMAND} -E echo "POST_BUILD: PROJECT_NAME = ${PROJECT_NAME}"
        COMMAND ${CMAKE_COMMAND} -E echo "POST_BUILD: Creating Headers Dir = ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/${PROJECT_NAME}.framework/Headers"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/${PROJECT_NAME}.framework/Headers"
            COMMENT "Creating Headers directory in Release framework (direct)"
        COMMAND ${CMAKE_COMMAND} -E echo "POST_BUILD: Copying ${CMAKE_CURRENT_SOURCE_DIR}/src/whillats.h to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/${PROJECT_NAME}.framework/Headers/"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/src/whillats.h" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/${PROJECT_NAME}.framework/Headers/"
            COMMENT "Copying whillats.h to Release framework (direct)"
        #COMMAND ${CMAKE_COMMAND} -E echo "POST_BUILD: Copying ${CMAKE_CURRENT_SOURCE_DIR}/src/whillats_ios.h to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/${PROJECT_NAME}.framework/Headers/"
        #COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/src/whillats_ios.h" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/${PROJECT_NAME}.framework/Headers/"
        #    COMMENT "Copying whillats_ios.h to Release framework (direct)"
        VERBATIM
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE
        $<$<PLATFORM_ID:Darwin>:PLATFORM_DARWIN>
        $<$<PLATFORM_ID:iOS>:WEBRTC_IOS>
        # Add other necessary definitions for Apple platforms
    )

    target_include_directories(${PROJECT_NAME} PUBLIC # Headers for the framework interface
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
     PRIVATE # Internal include paths
        ${llama_SOURCE_DIR}/include
        ${llama_SOURCE_DIR}/ggml/include
        ${whisper_SOURCE_DIR}/include
        ${whisper_SOURCE_DIR}/ggml/include
    )

    # Link only if the XCFrameworks actually exist (they might not on the first configure pass)
    if(EXISTS ${LLAMA_XCFRAMEWORK_PATH} AND EXISTS ${WHISPER_XCFRAMEWORK_PATH})
        target_link_libraries(${PROJECT_NAME} PRIVATE
            # Link against the pre-built XCFrameworks
            ${LLAMA_XCFRAMEWORK_PATH}
            ${WHISPER_XCFRAMEWORK_PATH}
            # Link system frameworks
            "-framework Foundation"
            "-framework Accelerate"
            "-framework Metal"
            "-framework MetalKit" # Often needed with Metal
            # Add AVFoundation or AudioToolbox if audio processing is done within whillats itself
            "-framework AVFoundation"
        )
    else()
        message(STATUS "Skipping XCFramework linking because paths not found (will retry on next configure)")
    endif()

    # Set RPATH so the framework can find the embedded XCFrameworks if needed
    # Usually handled by Xcode build process when linking frameworks
    # set_target_properties(${PROJECT_NAME} PROPERTIES
    #     XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@loader_path/Frameworks"
    # )

endif() # APPLE / NOT APPLE

# --- Common Post-Build Steps (If any) ---
# (e.g., Installation rules - potentially platform-specific)

message(STATUS "CMake Project Name: ${PROJECT_NAME}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Apple Target: ${APPLE}")
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(IS_IOS_TARGET TRUE)
else()
    set(IS_IOS_TARGET FALSE)
endif()
message(STATUS "iOS Target: ${IS_IOS_TARGET}")
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