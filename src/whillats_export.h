#ifndef WHILLATS_EXPORT_H
#define WHILLATS_EXPORT_H

// Include TargetConditionals for TARGET_OS_IOS macro
#if defined(__APPLE__)
    #include <TargetConditionals.h>
    // Exclude TTS (espeak-ng) for iOS builds
    #if  TARGET_OS_IOS
        #define TTS_PLATFORMS 0 // Building for iOS
    #else
        #define TTS_PLATFORMS 1 // Building for macOS or other non-iOS platforms
    #endif
#else
    #define TTS_PLATFORMS 1 // Building for other platforms
#endif

#if defined(_MSC_VER)
    #define WHILLATS_EXPORT __declspec(dllexport)
    #define WHILLATS_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    #define WHILLATS_EXPORT __attribute__((visibility("default")))
    #define WHILLATS_IMPORT __attribute__((visibility("default")))
#else
    #define WHILLATS_EXPORT
    #define WHILLATS_IMPORT
#endif

#ifdef WHILLATS_BUILDING_DLL
    #define WHILLATS_API WHILLATS_EXPORT
#else
    #define WHILLATS_API WHILLATS_IMPORT
#endif

#endif // WHILLATS_EXPORT_H 