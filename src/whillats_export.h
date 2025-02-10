#ifndef WHILLATS_EXPORT_H
#define WHILLATS_EXPORT_H

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