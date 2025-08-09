/*
 *  (c) 2025, wilddolphin2022 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

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
