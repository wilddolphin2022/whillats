#!/bin/sh

set -e

usage()
{
    echo 'usage: third_party [-d|-r|-c]
    where:
        -d to create a debug build default)
        -r to create a release build
        -c to clean the build artifacts'
}

clean()
{
    # Remove all possible artifact directories.
    if [ -f ${THIRD_PARTY}/whisper.cpp/CMakeLists.txt ]; then
    cd ${THIRD_PARTY}/whisper.cpp
    echo "rm -rf build in third_party/whisper.cpp"
    fi
    rm -rf build
    if [ -f ${THIRD_PARTY}/llama.cpp/CMakeLists.txt ]; then
    cd ${THIRD_PARTY}/llama.cpp
    echo "rm -rf build in third_party/llama.cpp "
    fi
    rm -rf build
    if [ -f ${THIRD_PARTY}/espeak-ng/CMakeLists.txt ]; then
    cd ${THIRD_PARTY}/espeak-ng
    echo "rm -rf build in third_party/espeak-ng"
    fi
    rm -rf build
    if [ -f ${THIRD_PARTY}/pcaudiolib/Makefile.am ]; then
    cd ${THIRD_PARTY}/pcaudiolib
    echo "make clean in third_party/pcaudiolib"
    make clean
    fi
}

THIRD_PARTY="${PWD}/third_party"

case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
    linux)
        HOST_PLATFORM="linux"
        ;;
    msys*|mingw*)
        HOST_PLATFORM="windows"
        ;;
    darwin)
        HOST_PLATFORM="mac"
        ;;
    *)
        HOST_PLATFORM="unknown"
        ;;
esac

BUILD_TYPE=debug

while [ "$1" != "" ]; do
    case $1 in
        -d | --debug )
            BUILD_TYPE=debug
            ;;
        -r | --release )
            BUILD_TYPE=release
            ;;
        -c | --clean )
            clean
            exit
            ;;
        -h | --help )
            usage
            exit
            ;;
        * )
            usage
            exit 1
    esac
    shift
done

    echo "HOST_PLATFORM is ${HOST_PLATFORM}, BUILD_TYPE is ${BUILD_TYPE}, THIRD_PARTY is ${THIRD_PARTY}"

    cd ${THIRD_PARTY}
    if [ ! -f ${THIRD_PARTY}/whisper.cpp/CMakeLists.txt ]; then
    echo "cloning whisper.cpp to ${PWD}"
    git clone https://github.com/ggerganov/whisper.cpp
    fi

    if [ ! -f ${THIRD_PARTY}/llama.cpp/CMakeLists.txt ]; then
    echo "cloning llama.cpp to ${PWD}"
    git clone https://github.com/ggerganov/llama.cpp
    fi

    if [ ! -f ${THIRD_PARTY}/espeak-ng/CMakeLists.txt ]; then
    echo "cloning espeak-ng to ${PWD}"
    git clone https://github.com/espeak-ng/espeak-ng
    fi

    if [ ! -f ${THIRD_PARTY}/pcaudiolib/Makefile.am ]; then
    echo "cloning pcaudiolib to ${PWD}"
    git clone https://github.com/espeak-ng/pcaudiolib
    fi

    cd ${THIRD_PARTY}/whisper.cpp
    
    if [ "${HOST_PLATFORM}" = "linux" ]
    then
        sed -i 's/Wunreachable-code-break/Wunreachable-code/g' ggml/src/CMakeLists.txt 
        sed -i 's/Wunreachable-code-return/Wunreachable-code/g' ggml/src/CMakeLists.txt
    fi

    echo "building whisper.cpp"
    cmake -B build

    if [ "${BUILD_TYPE}" = "release" ]
        cmake --build build --config Release  
    then
        cmake --build build --config Debug  
    fi

    if [ "${HOST_PLATFORM}" = "linux" ]
    then
        echo "installing whisper.cpp"
        cd build; sudo make install; cd ..
    fi

    cd ${THIRD_PARTY}/llama.cpp

    if [ "${HOST_PLATFORM}" = "linux" ]
    then
        sed -i 's/Wunreachable-code-break/Wunreachable-code/g' ggml/src/CMakeLists.txt 
        sed -i 's/Wunreachable-code-return/Wunreachable-code/g' ggml/src/CMakeLists.txt
    fi

    echo "building llama.cpp"
    cmake -B build

    if [ "${BUILD_TYPE}" = "release" ]
        cmake --build build --config Release  
    then
        cmake --build build --config Debug  
    fi

    if [ "${HOST_PLATFORM}" = "linux" ]
    then
        echo "installing llama.cpp"
        cd build; sudo make install; cd ..
    fi

    cd ${THIRD_PARTY}/espeak-ng

    echo "building espeak-ng"

    cmake -B build

    if [ "${BUILD_TYPE}" = "release" ]
        cmake --build build --config Release  
    then
        cmake --build build --config Debug  
    fi

    if [ "${HOST_PLATFORM}" = "linux" ]
    then
        echo "installing espeak-ng"
        sudo make install
    fi

    echo "building pcaudio"

    cd ${THIRD_PARTY}/pcaudiolib

    if [ "${HOST_PLATFORM}" = "mac" ]
    then
        echo "building pcaudiolib"
        ./autogen.sh
        ./configure
        make
        ./libtool --mode=install cp src/libpcaudio.la  ${THIRD_PARTY}/pcaudiolib/src/libpcaudio.dylib
    fi

