#!/bin/sh
cd /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation
mkdir -p build
cd ./build
echo $(pwd)
rm -r *

while getopts ":f:" opt
do
    case $opt in
        f)
        echo "arch = $OPTARG"
        SNPE_ARCH=$OPTARG
        ;;
        ?)
        echo "usage：sh build.sh -f android64
            -f -> compile arch : android64/x64"
        exit 1;;
    esac
done

ANDROID_NDK=/home/cvte-vm/Downloads/unify_ndk-master
CMAKE_TOOLCHAIN=$ANDROID_NDK/build/cmake/android.toolchain.cmake

if [ "$SNPE_ARCH" = "android64" ]
then
    cmake -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN \
	-DANDROID_ABI="arm64-v8a" \
	-DANDROID_NDK=$ANDROID_NDK \
	-DANDROID_PLATFORM=android-24 \
    ..
    make -j16
else
    cmake ..
    make -j16
fi
