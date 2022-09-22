# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/cvte-vm/Downloads/cmake-3.22.1-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/cvte-vm/Downloads/cmake-3.22.1-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/build

# Include any dependencies generated for this target.
include CMakeFiles/SNPE_superpoint.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SNPE_superpoint.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SNPE_superpoint.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SNPE_superpoint.dir/flags.make

CMakeFiles/SNPE_superpoint.dir/main.cpp.o: CMakeFiles/SNPE_superpoint.dir/flags.make
CMakeFiles/SNPE_superpoint.dir/main.cpp.o: ../main.cpp
CMakeFiles/SNPE_superpoint.dir/main.cpp.o: CMakeFiles/SNPE_superpoint.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SNPE_superpoint.dir/main.cpp.o"
	/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SNPE_superpoint.dir/main.cpp.o -MF CMakeFiles/SNPE_superpoint.dir/main.cpp.o.d -o CMakeFiles/SNPE_superpoint.dir/main.cpp.o -c /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/main.cpp

CMakeFiles/SNPE_superpoint.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SNPE_superpoint.dir/main.cpp.i"
	/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/main.cpp > CMakeFiles/SNPE_superpoint.dir/main.cpp.i

CMakeFiles/SNPE_superpoint.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SNPE_superpoint.dir/main.cpp.s"
	/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/cvte-vm/Downloads/unify_ndk-master/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/main.cpp -o CMakeFiles/SNPE_superpoint.dir/main.cpp.s

# Object files for target SNPE_superpoint
SNPE_superpoint_OBJECTS = \
"CMakeFiles/SNPE_superpoint.dir/main.cpp.o"

# External object files for target SNPE_superpoint
SNPE_superpoint_EXTERNAL_OBJECTS =

android/bin/SNPE_superpoint: CMakeFiles/SNPE_superpoint.dir/main.cpp.o
android/bin/SNPE_superpoint: CMakeFiles/SNPE_superpoint.dir/build.make
android/bin/SNPE_superpoint: android/library/libsp_SNPE_lib.so
android/bin/SNPE_superpoint: CMakeFiles/SNPE_superpoint.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable android/bin/SNPE_superpoint"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SNPE_superpoint.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SNPE_superpoint.dir/build: android/bin/SNPE_superpoint
.PHONY : CMakeFiles/SNPE_superpoint.dir/build

CMakeFiles/SNPE_superpoint.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SNPE_superpoint.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SNPE_superpoint.dir/clean

CMakeFiles/SNPE_superpoint.dir/depend:
	cd /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/build /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/build /home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/build/CMakeFiles/SNPE_superpoint.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SNPE_superpoint.dir/depend
