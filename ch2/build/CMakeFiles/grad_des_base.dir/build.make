# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gfeng/gfeng_ws/Machine-Vision/ch2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gfeng/gfeng_ws/Machine-Vision/ch2/build

# Include any dependencies generated for this target.
include CMakeFiles/grad_des_base.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/grad_des_base.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/grad_des_base.dir/flags.make

CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o: CMakeFiles/grad_des_base.dir/flags.make
CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o: ../src/gradient_descent_base.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gfeng/gfeng_ws/Machine-Vision/ch2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o -c /home/gfeng/gfeng_ws/Machine-Vision/ch2/src/gradient_descent_base.cpp

CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gfeng/gfeng_ws/Machine-Vision/ch2/src/gradient_descent_base.cpp > CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.i

CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gfeng/gfeng_ws/Machine-Vision/ch2/src/gradient_descent_base.cpp -o CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.s

CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.requires:

.PHONY : CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.requires

CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.provides: CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.requires
	$(MAKE) -f CMakeFiles/grad_des_base.dir/build.make CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.provides.build
.PHONY : CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.provides

CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.provides.build: CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o


# Object files for target grad_des_base
grad_des_base_OBJECTS = \
"CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o"

# External object files for target grad_des_base
grad_des_base_EXTERNAL_OBJECTS =

libgrad_des_base.a: CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o
libgrad_des_base.a: CMakeFiles/grad_des_base.dir/build.make
libgrad_des_base.a: CMakeFiles/grad_des_base.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gfeng/gfeng_ws/Machine-Vision/ch2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libgrad_des_base.a"
	$(CMAKE_COMMAND) -P CMakeFiles/grad_des_base.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/grad_des_base.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/grad_des_base.dir/build: libgrad_des_base.a

.PHONY : CMakeFiles/grad_des_base.dir/build

CMakeFiles/grad_des_base.dir/requires: CMakeFiles/grad_des_base.dir/src/gradient_descent_base.cpp.o.requires

.PHONY : CMakeFiles/grad_des_base.dir/requires

CMakeFiles/grad_des_base.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/grad_des_base.dir/cmake_clean.cmake
.PHONY : CMakeFiles/grad_des_base.dir/clean

CMakeFiles/grad_des_base.dir/depend:
	cd /home/gfeng/gfeng_ws/Machine-Vision/ch2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gfeng/gfeng_ws/Machine-Vision/ch2 /home/gfeng/gfeng_ws/Machine-Vision/ch2 /home/gfeng/gfeng_ws/Machine-Vision/ch2/build /home/gfeng/gfeng_ws/Machine-Vision/ch2/build /home/gfeng/gfeng_ws/Machine-Vision/ch2/build/CMakeFiles/grad_des_base.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/grad_des_base.dir/depend

