#!/usr/bin/env bash
#
# Runs Clang tidy on selected files to address style-related guidelines.
# Not yet ran on all files to avoid massive non-functional change commits
# polluting the commit history.

SCRIPTPATH=$( realpath "$0"  )
SCRIPTDIR=$(dirname "$SCRIPTPATH")

if [ ! -f config.h -o ! -f ../lib/llvmopencl/WorkitemLoops.cc ];
then
    echo "Must be ran in a build dir located inside the src dir."
    exit 1
fi

SRCROOT=$(git rev-parse --show-toplevel 2>/dev/null)

clang-tidy -extra-arg=-Wno-unknown-warning-option --config-file=$SCRIPTDIR/clang-tidy-llvm.config --fix --format-style=llvm \
../lib/llvmopencl/BarrierTailReplication.cc \
../lib/llvmopencl/CanonicalizeBarriers.cc \
../lib/llvmopencl/DeSPMD.cpp \
../lib/llvmopencl/ImplicitConditionalBarriers.cc \
../lib/llvmopencl/ImplicitLoopBarriers.cc \
../lib/llvmopencl/LoopBarriers.cc \
../lib/llvmopencl/WorkitemLoops.cc
