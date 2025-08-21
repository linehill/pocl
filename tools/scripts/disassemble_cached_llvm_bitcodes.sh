#! /bin/bash

# Script for automatically disassembling all the parallel.bc bitcode files
# found in the cache.
#
# Requires:
# 1) POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1
# 2) POCL_KERNEL_CACHE=1 (if not default)
# to have the parallel.bc dumps.
#
# USAGE:
# disassemble_cached_llvm_bitcodes.sh <llvm-dis> <cache-dir> <output-dir>
# NOTE! output-dir HAS to exist!

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <arg1> <arg2> <arg3>"
  exit 1
fi

LLVM_DIS=$1
CACHE_DIR=$2
OUTPUT_DIR=$3

if ! command -v $LLVM_DIS &> /dev/null; then
    echo "llvm-dis not found in PATH!"
    exit 1
else
    echo "Found $LLVM_DIS"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory '$OUTPUT_DIR' does not exist!"
    exit 1
else
    echo "Using $OUTPUT_DIR as ouput directory."
fi

if [ ! -d "$CACHE_DIR" ]; then
    echo "Cache directory '$CACHE_DIR' does not exist!"
    exit 1
else
    echo "Looking at $CACHE_DIR"
fi

find "$CACHE_DIR" -type f -name "parallel.bc" | while read -r bcfile; do
  echo "Disassembling: $bcfile"
  rel_path="${bcfile#*kcache/}"
  mod_path="${rel_path//\//-}"
  output_file="${mod_path%.bc}.ll"
  echo "Outputting: $OUTPUT_DIR/$output_file"

  $LLVM_DIS "$bcfile" -o "$OUTPUT_DIR/$output_file"
done