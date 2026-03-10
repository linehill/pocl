#!/bin/bash
#
# Copyright (c) 2026 Tapio Nevalainen / Tampere University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# Executes GROMACS simulation with given input (Cold run for kernel
# compilation and proper longer run). Runs python script to produce a JSON that
# contains execution times of the run.

# Immediately exit with non-zero status.
set -e

# Path to input
SAMPLE_DIR=$1
# Path to gmx executable
GMX=$2/bin/gmx

# Path to python script
SCRIPT_DIR=$3
# Input name
INPUT_NAME=$4

# Output stack info
# $GMX --version
# acpp-info
# clinfo -l

# Number of steps to run the GROMACS md
# For cold start to get the kernels compiled
N_STEPS_COLD=200
# Actual run
N_STEPS=1000

# Allow GROMACS to utilize other than GPU devices
export GMX_SYCL_ALLOW_ALL_DEVICES=1

# Force subgroup size to 32 and disable rematerialization (until fixed)
export POCL_SUB_GROUP_SIZE=32
export POCL_PREGION_VALUE_REMAT=0
export POCL_CPU_MAX_CU_COUNT=1

# Cold run to get the kernels compiled
$GMX mdrun -s ${SAMPLE_DIR}/${INPUT_NAME}.tpr -ntmpi 1 -ntomp 1 -pin on -nb gpu -bonded cpu -update cpu -nsteps ${N_STEPS_COLD} -nobackup -noconfout -resethway -g ${SAMPLE_DIR}/${INPUT_NAME} -v

# Actual run, this will overwrite the log file.
$GMX mdrun -s ${SAMPLE_DIR}/${INPUT_NAME} -ntmpi 1 -ntomp 1 -pin on -nb gpu -bonded cpu -update cpu -nsteps ${N_STEPS} -nobackup -noconfout -resethway -g ${SAMPLE_DIR}/${INPUT_NAME} -v

# Parameters are:
# 1. Path to log-file
# 2. Path to directory where the result json should be written
${SCRIPT_DIR}/extract_gmx_results.py ${SAMPLE_DIR}/${INPUT_NAME}.log ${SAMPLE_DIR}/${INPUT_NAME}
