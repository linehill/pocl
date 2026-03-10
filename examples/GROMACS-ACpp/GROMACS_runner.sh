#!/bin/bash

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
