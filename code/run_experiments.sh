#!/bin/bash

# Shell script for running probabilistic prediction performance evaluation experiments
#
# Pre-requisites:
# - Option 1 [Recommended]: Run Python code inside a docker container
#                           Please refer to ../docker/README.md
# - Option 2: Install python packages on your local machine
#   This typically requires the following steps as a minimum
#   $ cd /home/$USER/experiments/
#   $ /usr/local/bin/python3.8 -m venv python3.8
#   $ source python3.8/bin/activate
#   $ pip install --upgrade pip
#   $ pip install matplotlib jupyter numpy scipy scikit-learn scikit-gstat pandas tqdm
#
# Commands for running the experiments:
# - For a single inference period and geological domain (e.g. mA=4, gD=2310) with nS=64 simulations
#   $ python -m run_experiments $mA $gD $nS
# - For a series of experiments, iterating over (mA, gD) pairs using this bash script
#   $ ./run_experiments.sh > run_experiments_yyyymmdd.log
#
# Command to terminate running processes:
#   $ ps aux | grep -i run_experiments | awk '{print $2}' | xargs kill
#
# SPDX-FileCopyrightText: 2024 Raymond Leung <raymond.leung@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------------------------

# Configure the following paths
PYTHON_VENV_DIR="/home/$USER/experiments/python3.8"
PYTHON_CMD="python"
PYTHON_APP="run_experiments"
REPO_SOURCE_DIR="/home/$USER/experiments/eup3m"
CODE_DIR="${REPO_SOURCE_DIR}/code"
ARCHIVE_DIR="${REPO_SOURCE_DIR}/archive"
RESULTS_DIR="${REPO_SOURCE_DIR}/results"
PARAMS_DIR="${REPO_SOURCE_DIR}/data"
LOGS_DIR="${RESULTS_DIR}/z-logs"

num_simulations=128
max_processes=28 #$((`nproc --all` / 2))
active_processes=0

# Configure experiment as one of ["future-bench-prediction", "in-situ-regression"]
inference_type="future-bench-prediction"

# (Optional) Remove learned parameters to start from a clean slate
rm -f ${PARAMS_DIR}/gstatsim3d_optimised_*parameters*.csv

declare -a DomainArray=(2210 2310 3016 3026 3110 3121 3210 3221 3310 3321 3521)
declare -a InferencePrefixArray=(4 5 6 7 8 9 10 11 12 13 14 15)
declare -a LearnInRotatedSpaceArray=(True)

task_count=0
task_total=$((${#DomainArray[@]} * ${#InferencePrefixArray[@]} * ${#LearnInRotatedSpaceArray[@]}))

# When the USER variable is set, the python virtual environment is activated assuming
# this script is not run inside a Docker container (otherwise USER would be NULL)
if [ -n "$USER" ]; then
    source ${PYTHON_VENV_DIR}/bin/activate
fi

mkdir -p ${RESULTS_DIR}
mkdir -p ${LOGS_DIR}

echo "Installed `${PYTHON_CMD} --version`"
echo "Experiment parameters:"
echo "- mA = inference period <int>"
echo "- gd = geological domain id <int>"
echo "- lirf = learning in rotated frame <bool>\n"
echo ""
echo "Bash script starts at `date '+%Y-%m-%d %H:%M:%S'`"
echo ""
echo ${max_processes} > max_processes.txt

for mA in "${!InferencePrefixArray[@]}"; do
  inference_prefix=${InferencePrefixArray[$mA]}

  for gd in "${!DomainArray[@]}"; do
    domain_id=${DomainArray[$gd]}

    for rs in "${!LearnInRotatedSpaceArray[@]}"; do
      learning_rotated=${LearnInRotatedSpaceArray[$rs]}

      first_encounter=true
      while [[ ${active_processes} -ge ${max_processes} ]]; do

        sleep 20
        active_processes=$((`ps aux | grep "${PYTHON_CMD} -m ${PYTHON_APP}" | wc -l` - 1))
        if [ ${first_encounter} = true ]; then
          echo "  active_processes=${active_processes}, waiting for job to finish..."
          first_encounter=false
        fi
        max_processes=$(< max_processes.txt)

      done

      if [ ${first_encounter} = false ]; then
        echo "  active_processes=${active_processes}, resuming task..."
      fi

      logfile=$(printf "log%03d-${inference_prefix}_${domain_id}_${rs}.txt" $task_count)

      ${PYTHON_CMD} -m ${PYTHON_APP} \
          ${inference_prefix} ${domain_id} ${num_simulations} \
          "inference_type=${inference_type}" \
          "kriging:transform_data=${learning_rotated}" \
          "gp:learning_inference_in_rotated_space=${learning_rotated}" &> "${LOGS_DIR}/${logfile}" &
      task_count=$((${task_count} + 1))
      echo "[${task_count}/${task_total}] Running for mA=${inference_prefix}, gd=${domain_id}, lirf=${learning_rotated}"
      sleep 1
      active_processes=$((`ps aux | grep "${PYTHON_CMD} -m ${PYTHON_APP}" | wc -l` - 1))

    done
  done
done

echo "Finished running `basename $0`"
echo "Waiting for asynchronous processes to be completed..."

while [[ ${active_processes} -gt 0 ]]; do

  sleep 20
  active_processes=$((`ps aux | grep "${PYTHON_CMD} -m ${PYTHON_APP}" | wc -l` - 1))

done

datetime=`date '+%Y%m%d-%H%M%S'`
echo "All processes have now completed!"
echo ""

DESTINATION_DIR="${ARCHIVE_DIR}/${datetime}"
mkdir -p ${DESTINATION_DIR}/code

echo "Archiving results in ${DESTINATION_DIR}"
echo ""
cp ${PARAMS_DIR}/gstatsim3d_optimised_*parameters*.csv ${RESULTS_DIR}
cp -r ${RESULTS_DIR}/* ${DESTINATION_DIR}

echo "Archiving code in ${DESTINATION_DIR}/code"
cp ${CODE_DIR}/* ${DESTINATION_DIR}/code 2>&1 | grep -v 'omitting directory'

echo "Bash script finishes at ${datetime}"
echo ""
