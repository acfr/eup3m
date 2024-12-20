#!/usr/bin/env python
# coding: utf-8

# Sort records in "gstatsim3d_optimised_[hyper]parameters_*.csv"
# by [inference_period, domain, experiment_serial] in ascending order.
# In the file string, the asterisk denotes either 'gp' or 'kriging'.
#
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Raymond Leung <raymond.leung@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#-------------------------------------------------------------------------------

from run_experiments import reorder_optimised_parameters


if __name__=='__main__':
    # Configure specs['info:data_dir'] if default paths are not used
    # Configure specs['{cat}:hyperparams_csv_file'] for cat in ['gp', 'kriging']
    # if default file names are not used.
    specs = {} 
    reorder_optimised_parameters(0, 1, specs)
