#!/usr/bin/python3
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
# Extracts execution times of GROMACS simulation from GROMACS log file.
# Produces JSON for CI runner.
import sys
import json

if __name__ == "__main__":

    # Path to log file
    gmx_output_path = sys.argv[1]

    # Path for output file
    json_out_path = sys.argv[2]

    # Labels of interest
    #labels = ("Neighbor search", "Launch PP GPU ops.", "Force", "Wait GPU NB local", "Wait GPU state copy", "NB X/F buffer ops.", "Update", "Constraints", "Kinetic energy", "Rest ")
    # Pick only a meaningful ones:
    labels = ("Neighbor search", "Launch PP GPU ops.", "Wait GPU NB local", "Wait GPU state copy", "Constraints")


    data = []

    with open(gmx_output_path, "r") as res:
        for line in res:
            for label in labels:
                if line.lstrip().startswith(label):
                    current_label = label.replace(" ", "_")
                    wall_time = line.strip().split()[-3]
                    data.append({ "name": current_label, "unit": "Seconds", "value": float(wall_time)})

            # Also extract the total core and wall time
            if line.lstrip().startswith("Time:"):
                split_total_time = line.lstrip().split()
                data.append({ "name": "Total_core_time", "unit": "Seconds", "value": float(split_total_time[1])})
                data.append({ "name": "Total_wall_time", "unit": "Seconds", "value": float(split_total_time[2])})


    with open(json_out_path+".json", 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=2)
