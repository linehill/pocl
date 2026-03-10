#!/usr/bin/python3
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
