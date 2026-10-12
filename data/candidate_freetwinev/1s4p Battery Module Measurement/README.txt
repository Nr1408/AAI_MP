Electro-Thermal Measurements of a 1s4p Molicel INR21700-P45B Battery Module

This file – provides guidance on the repository contents.

This repository contains experimental data and documentation from the FreeTwinEV project related to electro-thermal measurements of a battery module composed of four Molicel INR21700-P45B lithium-ion cells connected in parallel (1s4p configuration). The dataset is intended for calibration and validation of electro-thermal models and digital twins of battery modules.

Repository structure

"Measurement_protocol.xlsx"
Describes the laboratory measurement protocol, experimental setup, test conditions, sensor layout, and supporting documentation.
The file contains the following worksheets:
• Protocol – description of the document and process of the data acquisition.
• Test_Matrix – overview of all performed measurements and corresponding operating conditions.
• Photographs – photographs of the experimental setup captured during laboratory measurements.
• 3D_Model – annotated three-dimensional model of the battery module geometry and component layout.
• Sensor_Layout – description of all sensors used during the experiments, including sensor names and corresponding DAQ measurement channels. Sensors include temperature sensors on battery cells, temperature sensors on the cooling plate, coolant temperature sensors, and an ambient temperature sensor.
• Materials – list of materials used in the battery module assembly together with their relevant material properties.

"AllData"
Directory containing experimental measurement data.
The folder includes 14 subdirectories, each corresponding to a single measurement.
Each subdirectory is named using a unique measurement ID and DAQ timestamp, for example:
ID01_20251126_150908248

Each measurement subdirectory contains two files with identical content stored in different formats:
• FreeTwinEV_1s4p_ID_XX.csv
• FreeTwinEV_1s4p_ID_XX.xlsx

The data files contain time-series measurements of the following quantities:
• timestamp [s] - Timestamp
• current [A] - Battery module current
• voltage [V] - Battery module voltage
• SOC [%] - State of charge
• liquid_flow [L/min] - Coolant flow rate
• temp_ambient [K] - Ambient temperature
• temp_inlet [K] - Coolant inlet temperature
• temp_outlet [K] - Coolant outlet temperature
• temp_TH1-TH10 - Temperature measurements from 10 cooling plate temperature sensors
• temp_cell-1_TH20-TH22 - Temperature measurements from 3 temperature sensors placed on battery cell number 1
• temp_cell-2_TH17-TH19 - Temperature measurements from 3 temperature sensors placed on battery cell number 2
• temp_cell-3_TH14-TH16 - Temperature measurements from 3 temperature sensors placed on battery cell number 3
• temp_cell-4_TH11-TH13 - Temperature measurements from 3 temperature sensors placed on battery cell number 4

Exact sensor names, locations, and measurement channels are defined in the Sensor_Layout worksheet of the measurement protocol.

"LICENSE.txt"
License file specifying the terms under which the dataset is distributed.

Notes
• The experimental procedure and sensor placement are fully documented in Measurement_protocol.xlsx.
• CSV and XLSX files contain identical data and are provided for user convenience.
• The dataset is provided as a technical reference for electro-thermal modeling and validation purposes.

Citation
If you use this dataset, please cite it as:

Author(s), Year. Electro-Thermal Measurements of a 1s4p Molicel INR21700-P45B Battery Module. Zenodo. DOI: [to be assigned by Zenodo]