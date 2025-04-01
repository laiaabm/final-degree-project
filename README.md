# CHIP Detection in Histology

## Project Description
This repository contains the code for the detection of CHIP in peripheral blood (PB) smears. Two alternative approaches can be found: a patch-based approach and a single-cell-based approach.

The input for our analysis consists of blood patches, and the output is the CHIP diagnosis (either CHIP-positive or CHIP-negative).

The pipeline for the analysis consists of ___ steps:
1. Quality control of the PB slides -- Each 224×224 tile goes through a preprocessing step where its quality is assessed. Tiles with excessive or deficient cellular density and/or poor resolution are removed. The corresponding code can be found in `quality_control/quality_control.py`
2. Segmentation of WBC -- Each individual WBC is segmented in 40x40 single-cell patches using a neural cellular automata approach. This step is only for the *single-cell-based approach*. This code is stored in `segmentation/nca_segment.py`
3. Feature extraction -- 
4. Aggregation --



## Contact Information
Laia Barcenilla: laia.barcenilla@alum.esci.upf.edu  

