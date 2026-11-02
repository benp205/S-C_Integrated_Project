# 41014 Integrated Project - Track B
## Task Description
UR3 in MuJoCo with a wrist-mounted RGB-D
camera. Detect a target object on a tabletop, plan an approach and execute a pick using IBVS or
PBVS. Required: camera calibration; depth-based object localisation; state-space joint controller;
observer for unmeasured states
## Group information
- Group 5
- - Ben Palmer > 25364542
- - Julian Austin > 25991454
- - Nick Stenger > 24951244
- - Mary Tran > 25366345

## Install info
Python 3.10+ Required

All required dependencies:
```bash
pip install mujoco mujoco-menagerie numpy scipy opencv-python matplotlib spatialmath-python machinevision-toolbox-python
```

## Run commands
Running main.py shows the normal simulation with a small randomisation to joint encoder read positions, alongside small camera errors.
```bash
python main.py
```
Running evaluate_runs.py creates graphs based on 5 separate trials. Used to demonstrate the consistency of the system and save graphs for the github page.
```bash
python evaluate_runs.py
```
