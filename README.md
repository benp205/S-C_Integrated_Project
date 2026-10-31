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
- - Nick Stenger >
- - Mary Tran > 25366345

## Install info
Python 3.10+ Required

All required dependencies:
```bash
pip install mujoco mujoco-menagerie numpy scipy opencv-python matplotlib spatialmath-python machinevision-toolbox-python
```

## Run commands
Explanation of how to run your code and what is expected to happen
```bash
python main.py
```
