# Bone Eyedropper Add-on for Blender
![Blender](https://img.shields.io/badge/Blender-4.2.0%2B-orange)
![License](https://img.shields.io/badge/License-GPLv3-blue)

## Overview
The Bone Eyedropper add-on for Blender allows users to easily select bones in an armature using a eyedropper tool.  This tool is especially useful for directly selecting bones in the 3D view and setting up subtargets of the constraint.

## Features
- Bones within the constraint target armature can be selected.
- The bone closest to the mouse cursor is selected.
- Visual feedback with a highlighted bone name and location in 3D view.



## Installation
Download the latest release from the [releases page](https://github.com/Puls-r/BoneEyedropper/releases/).
In Blender, go to `Edit > Preferences > Add-ons`.
Click Install and select the downloaded zip file.
Enable the Bone Eyedropper add-on.

## Usage
1. **Set Up Bone Constraints**:
   - Select the bone to which you want to apply the constraint.
   - In the Properties panel, go to the `Bone Constraints` tab.
   - Add a constraint and specify the target.
2. **Use the Bone Eyedropper Tool**:
   - Click the eyedropper icon next to the constraint subtarget field.
   - In the 3D Viewport, click on the bone you want to select.
   - Only bones within the target armature can be selected.
   - The selected bone will be set as the subtarget of the constraint.

>[!TIP]
If the UI breaks, you can try to fix it by running `reload_overwrite` from the Menu Search (`F3`).
## Note
This add-on uses a modified version of Blender's official [`scripts/startup/bl_ui/properties_constraint.py`](https://projects.blender.org/blender/blender/src/branch/main/scripts/startup/bl_ui/properties_constraint.py) script.