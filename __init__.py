# ------------------------------------------------------------------------------------------
#  Copyright (c) Nifs. All rights reserved.
#  Licensed under the GPL-3.0 License. See LICENSE in the project root for license information.
# ------------------------------------------------------------------------------------------

import importlib
from . import bone_eyedropper

if "bpy" in locals():
    import sys

    for k, v in list(sys.modules.items()):
        if k.startswith("BoneEyedropper."):
            importlib.reload(v)


def register():
    bone_eyedropper.register_component()


def unregister():
    bone_eyedropper.unregister_component()


if __name__ == "__main__":
    register()
