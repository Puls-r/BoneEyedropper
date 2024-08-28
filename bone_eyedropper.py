# ------------------------------------------------------------------------------------------
#  Copyright (c) Nifs. All rights reserved.
#  Licensed under the GPL-3.0 License. See LICENSE in the project root for license information.
# ------------------------------------------------------------------------------------------
import bpy
import blf
import gpu
import math
from mathutils import Vector
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy.app.handlers import persistent


class OBJECT_OT_BoneEyedropper(bpy.types.Operator):
    bl_idname = "object.bone_eyedropper"
    bl_label = "Bone Eyedropper"
    bl_description = "Select the bone of the target (Armature) of the constraint with the picker"
    bl_options = {"REGISTER", "UNDO"}

    obj: bpy.props.StringProperty(name="Object", default="")
    path: bpy.props.StringProperty(name="Path", default="")
    __handler = None

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == "ARMATURE"

    def __init__(self):
        self.__bonenname = None
        self.__mousecoord = None
        self.__bonecoord_head = None
        self.__bonecoord_tail = None
        self.object = bpy.data.objects.get(self.obj)
        self.constraint: bpy.types.Constraint = self.object.path_resolve(self.path)

    def __bonecoord(self):
        return Vector((self.__bonecoord_head + self.__bonecoord_tail) / 2)

    def __handle_add(self, context):
        if OBJECT_OT_BoneEyedropper.__handler is None:
            OBJECT_OT_BoneEyedropper.__handler = bpy.types.SpaceView3D.draw_handler_add(self.__draw, (context,), "WINDOW", "POST_PIXEL")

    def __handle_remove(self, context):
        if OBJECT_OT_BoneEyedropper.__handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(OBJECT_OT_BoneEyedropper.__handler, "WINDOW")
            OBJECT_OT_BoneEyedropper.__handler = None

    def __draw(self, context):
        if self.__bonenname and self.__mousecoord and self.__bonecoord_head and self.__bonecoord_tail:
            # Calculate position
            x = self.__mousecoord.x + 50
            y = self.__mousecoord.y + 50

            # Set text size and get dimensions
            font_id = 0
            blf.size(font_id, 17)
            text_width, text_height = blf.dimensions(font_id, self.__bonenname)

            # Draw dashed line (Debug)
            self.__draw_dashed_line(self.__mousecoord, self.__bonecoord())

            # Draw triangle and dot
            self.__draw_bone_line()

            # Set background color and draw rounded rectangle
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            radius = 5
            vertices = self.__rounded_rect_vertices(x - 5, y - 5, text_width + 15, text_height + 10, radius)
            batch = batch_for_shader(shader, "TRI_FAN", {"pos": vertices})
            shader.bind()
            shader.uniform_float("color", (0.1, 0.1, 0.1, 0.8))
            batch.draw(shader)

            # Draw text
            blf.position(font_id, x, y, 0)
            blf.color(font_id, 1, 1, 1, 1)
            blf.draw(font_id, self.__bonenname)

    def __draw_dashed_line(self, start, end, dash_length=10):
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))

        total_length = (end - start).length
        num_dashes = int(total_length / dash_length)
        direction = (end - start).normalized()

        vertices = []
        for i in range(num_dashes):
            if i % 2 == 0:
                segment_start = start + direction * (i * dash_length)
                segment_end = start + direction * ((i + 1) * dash_length)
                vertices.extend([segment_start, segment_end])

        batch = batch_for_shader(shader, "LINES", {"pos": vertices})
        batch.draw(shader)

    def __draw_bone_line(self):
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        length = (self.__bonecoord_head - self.__bonecoord_tail).length
        gpu.state.line_width_set(1.0 * length / 10)

        # Draw line
        vertices = [self.__bonecoord_head, self.__bonecoord_tail]
        batch = batch_for_shader(shader, "LINE_STRIP", {"pos": vertices})
        shader.uniform_float("color", (0.0, 1.0, 0.0, 0.5))
        batch.draw(shader)

    def __rounded_rect_vertices(self, x, y, width, height, radius):
        vertices = []
        segments = 4
        # Bottom-left corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append((x + radius - radius * math.cos(angle), y + radius - radius * math.sin(angle)))

        # Bottom-right corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append((x + width - radius + radius * math.sin(angle), y + radius - radius * math.cos(angle)))

        # Top-right corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append((x + width - radius + radius * math.cos(angle), y + height - radius + radius * math.sin(angle)))

        # Top-left corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append((x + radius - radius * math.sin(angle), y + height - radius + radius * math.cos(angle)))

        return vertices

    def modal(self, context, event):
        context.window.cursor_set("EYEDROPPER")
        region, area, space = get_region_under_cursor(context, event)
        if region is None:
            # Cursor is not in a 3D view
            return {"PASS_THROUGH"}

        obj = self.constraint.target
        consider_hidden_bones = event.shift
        min_bone = get_bone_from_cursor(context, event, obj, region, space, consider_hidden_bones)
        if event.type == "MOUSEMOVE":
            # Update bone and cursor position
            self.__bonenname = min_bone.name if min_bone else None
            self.__mousecoord = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
            if min_bone:
                self.__bonecoord_head = location_3d_to_region_2d(region, space.region_3d, obj.matrix_world @ min_bone.head)
                self.__bonecoord_tail = location_3d_to_region_2d(region, space.region_3d, obj.matrix_world @ min_bone.tail)
            else:
                self.__bonecoord_head = None
                self.__bonecoord_tail = None
            area.tag_redraw()
        elif event.type == "LEFTMOUSE" and event.value == "PRESS":
            # Set subtarget
            if min_bone:
                cot = self.constraint
                if cot:
                    cot.subtarget = min_bone.name
                    self.report({"INFO"}, f"Subtarget set to {min_bone.name}")
            context.window.cursor_set("DEFAULT")
            self.__handle_remove(context)
            area.tag_redraw()
            return {"FINISHED"}
        elif event.type in {"RIGHTMOUSE", "ESC"}:
            # Cancel operation
            context.window.cursor_set("DEFAULT")
            self.__handle_remove(context)
            area.tag_redraw()
            return {"CANCELLED"}
        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        if context.active_object and context.active_object.type == "ARMATURE":
            self.__handle_remove(context)
            self.__handle_add(context)
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        else:
            self.report({"ERROR"}, "Active object is not an armature")
            return {"CANCELLED"}


def get_region_under_cursor(context, event) -> tuple[bpy.types.Region, bpy.types.Area, bpy.types.SpaceView3D]:
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    # Check if cursor is in the region
                    if region.x <= event.mouse_x <= region.x + region.width and region.y <= event.mouse_y <= region.y + region.height:
                        space = area.spaces.active
                        return region, area, space
    return None, None, None


def get_bone_from_cursor(context, event, obj, region, space, consider_hidden_bones=False):
    """
    Returns the bone closest to the cursor position in the specified region.
    """
    coord = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
    min_dist = float("inf")
    min_bone = None
    bones = get_visible_pose_bones(obj, consider_hidden_bones)
    for b in bones:
        # world pos
        head = obj.matrix_world @ b.head
        tail = obj.matrix_world @ b.tail
        bone_world_center = (head + tail) / 2
        # convert to region 2d
        bcoord = location_3d_to_region_2d(region, space.region_3d, bone_world_center)
        if bcoord is None:
            continue
        dist = (bcoord - coord).length
        # find the closest bone
        if dist < min_dist:
            min_dist = dist
            min_bone = b
    return min_bone


def get_visible_pose_bones(obj: bpy.types.Object, consider_hidden_bones=False):
    # Get bones from visible Bone Collections
    visible_bones = [
        b for c in obj.data.collections_all if c.is_visible or consider_hidden_bones for b in c.bones if not b.hide or consider_hidden_bones
    ]

    # Get bones that don't belong to any collection
    all_bones = set(obj.data.bones)
    collection_bones = {b for c in obj.data.collections_all for b in c.bones}
    # Bones that don't belong in either collection
    non_collection_bones = all_bones - collection_bones
    visible_bones.extend(b for b in non_collection_bones if not b.hide or consider_hidden_bones)

    # Convert to pose bones
    pose_bones = obj.pose.bones
    return [pose_bones.get(b.name) for b in visible_bones if pose_bones.get(b.name) is not None]


def get_active_constraint(context: bpy.types.Context):
    constraints = context.pose_bone.constraints
    return (
        constraints.active
        if constraints.active and constraints.active.target and constraints.active.target.type == "ARMATURE"
        else next((con for con in constraints if con.target and con.target.type == "ARMATURE"), None)
    )


def draw_bone(self: bpy.types.Panel, context: bpy.types.Context):
    layout = self.layout
    sp = layout.row().split(factor=0.4)
    active_constraint = get_active_constraint(context)
    if active_constraint is None:
        sp.label(text="Bone Eyedropper : No valid constraint", icon="ERROR")
        return
    sp.label(text=active_constraint.name)
    op = sp.operator("object.bone_eyedropper", text="Bone Eyedropper", icon="EYEDROPPER")
    op.obj = active_constraint.id_data.name
    op.path = active_constraint.path_from_id()


classes = [
    OBJECT_OT_BoneEyedropper,
]


def register_component():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.BONE_PT_constraints.append(draw_bone)


def unregister_component():
    bpy.types.BONE_PT_constraints.remove(draw_bone)
    for cls in classes:
        bpy.utils.unregister_class(cls)
