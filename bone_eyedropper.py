# ------------------------------------------------------------------------------------------
#  Copyright (c) Nifs. All rights reserved.
#  Licensed under the GPL-3.0 License. See LICENSE in the project root for license information.
# ------------------------------------------------------------------------------------------
import bpy
import blf
import gpu
import math
import os
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from bpy_extras.view3d_utils import location_3d_to_region_2d

vertex_shader = '''
in vec3 position;
in vec4 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec4 fcolor;
void main()
{
    vec3 pos = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * vec4(pos, 1.0);
    fcolor = color;
}
'''
fragment_shader = '''
in vec4 fcolor;
out vec4 fragColor;
void main()
{
    fragColor = blender_srgb_to_framebuffer_space(fcolor);
}
'''


class OBJECT_OT_BoneEyedropper(bpy.types.Operator):
    bl_idname = "object.bone_eyedropper"
    bl_label = "Bone Eyedropper"
    bl_description = (
        "Select the bone of the target (Armature) of the constraint with the picker"
    )
    bl_options = {"REGISTER", "UNDO"}

    __handler = None

    @classmethod
    def poll(cls, context):
        return context.active_object

    def __init__(self):
        self.__bonenname = None
        self.__mousecoord = None
        self.__bonecoord_head = None
        self.__bonecoord_tail = None
        self.data_path = None
        self.object = None
        self.constraint = None
        self.target = None
        self.hidden = False
        self.bone_mesh = None

    def __bonecoord(self):
        return Vector((self.__bonecoord_head + self.__bonecoord_tail) / 2)

    def __handle_add(self, context):
        if OBJECT_OT_BoneEyedropper.__handler is None:
            OBJECT_OT_BoneEyedropper.__handler = bpy.types.SpaceView3D.draw_handler_add(
                self.__draw, (context,), "WINDOW", "POST_PIXEL"
            )

    def __handle_remove(self, context):
        if OBJECT_OT_BoneEyedropper.__handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                OBJECT_OT_BoneEyedropper.__handler, "WINDOW"
            )
            OBJECT_OT_BoneEyedropper.__handler = None

    def __end(self, context, area):
        # Cancel operation
        context.window.cursor_set("DEFAULT")
        self.__handle_remove(context)
        area.tag_redraw()
        try:
            # Remove bone mesh
            bpy.data.meshes.remove(self.bone_mesh.data)
            # bpy.data.objects.remove(self.bone_mesh)
        except Exception as e:
            self.report({"ERROR"}, f"Error removing bone mesh: {e}")

    def __draw(self, context):
        if (
            self.__bonenname
            and self.__mousecoord
            and self.__bonecoord_head
            and self.__bonecoord_tail
        ):
            # pref
            pref = get_prefereces()
            gpu.state.blend_set("ALPHA")
            # Calculate position
            x = self.__mousecoord.x + 50
            y = self.__mousecoord.y + 50

            # Set text size and get dimensions
            font_id = 0
            blf.size(font_id, pref.text_size)
            text_width, text_height = blf.dimensions(font_id, self.__bonenname)

            # Draw dashed line (Debug)
            self.__draw_dashed_line(self.__mousecoord, self.__bonecoord())

            # Draw bone mesh
            self.__draw_bone_mesh()

            # Set background color and draw rounded rectangle
            gpu.state.blend_set("ALPHA")
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            radius = 5
            vertices = self.__rounded_rect_vertices(
                x - 5, y - 5, text_width + 15, text_height + 10, radius
            )
            batch = batch_for_shader(shader, "TRI_FAN", {"pos": vertices})
            shader.bind()
            shader.uniform_float("color", pref.back_color)
            batch.draw(shader)
            gpu.state.blend_set("NONE")

            # Draw Bone Name
            blf.position(font_id, x, y, 0)
            blf.color(font_id, pref.text_color[0], pref.text_color[1],
                      pref.text_color[2], pref.text_color[3])
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

    def __draw_bone_mesh(self):
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")

        def generate_shader_batch(object, ):
            me = object.data
            me.calc_loop_triangles()
            vs = np.zeros((len(me.vertices) * 3, ), dtype=np.float32, )
            me.vertices.foreach_get('co', vs)
            vs.shape = (-1, 3, )
            ns = np.zeros((len(me.vertices) * 3, ), dtype=np.float32, )
            me.vertices.foreach_get('normal', ns)
            ns.shape = (-1, 3, )
            fs = np.zeros((len(me.loop_triangles) * 3, ), dtype=np.int32, )
            me.loop_triangles.foreach_get('vertices', fs)
            fs.shape = (-1, 3, )
            cs = np.full((len(me.vertices), 4),
                         get_prefereces().bone_suggestions_color, dtype=np.float32, )
            shader = gpu.types.GPUShader(vertex_shader, fragment_shader, )
            gpu.state.blend_set("ADDITIVE")
            batch = batch_for_shader(
                shader, 'TRIS', {"position": vs, "color": cs, }, indices=fs, )
            return shader, batch
        obj = self.bone_mesh
        pose_bone = self.target.pose.bones[self.__bonenname]
        bone_w_mat = self.target.matrix_world @ pose_bone.matrix
        obj.matrix_world = bone_w_mat
        # Scale the object to match the bone length
        length = pose_bone.length
        obj.matrix_world = obj.matrix_world @ Matrix.Scale(length, 4)
        shader, batch = generate_shader_batch(obj)
        shader.bind()
        shader.uniform_float("model", obj.matrix_world)
        shader.uniform_float("view", bpy.context.region_data.view_matrix)
        shader.uniform_float(
            "projection", bpy.context.region_data.window_matrix)
        batch.draw(shader)

    def __rounded_rect_vertices(self, x, y, width, height, radius):
        vertices = []
        segments = 4
        # Bottom-left corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append(
                (
                    x + radius - radius * math.cos(angle),
                    y + radius - radius * math.sin(angle),
                )
            )

        # Bottom-right corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append(
                (
                    x + width - radius + radius * math.sin(angle),
                    y + radius - radius * math.cos(angle),
                )
            )

        # Top-right corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append(
                (
                    x + width - radius + radius * math.cos(angle),
                    y + height - radius + radius * math.sin(angle),
                )
            )

        # Top-left corner
        for i in range(segments + 1):
            angle = (i / segments) * (math.pi / 2)
            vertices.append(
                (
                    x + radius - radius * math.sin(angle),
                    y + height - radius + radius * math.cos(angle),
                )
            )

        return vertices

    def modal(self, context, event):
        context.window.cursor_set("EYEDROPPER")
        region, area, space = get_region_under_cursor(context, event)
        if event.shift:
            self.hidden = True
        else:
            self.hidden = False
        if region is None:
            # Cursor is not in a 3D view or Outliner
            return {"PASS_THROUGH"}
        if area.type == "OUTLINER":
            # How to get active bone from outliner context?
            return {"PASS_THROUGH"}
        elif area.type == "VIEW_3D":
            self.constraint, pro = get_constraint_from_datapath(self.data_path)
            if pro == "subtarget":
                self.target = self.constraint.target
            elif pro == "pole_subtarget":
                self.target = self.constraint.pole_target
            else:
                self.report({"ERROR"}, "Invalid property")
                self.__end(context, area)
                return {"CANCELLED"}

            min_bone = get_bone_from_cursor(
                context, event, self.target, region, space
            )
            if event.type == "MOUSEMOVE":
                # Update bone and cursor position
                self.__bonenname = min_bone.name if min_bone else None
                self.__mousecoord = Vector(
                    (event.mouse_x - region.x, event.mouse_y - region.y)
                )
                if min_bone:
                    self.__bonecoord_head = location_3d_to_region_2d(
                        region, space.region_3d, self.target.matrix_world @ min_bone.head
                    )
                    self.__bonecoord_tail = location_3d_to_region_2d(
                        region, space.region_3d, self.target.matrix_world @ min_bone.tail
                    )
                else:
                    self.__bonecoord_head = None
                    self.__bonecoord_tail = None
                area.tag_redraw()
            elif event.type == "LEFTMOUSE" and event.value == "PRESS":
                # Set
                if min_bone:
                    try:
                        setattr(self.constraint, pro, min_bone.name)
                        self.report(
                            {"INFO"}, f"{self.data_path} set to {min_bone.name}")
                    except Exception as e:
                        self.report({"ERROR"}, f"Error setting property: {e}")
                        self.__end(context, area)
                        return {"CANCELLED"}
                self.__end(context, area)
                return {"FINISHED"}
            elif event.type in {"RIGHTMOUSE", "ESC"}:
                self.__end(context, area)
                return {"CANCELLED"}
        area.tag_redraw()
        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        tmp = context.window_manager.clipboard
        # I honestly can't believe this is the solution.
        # Get the data path of the property
        bpy.ops.ui.copy_data_path_button(full_path=True)
        self.data_path = context.window_manager.clipboard
        context.window_manager.clipboard = tmp
        self.bone_mesh = get_asset()
        self.__handle_remove(context)
        self.__handle_add(context)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


def load_object(blend_file_path, object_name):
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        if object_name in data_from.objects:
            data_to.objects.append(object_name)
            return data_to.objects[0]
        else:
            print(f"Object '{object_name}' not found in '{blend_file_path}'")


def get_asset():
    asset_name = "BoneEyeDropper_Bone_Default"
    try:
        # Fix: use a unique name or another way to get the object
        return bpy.data.objects[asset_name]
    except KeyError:
        pass
    addon_directory = os.path.dirname(__file__)
    blend_file_path = os.path.join(
        addon_directory, "assets", "Bone_Asset.blend")
    load_object(blend_file_path, asset_name)
    return bpy.data.objects[asset_name]


def get_constraint_from_datapath(data_path) -> tuple[bpy.types.Constraint, str | None]:
    def remove_last_property(data_path):
        import re
        # Example:
        # bpy.data.objects["Armature"].pose.bones["Bone"].constraints["Constraint"].subtarget
        # -> bpy.data.objects["Armature"].pose.bones["Bone"].constraints["Constraint"]  subtarget
        pattern = r'(.*)\.[^.]+$'
        match = re.match(pattern, data_path)
        if match:
            return match.group(1), data_path.replace(match.group(1) + ".", "")
        return None, None
    tuple_data_path = remove_last_property(data_path)
    return eval(tuple_data_path[0]), tuple_data_path[1]


def get_prefereces():
    return bpy.context.preferences.addons[__package__].preferences


def get_region_under_cursor(
    context, event
) -> tuple[bpy.types.Region, bpy.types.Area, bpy.types.SpaceView3D]:
    for area in context.screen.areas:
        if area.type == "VIEW_3D" or "OUTLINER":
            for region in area.regions:
                if region.type == "WINDOW":
                    # Check if cursor is in the region
                    if (
                        region.x <= event.mouse_x <= region.x + region.width
                        and region.y <= event.mouse_y <= region.y + region.height
                    ):
                        space = area.spaces.active
                        return region, area, space
    return None, None, None


def get_bone_from_cursor(
        context, event, obj, region, space):
    """
    Returns the bone closest to the cursor position in the specified region.
    """
    coord = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
    min_dist = float("inf")
    min_bone = None
    bones = get_visible_pose_bones(obj, event.shift)
    for b in bones:
        # world pos
        head = obj.matrix_world @ b.head
        tail = obj.matrix_world @ b.tail
        bone_world_center = (head + tail) / 2
        # convert to region 2d
        bcoord = location_3d_to_region_2d(
            region, space.region_3d, bone_world_center)
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
        b
        for c in obj.data.collections_all
        if c.is_visible or consider_hidden_bones
        for b in c.bones
        if not b.hide or consider_hidden_bones
    ]

    # Get bones that don't belong to any collection
    all_bones = set(obj.data.bones)
    collection_bones = {b for c in obj.data.collections_all for b in c.bones}
    # Bones that don't belong in either collection
    non_collection_bones = all_bones - collection_bones
    visible_bones.extend(
        b for b in non_collection_bones if not b.hide or consider_hidden_bones
    )

    # Convert to pose bones
    pose_bones = obj.pose.bones
    return [
        pose_bones.get(b.name)
        for b in visible_bones
        if pose_bones.get(b.name) is not None
    ]


def draw_menu(self, context: bpy.types.Context):
    # context menu
    if not context.property[1] in {"subtarget", "pole_subtarget"}:
        return
    layout = self.layout
    layout.separator()
    layout.operator(OBJECT_OT_BoneEyedropper.bl_idname,
                    text="Bone Eyedropper", icon="EYEDROPPER")


class BoneEyedropperPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    bone_suggestions_color: bpy.props.FloatVectorProperty(
        name="Bone Suggestions Color",
        subtype="COLOR",
        size=4,
        default=(0.0, 1.0, 0.0, 0.7),
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
    )

    text_size: bpy.props.IntProperty(
        name="Text Size",
        default=25,
        min=1,
        max=100,
        soft_min=1,
        soft_max=100,
        subtype="FACTOR",
    )

    text_color: bpy.props.FloatVectorProperty(
        name="Text Color",
        subtype="COLOR",
        size=4,
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
    )

    back_color: bpy.props.FloatVectorProperty(
        name="Background Color",
        subtype="COLOR",
        size=4,
        default=(0.1, 0.1, 0.1, 0.8),
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
    )

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "bone_suggestions_color")
        row = layout.row()
        row.prop(self, "text_size")
        row = layout.row()
        row.prop(self, "text_color")
        row = layout.row()
        row.prop(self, "back_color")


classes = [
    OBJECT_OT_BoneEyedropper,
    BoneEyedropperPreferences,
]


def register_component():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.UI_MT_button_context_menu.append(draw_menu)


def unregister_component():
    bpy.types.UI_MT_button_context_menu.remove(draw_menu)
    for cls in classes:
        bpy.utils.unregister_class(cls)
