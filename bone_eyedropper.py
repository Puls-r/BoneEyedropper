# ------------------------------------------------------------------------------------------
#  Copyright (c) Nifs. All rights reserved.
#  Licensed under the GPL-3.0 License. See LICENSE in the project root for license information.
# ------------------------------------------------------------------------------------------
import bpy
import blf
import bmesh
import gpu
import math
import os
import numpy as np
from typing import List, Tuple
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
vaild_type = [{
    "type": bpy.types.Constraint,
    "property": ["subtarget", "pole_subtarget"],
    "property_name": ["target", "object"]
}
]


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
        self.__min_bone: bpy.types.PoseBone = None
        self.__mousecoord = None
        self.__bonecoord = None
        self.struct = None
        self.property = None
        self.property_name = None
        self.target = None
        self.hidden = False
        self.bones = []
        self.current_bone_index = 0
        self.copy_name_mode = False
        self.depsgraph = None
        self.evaluated_cache = {}
        self.visible_bones_cache = {}

    def __custom_shape_matrix(self, bone: bpy.types.PoseBone) -> Matrix:
        '''Get the custom shape matrix of the bone'''
        if bone.custom_shape:
            translation_matrix = Matrix.Translation(
                bone.custom_shape_translation)
            scale_matrix = Matrix.Diagonal(
                bone.custom_shape_scale_xyz).to_4x4()
            rotation_matrix = bone.custom_shape_rotation_euler.to_matrix().to_4x4()
            return translation_matrix @ rotation_matrix @ scale_matrix
        return Matrix()

    def __get_evaluated(self, bone: bpy.types.PoseBone) -> tuple[Matrix, bpy.types.Mesh]:
        '''Get the evaluated of the custom shape of the bone'''
        def create_bbone_mesh(bbone: bpy.types.PoseBone):
            bm = bmesh.new()

            for i in range(bbone.bone.bbone_segments):
                cube = bmesh.ops.create_cube(bm, size=0.2)

                id_data_matrix = bbone.id_data.matrix_world
                if bbone.bone.bbone_segments > 1:
                    # Get the current and next segment matrices
                    mat_current = bbone.bbone_segment_matrix(i, rest=False)
                    mat_next = bbone.bbone_segment_matrix(i + 1, rest=False)
                    # Lerp between the two matrices
                    blended_matrix = mat_current.lerp(mat_next, 0.5)
                    matrix_final = bbone.matrix @ blended_matrix
                else:
                    matrix_final = bbone.matrix
                    bmesh.ops.translate(
                        bm, verts=cube["verts"], vec=Vector((0, 0.1, 0)))
                bmesh.ops.scale(bm, verts=cube["verts"], vec=Vector((1, 5, 1)))
                # Display size Scale
                x = bbone.bone.bbone_x * 10
                z = bbone.bone.bbone_z * 10
                l = bbone.length / bbone.bone.bbone_segments
                bmesh.ops.scale(bm, verts=cube["verts"], vec=Vector((x, l, z)))
                bmesh.ops.transform(
                    bm, verts=cube["verts"], matrix=matrix_final)

            mesh_data = bpy.data.meshes.new("BBone_Mesh")
            bm.to_mesh(mesh_data)
            bm.free()
            mesh_obj = bpy.data.objects.new("BBone", mesh_data)
            return mesh_obj

        if bone.custom_shape:
            mesh_obj = bone.custom_shape
            if bone.custom_shape_transform:
                override_mat = bone.custom_shape_transform.matrix
                bone_w_mat = self.target.matrix_world @ override_mat
            else:
                bone_w_mat = self.target.matrix_world @ bone.matrix
            mesh_obj = mesh_obj.evaluated_get(self.depsgraph)
            # Apply custom shape transformation
            # Scale the object to match the bone length
            mat = bone_w_mat @ self.__custom_shape_matrix(
                bone) @ Matrix.Scale(bone.length, 4)
        elif bone.id_data.data.display_type == 'BBONE':
            mesh_obj = create_bbone_mesh(bone)
            bone_w_mat = Matrix()
            mat = bone_w_mat
        else:
            mesh_obj = get_asset()
            bone_w_mat = self.target.matrix_world @ bone.matrix @ Matrix.Scale(
                bone.length, 4)
            mat = bone_w_mat
        return mat, mesh_obj.to_mesh()

    def __get_evaluated_cached(self, bone: bpy.types.PoseBone) -> tuple[Matrix, bpy.types.Mesh]:
        '''Get the evaluated of the custom shape of the bone with caching'''
        if bone in self.evaluated_cache:
            return self.evaluated_cache[bone]
        result = self.__get_evaluated(bone)
        self.evaluated_cache[bone] = result
        return result

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
        context.workspace.status_text_set(None)

    def __draw(self, context):
        if (
            self.__min_bone
            and self.__mousecoord
            and self.__bonecoord
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
            text_width, text_height = blf.dimensions(
                font_id, self.__min_bone.name)

            # Draw dashed line
            self.__draw_dashed_line()

            # Draw bone mesh
            self.__draw_bone_mesh(context)

            # Set background color and draw rounded rectangle
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
            blf.draw(font_id, self.__min_bone.name)

    def __draw_dashed_line(self, dash_length=10):
        start = self.__mousecoord
        end = self.__bonecoord
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

    def __draw_bone_mesh(self, context):
        def generate_shader_batch(matrix, data):
            me = data
            me.calc_loop_triangles()
            # vertices
            vs = np.zeros((len(me.vertices) * 3, ), dtype=np.float32, )
            me.vertices.foreach_get('co', vs)
            vs.shape = (-1, 3, )
            # edges
            es = np.zeros((len(me.edges) * 2, ), dtype=np.int32, )
            me.edges.foreach_get('vertices', es)
            es.shape = (-1, 2, )
            # faces
            fs = np.zeros((len(me.loop_triangles) * 3, ), dtype=np.int32, )
            me.loop_triangles.foreach_get('vertices', fs)
            fs.shape = (-1, 3, )
            # colors
            cs = np.full((len(me.vertices), 4),
                         get_prefereces().bone_suggestions_color, dtype=np.float32, )
            # if object has no faces, draw edges
            shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
            shader.uniform_float("model", matrix)
            shader.uniform_float(
                "view", bpy.context.region_data.view_matrix)
            shader.uniform_float(
                "projection", bpy.context.region_data.window_matrix)
            if len(fs) == 0:
                batch = batch_for_shader(
                    shader, 'LINES', {"position": vs, "color": cs, }, indices=es, )
                # Set line width
                gpu.state.line_width_set(get_prefereces().line_width)
            else:
                batch = batch_for_shader(
                    shader, 'TRIS', {"position": vs, "color": cs, }, indices=fs, )
            return shader, batch

        matrix,  data = self.__get_evaluated_cached(self.__min_bone)
        shader, batch = generate_shader_batch(matrix, data)
        gpu.state.blend_set("ADDITIVE")
        shader.bind()

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

    def __get_closest_bones(self, context, event, region, space):
        """
        Returns the list of bones closes to the cursor position in the specified region.
        """
        def get_closest_vertex_to_cursor(mesh, mat, region, space, coord):
            bcoords = [
                location_3d_to_region_2d(
                    region, space.region_3d, mat @ Vector(v.co))
                for v in mesh.vertices
            ]
            bcoords = [bc for bc in bcoords if bc is not None]
            if bcoords:
                return min(bcoords, key=lambda x: (x - coord).length)
            return None
        coord = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
        bones = self.__get_visible_pose_bones(event.shift)
        bone_distances: List[Tuple[bpy.types.PoseBone, float, Vector]] = []
        evaluated_bones = {b: self.__get_evaluated_cached(b) for b in bones}
        for b, (mat, mesh) in evaluated_bones.items():
            bcoord = None
            if b.custom_shape:
                bcoord = get_closest_vertex_to_cursor(
                    mesh, mat, region, space, coord)
            elif b.id_data.data.display_type == 'BBONE':
                bcoord = get_closest_vertex_to_cursor(
                    mesh, mat, region, space, coord)
            else:
                head = self.target.matrix_world @ b.head
                tail = self.target.matrix_world @ b.tail
                bone_world_center = (head + tail) / 2
                bcoord = location_3d_to_region_2d(
                    region, space.region_3d, bone_world_center)

            if bcoord:
                dist = (bcoord - coord).length
                bone_distances.append((b, dist, bcoord))

        bone_distances.sort(key=lambda x: x[1])
        return bone_distances

    def __get_visible_pose_bones(self, consider_hidden_bones=False):
        '''Get visible pose bones with caching'''
        cache_key = (self.target, consider_hidden_bones)
        if cache_key in self.visible_bones_cache:
            return self.visible_bones_cache[cache_key]

        # Get bones from visible Bone Collections
        visible_bones = {
            b.name: b
            for c in self.target.data.collections_all
            if c.is_visible or consider_hidden_bones
            for b in c.bones
            if not b.hide or consider_hidden_bones
        }

        # Get bones that don't belong to any collection
        all_bones = set(self.target.data.bones)
        try:
            collection_bones = {
                b for c in self.target.data.collections_all for b in c.bones
            }
        except Exception:
            collection_bones = set()

        # Bones that don't belong in either collection
        non_collection_bones = all_bones - collection_bones
        visible_bones.update({
            b.name: b for b in non_collection_bones if not b.hide or consider_hidden_bones
        })

        # Convert to pose bones
        pose_bones = self.target.pose.bones
        result = [
            pose_bones.get(b_name)
            for b_name in visible_bones
            if pose_bones.get(b_name) is not None
        ]

        self.visible_bones_cache[cache_key] = result
        return result

    def modal(self, context, event):

        # Update status text
        context.workspace.status_text_set(
            f"Ctl+Wheel: Change closest bone | Shift: Get hidden bones | LMB: Set bone | RMB/Esc: Cancel")

        context.window.cursor_set("EYEDROPPER")
        region, area, space = get_region_under_cursor(context, event)
        if event.type in {"RIGHTMOUSE", "ESC"}:
            self.__end(context, area)
            return {"CANCELLED"}
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
            if event.type == "MOUSEMOVE":
                self.current_bone_index = 0
            # Get min bone from the list
            self.bones = self.__get_closest_bones(
                context, event, region, space
            )
            if event.type == "WHEELUPMOUSE" and event.ctrl:
                # limit the index to the length of the list
                self.current_bone_index = min(
                    self.current_bone_index + 1, len(self.bones) - 1)
            elif event.type == "WHEELDOWNMOUSE" and event.ctrl:
                # limit the index to 0
                self.current_bone_index = max(
                    self.current_bone_index - 1, 0)
            self.__min_bone = self.bones[self.current_bone_index][0]
            self.__bonecoord = self.bones[self.current_bone_index][2]
            self.__mousecoord = Vector(
                (event.mouse_x - region.x, event.mouse_y - region.y)
            )
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                # Set
                if self.__min_bone:
                    if self.copy_name_mode:
                        # Copy bone name to clipboard
                        bpy.context.window_manager.clipboard = self.__min_bone.name
                        self.report(
                            {"INFO"}, f"Copied {self.__min_bone.name} to clipboard")
                        self.__end(context, area)
                        return {"FINISHED"}
                    try:
                        setattr(self.struct, self.property_name,
                                self.__min_bone.name)
                        self.report(
                            {"INFO"}, f"Set property to {self.__min_bone.name}")
                        self.__end(context, area)
                        return {"FINISHED"}
                    except Exception as e:
                        self.report({"ERROR"}, f"Error setting property: {e}")
                        self.__end(context, area)
                        return {"CANCELLED"}
                self.__end(context, area)
                return {"FINISHED"}
        area.tag_redraw()
        # If viewpoint manipulation is disabled during the modal, the performance is further improved because the cache of vertex positions can be used. 
        # return {"RUNNING_MODAL"} 
        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        try:
            self.struct = context.button_pointer
            self.property = context.button_prop
            self.property_name = context.property[1]

            # Get struct
            dict = next(
                (item for item in vaild_type if item["type"] == type(self.struct) or issubclass(type(self.struct), item["type"])), None)
            if dict:
                # if self.property_name in dict["property"]:
                for prop in dict["property_name"]:
                    if hasattr(self.struct, prop):
                        self.target = getattr(self.struct, prop)
                        if type(self.target) != bpy.types.Object:
                            # TODO: temporary
                            self.target = context.active_object
                        break
                else:
                    self.report(
                        {"ERROR"}, f"None of the properties {dict['property_name']} found in struct")
                    return {"CANCELLED"}

        except Exception as e:
            # When called directly, mode to copy only the name of the bone for the active object
            self.copy_name_mode = True
            self.target = context.active_object
            self.report(
                {"INFO"}, "Copy bone name mode")
        self.depsgraph = context.evaluated_depsgraph_get()
        self.evaluated_cache.clear()
        self.visible_bones_cache.clear()
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


def get_asset(type=None):
    if type == "BBone":
        asset_name = "BoneEyeDropper_BBone_Default"
    else:
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


def draw_menu(self, context: bpy.types.Context):
    # context menu
    if context and not context.property[1] in vaild_type[0]["property"]:
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

    line_width: bpy.props.IntProperty(
        name="Line Width",
        default=5,
        min=1,
        max=100,
        soft_min=1,
        soft_max=100,
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
        row.prop(self, "line_width")
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
