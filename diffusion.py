bl_info = {
    "name": "Diffusion Model Add-on",
    "author": "Your Name",
    "version": (0, 1),
    "blender": (2, 93, 0),
    "location": "View3D > Sidebar > Diffusion",
    "description": "Generate images using a diffusion model from a text prompt",
    "warning": "",
    "doc_url": "",
    "category": "Object",
}

import bpy
import os
import tempfile
import torch
from diffusers import StableDiffusionPipeline

# 텍스트 프롬프트를 입력받아 이미지를 생성하는 Operator
class DiffusionImageOperator(bpy.types.Operator):
    bl_idname = "object.diffusion_image_operator"
    bl_label = "Generate Diffusion Image"

    # 사용자로부터 입력받을 텍스트 프롬프트
    prompt: bpy.props.StringProperty(
        name="Text Prompt",
        description="Enter text prompt for image generation",
        default="A futuristic cityscape",
    )

    def execute(self, context):
        self.report({'INFO'}, "Loading diffusion model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Stable Diffusion 파이프라인 로딩 (초기 실행 시 모델 다운로드가 진행됩니다)
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        except Exception as e:
            self.report({'ERROR'}, f"Error loading model: {e}")
            return {'CANCELLED'}
        
        pipe = pipe.to(device)
        self.report({'INFO'}, "Generating image...")
        try:
            result = pipe(self.prompt)
        except Exception as e:
            self.report({'ERROR'}, f"Error during image generation: {e}")
            return {'CANCELLED'}
        
        # 결과 이미지 가져오기
        image = result.images[0]
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "generated_diffusion_image.png")
        try:
            image.save(image_path)
        except Exception as e:
            self.report({'ERROR'}, f"Error saving image: {e}")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Image saved at: {image_path}")
        
        # Blender 내부로 이미지 불러오기 (이미 존재하면 갱신)
        try:
            if "Generated Diffusion Image" in bpy.data.images:
                img = bpy.data.images["Generated Diffusion Image"]
                img.reload()
            else:
                img = bpy.data.images.load(filepath=image_path)
                img.name = "Generated Diffusion Image"
        except Exception as e:
            self.report({'ERROR'}, f"Error loading image in Blender: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

    # 속성 입력 다이얼로그 호출
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "prompt")

# 3D 뷰포트 사이드바에 버튼을 배치하는 Panel
class DiffusionPanel(bpy.types.Panel):
    bl_label = "Diffusion Model Panel"
    bl_idname = "OBJECT_PT_diffusion_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Diffusion'

    def draw(self, context):
        layout = self.layout
        layout.operator(DiffusionImageOperator.bl_idname, text="Generate Image")

# 등록 및 해제 함수
def register():
    bpy.utils.register_class(DiffusionImageOperator)
    bpy.utils.register_class(DiffusionPanel)

def unregister():
    bpy.utils.unregister_class(DiffusionPanel)
    bpy.utils.unregister_class(DiffusionImageOperator)

if __name__ == "__main__":
    register()
