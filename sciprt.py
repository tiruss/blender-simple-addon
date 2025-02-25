bl_info = {
    "name": "Addon-test",
    "author": "DK",
    "version": (0, 1),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Demo",
    "description": "A simple demo add-on for Blender",
    "warning": "",
    "doc_url": "",
    "category": "Object",
}

import bpy

# 연산자(Operator) 정의: 버튼 클릭 시 실행할 동작을 구현합니다.
class HelloWorldOperator(bpy.types.Operator):
    bl_idname = "wm.hello_world"   # 고유한 ID, 다른 클래스와 겹치지 않도록 합니다.
    bl_label = "Hello World Operator"  # UI에 표시될 이름

    def execute(self, context):
        # 메시지를 Blender Info 영역과 콘솔에 출력합니다.
        self.report({'INFO'}, "Hello, Blender!")
        print("Hello, Blender!")
        return {'FINISHED'}

# 패널(Panel) 정의: UI에 버튼 등의 요소를 배치합니다.
class HelloWorldPanel(bpy.types.Panel):
    bl_label = "Hello World Panel"
    bl_idname = "OBJECT_PT_hello_world_panel"
    bl_space_type = 'VIEW_3D'   # 3D 뷰포트에 표시
    bl_region_type = 'UI'       # 사이드바 영역에 표시
    bl_category = 'Demo'        # 사이드바 탭 이름

    def draw(self, context):
        layout = self.layout
        # Operator를 버튼 형태로 추가합니다.
        layout.operator(HelloWorldOperator.bl_idname, text="Say Hello")

# 등록 함수: 정의한 클래스들을 Blender에 등록합니다.
def register():
    bpy.utils.register_class(HelloWorldOperator)
    bpy.utils.register_class(HelloWorldPanel)

# 해제 함수: Add-on 비활성화 시 등록된 클래스들을 해제합니다.
def unregister():
    bpy.utils.unregister_class(HelloWorldPanel)
    bpy.utils.unregister_class(HelloWorldOperator)

if __name__ == "__main__":
    register()
