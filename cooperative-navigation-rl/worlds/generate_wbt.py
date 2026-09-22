# Python 脚本：generate_wbt.py

url_1 =  "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
url_2 = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
url_3 = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"
url_4 = "E-puck.proto"

wbt_content = f'#VRML_SIM R2023b utf8\nEXTERNPROTO "{url_1}"\nEXTERNPROTO "{url_2}"\nIMPORTABLE EXTERNPROTO "{url_3}"\nIMPORTABLE EXTERNPROTO "{url_4}"\n'
wbt_content += "WorldInfo {\n  basicTimeStep 24\n}\n"
Viewpoint = """
Viewpoint {
  orientation -0.5773502691896258 0.5773502691896258 0.5773502691896258 2.0944
  position 0.00011706497804167808 0.1699505320272429 4.101142536591572
}
TexturedBackground {
}
TexturedBackgroundLight {
}
"""
wbt_content += Viewpoint
num_envs = 1
num_agents = 6
receiver_template = """
    Receiver {{
      name "receiver"
      channel {channel}
    }}
"""
emitter_template = """
    Emitter {{
      name "emitter"
      channel {channel}
    }}
"""
emitter_target_template = """
    Emitter {{
      name "emitter_target"
      channel {channel}
    }}
"""
# emitter_arb_template = """
#     Emitter {{
#       name "emitter_arb{name}-{agent}"
#       channel {channel}
#     }}
# """
# receivers_emitters = ""
# for i in range(num_envs):
#     receivers_emitters += receiver_template.format(name=i+1,channel=1+i*(num_agents+3))
#     receivers_emitters += emitter_template.format(name=i+1, channel=2+i*(num_agents+3))
#     receivers_emitters += emitter_target_template.format(name=i + 1, channel=3 + i * (num_agents + 3))
#     # for j in range(num_agents):
#     #     receivers_emitters += emitter_arb_template.format(name=i+1, agent=j+1,channel=j + i * (num_agents + 3) + 4)
    
receivers_emitters = ""
for i in range(num_envs):
    receivers_emitters += receiver_template.format(channel=1+i*(num_agents+3))
    receivers_emitters += emitter_template.format(channel=2+i*(num_agents+3))
    receivers_emitters += emitter_target_template.format(channel=3 + i * (num_agents + 3))

SUPERVISOR = """
DEF SUPERVISOR Robot {{
  children [
{receiver_and_emitter}
  ]
  name "supervisor"
  controller "<extern>"
  supervisor TRUE
}}
"""

SUPERVISOR_content = SUPERVISOR.format(receiver_and_emitter=receivers_emitters)
wbt_content += SUPERVISOR_content


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(current_dir + "/generated_world.wbt", "w") as f:
    f.write(wbt_content)

print("generated_world.wbt 文件已生成")