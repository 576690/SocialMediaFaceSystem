from pathlib import Path
import subprocess


ROOT = Path(r"D:\Personal\Documents\code\Python\SocialMediaFaceSystem\output\thesis_figures")
ROOT.mkdir(parents=True, exist_ok=True)
DRAWIO = Path(r"C:\Program Files\draw.io\draw.io.exe")

BASE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=14;"
RECT = "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=14;"
ELLIPSE = "ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=14;"
ACTOR = "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;fillColor=#ffffff;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=14;"
DECISION = "rhombus;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=14;"
DB = "shape=cylinder3;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=14;"
LANE = "swimlane;startSize=28;whiteSpace=wrap;html=1;fillColor=#f7f7f7;strokeColor=#666666;fontColor=#000000;fontFamily=SimSun;fontSize=14;fontStyle=1;"
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=12;"
LINE = "endArrow=none;html=1;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=12;"


def wrap(name, cells):
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="drawio" version="26.0.0"><diagram name="{name}"><mxGraphModel dx="1200" dy="800" grid="1" gridSize="10"><root>
<mxCell id="0"/><mxCell id="1" parent="0"/>
{cells}
</root></mxGraphModel></diagram></mxfile>'''


def save(name, cells):
    path = ROOT / f"{name}.drawio"
    svg = ROOT / f"{name}.svg"
    png = ROOT / f"{name}.png"
    path.write_text(wrap(name, cells), encoding="utf-8")
    subprocess.run([str(DRAWIO), "-x", "-f", "svg", "-e", "-o", str(svg), str(path)], check=True)
    subprocess.run([str(DRAWIO), "-x", "-f", "png", "-s", "2", "-o", str(png), str(path)], check=True)


save("figure-2-1-use-case", f'''
<mxCell id="2" value="" style="rounded=0;dashed=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;fontColor=#000000;fontFamily=SimSun;fontSize=14;" vertex="1" parent="1"><mxGeometry x="230" y="40" width="610" height="420" as="geometry"/></mxCell>
<mxCell id="3" value="管理员" style="{ACTOR}" vertex="1" parent="1"><mxGeometry x="50" y="80" width="70" height="120" as="geometry"/></mxCell>
<mxCell id="4" value="数据处理人员" style="{ACTOR}" vertex="1" parent="1"><mxGeometry x="50" y="260" width="90" height="120" as="geometry"/></mxCell>
<mxCell id="5" value="检索用户" style="{ACTOR}" vertex="1" parent="1"><mxGeometry x="920" y="180" width="70" height="120" as="geometry"/></mxCell>
<mxCell id="6" value="提交采集任务" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="290" y="90" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="7" value="采集源管理" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="290" y="170" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="8" value="人脸检测与质量过滤" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="290" y="250" width="180" height="55" as="geometry"/></mxCell>
<mxCell id="9" value="系统配置与维护" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="290" y="330" width="160" height="55" as="geometry"/></mxCell>
<mxCell id="10" value="文本语义检索" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="590" y="90" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="11" value="图片人脸检索" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="590" y="170" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="12" value="人物聚类" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="590" y="250" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="13" value="人物档案维护" style="{ELLIPSE}" vertex="1" parent="1"><mxGeometry x="590" y="330" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="14" value="" style="{LINE}" edge="1" parent="1" source="4" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="15" value="" style="{LINE}" edge="1" parent="1" source="4" target="8"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="17" value="" style="{LINE}" edge="1" parent="1" source="3" target="9"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="19" value="" style="{LINE}" edge="1" parent="1" source="5" target="10"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="20" value="" style="{LINE}" edge="1" parent="1" source="5" target="11"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="21" value="" style="{LINE}" edge="1" parent="1" source="5" target="13"><mxGeometry relative="1" as="geometry"/></mxCell>
''')

save("figure-3-2-data-flow", f'''
<mxCell id="2" value="外部输入&#xa;视频链接 / 图文 / 账号来源" style="{BASE}" vertex="1" parent="1"><mxGeometry x="60" y="80" width="180" height="70" as="geometry"/></mxCell>
<mxCell id="3" value="采集器与来源适配器" style="{BASE}" vertex="1" parent="1"><mxGeometry x="300" y="80" width="190" height="70" as="geometry"/></mxCell>
<mxCell id="4" value="contents&#xa;内容元数据" style="{DB}" vertex="1" parent="1"><mxGeometry x="560" y="70" width="170" height="85" as="geometry"/></mxCell>
<mxCell id="5" value="媒体文件&#xa;videos / content" style="{DB}" vertex="1" parent="1"><mxGeometry x="810" y="70" width="170" height="85" as="geometry"/></mxCell>
<mxCell id="6" value="人脸检测与质量过滤" style="{BASE}" vertex="1" parent="1"><mxGeometry x="300" y="250" width="190" height="70" as="geometry"/></mxCell>
<mxCell id="7" value="语义融合&#xa;视觉 / 字幕 / ASR / 正文" style="{BASE}" vertex="1" parent="1"><mxGeometry x="560" y="250" width="190" height="70" as="geometry"/></mxCell>
<mxCell id="8" value="faces&#xa;人脸记录与语义文本" style="{DB}" vertex="1" parent="1"><mxGeometry x="810" y="240" width="180" height="90" as="geometry"/></mxCell>
<mxCell id="9" value="FAISS 向量索引" style="{DB}" vertex="1" parent="1"><mxGeometry x="560" y="430" width="170" height="85" as="geometry"/></mxCell>
<mxCell id="10" value="文本检索 / 图片检索 / 人物聚类" style="{BASE}" vertex="1" parent="1"><mxGeometry x="810" y="430" width="210" height="70" as="geometry"/></mxCell>
<mxCell id="11" value="" style="{EDGE}" edge="1" parent="1" source="2" target="3"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="12" value="元数据" style="{EDGE}" edge="1" parent="1" source="3" target="4"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="13" value="文件" style="{EDGE}" edge="1" parent="1" source="3" target="5"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="14" value="帧/图片" style="{EDGE}" edge="1" parent="1" source="5" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="15" value="文本上下文" style="{EDGE}" edge="1" parent="1" source="4" target="7"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="16" value="有效人脸" style="{EDGE}" edge="1" parent="1" source="6" target="8"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="17" value="semantic_text" style="{EDGE}" edge="1" parent="1" source="7" target="8"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="18" value="embedding" style="{EDGE}" edge="1" parent="1" source="8" target="9"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="19" value="查询与聚类" style="{EDGE}" edge="1" parent="1" source="9" target="10"><mxGeometry relative="1" as="geometry"/></mxCell>
''')

save("figure-4-2-collection-flow", f'''
<mxCell id="2" value="用户提交来源" style="{RECT}" vertex="1" parent="1"><mxGeometry x="80" y="80" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="3" value="识别平台类型" style="{DECISION}" vertex="1" parent="1"><mxGeometry x="300" y="60" width="150" height="95" as="geometry"/></mxCell>
<mxCell id="4" value="yt-dlp 视频/频道" style="{BASE}" vertex="1" parent="1"><mxGeometry x="540" y="40" width="170" height="55" as="geometry"/></mxCell>
<mxCell id="5" value="微博用户图文" style="{BASE}" vertex="1" parent="1"><mxGeometry x="540" y="135" width="170" height="55" as="geometry"/></mxCell>
<mxCell id="6" value="X/Twitter 图片推文" style="{BASE}" vertex="1" parent="1"><mxGeometry x="540" y="230" width="180" height="55" as="geometry"/></mxCell>
<mxCell id="7" value="统一条目格式&#xa;platform / external_id / url / text / images" style="{BASE}" vertex="1" parent="1"><mxGeometry x="800" y="135" width="250" height="80" as="geometry"/></mxCell>
<mxCell id="8" value="写入 contents 并创建后台处理任务" style="{BASE}" vertex="1" parent="1"><mxGeometry x="800" y="330" width="250" height="70" as="geometry"/></mxCell>
<mxCell id="9" value="" style="{EDGE}" edge="1" parent="1" source="2" target="3"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="10" value="视频/频道" style="{EDGE}" edge="1" parent="1" source="3" target="4"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="11" value="微博" style="{EDGE}" edge="1" parent="1" source="3" target="5"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="12" value="X" style="{EDGE}" edge="1" parent="1" source="3" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="13" value="" style="{EDGE}" edge="1" parent="1" source="4" target="7"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="14" value="" style="{EDGE}" edge="1" parent="1" source="5" target="7"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="15" value="" style="{EDGE}" edge="1" parent="1" source="6" target="7"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="16" value="" style="{EDGE}" edge="1" parent="1" source="7" target="8"><mxGeometry relative="1" as="geometry"/></mxCell>
''')

save("figure-4-3-face-quality-flow", f'''
<mxCell id="2" value="读取视频帧或图文图片" style="{RECT}" vertex="1" parent="1"><mxGeometry x="80" y="80" width="190" height="60" as="geometry"/></mxCell>
<mxCell id="3" value="InsightFace&#xa;检测候选人脸" style="{RECT}" vertex="1" parent="1"><mxGeometry x="340" y="80" width="190" height="60" as="geometry"/></mxCell>
<mxCell id="4" value="质量检查&#xa;是否合格" style="{DECISION}" vertex="1" parent="1"><mxGeometry x="640" y="50" width="150" height="120" as="geometry"/></mxCell>
<mxCell id="5" value="提取并归一化&#xa;embedding" style="{RECT}" vertex="1" parent="1"><mxGeometry x="900" y="80" width="190" height="60" as="geometry"/></mxCell>
<mxCell id="6" value="保存人脸图、完整帧和数据库记录" style="{RECT}" vertex="1" parent="1"><mxGeometry x="900" y="245" width="230" height="65" as="geometry"/></mxCell>
<mxCell id="7" value="记录过滤原因&#xa;不进入索引" style="{RECT}" vertex="1" parent="1"><mxGeometry x="620" y="245" width="190" height="65" as="geometry"/></mxCell>
<mxCell id="8" value="质量指标&#xa;min_face_size&#xa;min_face_ratio&#xa;min_laplacian_var&#xa;max_pose_deviation" style="{BASE}" vertex="1" parent="1"><mxGeometry x="340" y="245" width="190" height="120" as="geometry"/></mxCell>
<mxCell id="10" value="" style="{EDGE}" edge="1" parent="1" source="2" target="3"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="11" value="" style="{EDGE}" edge="1" parent="1" source="3" target="4"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="12" value="通过" style="{EDGE}" edge="1" parent="1" source="4" target="5"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="13" value="不通过" style="{EDGE}" edge="1" parent="1" source="4" target="7"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="14" value="" style="{EDGE}" edge="1" parent="1" source="5" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
''')

save("figure-4-4-semantic-fusion-flow", f'''
<mxCell id="2" value="内容标题与来源元数据" style="{BASE}" vertex="1" parent="1"><mxGeometry x="80" y="70" width="190" height="55" as="geometry"/></mxCell>
<mxCell id="3" value="图文正文" style="{BASE}" vertex="1" parent="1"><mxGeometry x="80" y="160" width="190" height="55" as="geometry"/></mxCell>
<mxCell id="4" value="字幕 / ASR 片段" style="{BASE}" vertex="1" parent="1"><mxGeometry x="80" y="250" width="190" height="55" as="geometry"/></mxCell>
<mxCell id="5" value="Florence-2 视觉描述" style="{BASE}" vertex="1" parent="1"><mxGeometry x="80" y="340" width="190" height="55" as="geometry"/></mxCell>
<mxCell id="6" value="按时间戳与人脸记录对齐" style="{BASE}" vertex="1" parent="1"><mxGeometry x="360" y="210" width="220" height="70" as="geometry"/></mxCell>
<mxCell id="7" value="去重、截断与结构化组织" style="{BASE}" vertex="1" parent="1"><mxGeometry x="660" y="210" width="220" height="70" as="geometry"/></mxCell>
<mxCell id="8" value="semantic_text&#xa;Visual / Speech / Post" style="{DB}" vertex="1" parent="1"><mxGeometry x="960" y="200" width="190" height="90" as="geometry"/></mxCell>
<mxCell id="9" value="" style="{EDGE}" edge="1" parent="1" source="2" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="10" value="" style="{EDGE}" edge="1" parent="1" source="3" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="11" value="" style="{EDGE}" edge="1" parent="1" source="4" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="12" value="" style="{EDGE}" edge="1" parent="1" source="5" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="13" value="" style="{EDGE}" edge="1" parent="1" source="6" target="7"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="14" value="" style="{EDGE}" edge="1" parent="1" source="7" target="8"><mxGeometry relative="1" as="geometry"/></mxCell>
''')

save("figure-5-1-test-flow", f'''
<mxCell id="2" value="准备测试环境" style="{RECT}" vertex="1" parent="1"><mxGeometry x="80" y="80" width="170" height="55" as="geometry"/></mxCell>
<mxCell id="3" value="准备视频、图文、人脸样本" style="{RECT}" vertex="1" parent="1"><mxGeometry x="310" y="80" width="220" height="55" as="geometry"/></mxCell>
<mxCell id="4" value="功能测试" style="{RECT}" vertex="1" parent="1"><mxGeometry x="590" y="80" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="5" value="异常测试" style="{RECT}" vertex="1" parent="1"><mxGeometry x="590" y="200" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="6" value="自动化测试" style="{RECT}" vertex="1" parent="1"><mxGeometry x="310" y="200" width="150" height="55" as="geometry"/></mxCell>
<mxCell id="7" value="benchmark 辅助评估" style="{RECT}" vertex="1" parent="1"><mxGeometry x="80" y="200" width="180" height="55" as="geometry"/></mxCell>
<mxCell id="8" value="结果是否符合预期" style="{DECISION}" vertex="1" parent="1"><mxGeometry x="310" y="350" width="170" height="105" as="geometry"/></mxCell>
<mxCell id="9" value="记录问题并修正" style="{RECT}" vertex="1" parent="1"><mxGeometry x="80" y="375" width="170" height="55" as="geometry"/></mxCell>
<mxCell id="10" value="整理测试结果与局限" style="{RECT}" vertex="1" parent="1"><mxGeometry x="590" y="375" width="190" height="55" as="geometry"/></mxCell>
<mxCell id="11" value="" style="{EDGE}" edge="1" parent="1" source="2" target="3"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="12" value="" style="{EDGE}" edge="1" parent="1" source="3" target="4"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="13" value="" style="{EDGE}" edge="1" parent="1" source="4" target="5"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="14" value="" style="{EDGE}" edge="1" parent="1" source="5" target="6"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="15" value="" style="{EDGE}" edge="1" parent="1" source="6" target="7"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="16" value="" style="{EDGE}" edge="1" parent="1" source="7" target="8"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="17" value="否" style="{EDGE}" edge="1" parent="1" source="8" target="9"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="18" value="是" style="{EDGE}" edge="1" parent="1" source="8" target="10"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="19" value="修正后复测" style="{EDGE}" edge="1" parent="1" source="9" target="4"><mxGeometry relative="1" as="geometry"/></mxCell>
''')

print(f"generated {len(list(ROOT.glob('figure-*.drawio')))} drawio files in {ROOT}")
