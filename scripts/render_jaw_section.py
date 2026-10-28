"""【概要】sim の指(STL)を、立方体 30 mm を挟む角度で、y = 0 mm と +9 mm の断面で切って、横から見た内側の形を outputs/jaw_section.png に描く。
指の内側の歯(段差)と、立方体が当たる場所を確かめる用。使い方: .venv/bin/python scripts/render_jaw_section.py
注意: 指の読み 19.25 は、sim の尺度で「隙間の最小が 30 mm」になる角度(実機で立方体を挟んだ読みは約 13。尺度は一致していない)。
三角形の面を、平面 y = y0 で切った線分を描いている(MuJoCo の mesh_face = 三角形の頂点番号の並び)。"""
import sys, numpy as np, mujoco
sys.path.insert(0,"src")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in font_manager.findSystemFonts():
    if "NotoSansCJK" in fp or "Noto Sans CJK" in fp:
        font_manager.fontManager.addfont(fp); matplotlib.rcParams["font.family"]=font_manager.FontProperties(fname=fp).get_name()
base=mujoco.MjModel.from_xml_path("models/so101/scene_sort.xml"); d=mujoco.MjData(base)
jid=base.joint("gripper").id; qa=base.jnt_qposadr[jid]; lo,hi=base.jnt_range[jid]
ang=lo+(hi-lo)*19.25/100
d.qpos[qa]=ang; mujoco.mj_forward(base,d)
sp=d.site_xpos[base.site("gripperframe").id]; sR=d.site_xmat[base.site("gripperframe").id].reshape(3,3)
def section(gi,y0):
    mid=base.geom_dataid[gi]; a,n=base.mesh_vertadr[mid],base.mesh_vertnum[mid]; fa,fn=base.mesh_faceadr[mid],base.mesh_facenum[mid]
    w=d.geom_xpos[gi]+base.mesh_vert[a:a+n]@d.geom_xmat[gi].reshape(3,3).T; P=(w-sp)@sR*1000
    segs=[]
    for f in base.mesh_face[fa:fa+fn]:
        T=P[f]; pts=[]
        for i,j in ((0,1),(1,2),(2,0)):
            y1,y2=T[i,1]-y0,T[j,1]-y0
            if y1*y2<0:
                t=y1/(y1-y2); pts.append(T[i]+t*(T[j]-T[i]))
        if len(pts)==2: segs.append(pts)
    return segs
fig,axs=plt.subplots(1,2,figsize=(15,6),sharey=True)
for ax,y0 in zip(axs,(0.0,9.0)):
    for gi in range(base.ngeom):
        if base.geom_dataid[gi]<0 or base.geom_group[gi]!=3: continue
        nm=base.mesh(base.geom_dataid[gi]).name
        c={"wrist_roll_follower_so101_v1":"tab:blue","moving_jaw_so101_v1":"tab:orange"}.get(nm)
        if not c: continue
        for s in section(gi,y0): ax.plot([s[0][0],s[1][0]],[s[0][2],s[1][2]],c=c,lw=1)
    ax.add_patch(plt.Rectangle((-20,-0.0),30,30,fill=False,ec="red",lw=2))
    ax.set_xlim(-70,15); ax.set_ylim(-20,50); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"断面 y = {y0:+.0f} mm(青 = 固定側、橙 = 動く側、赤 = 立方体 30 mm)"); ax.set_xlabel("x [mm] 指先 0、手首側が負")
axs[0].set_ylabel("z [mm] 固定側の面 ≈ 0")
fig.savefig("outputs/jaw_section.png",dpi=100,bbox_inches="tight")
