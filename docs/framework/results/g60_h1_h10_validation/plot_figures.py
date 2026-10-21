"""仅根据已审核CSV绘图；不导入Torch，不读取模型，不重算检索。"""
from pathlib import Path
import csv,sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/"comparison_raw.csv").open(encoding="utf-8")))
epochs=[int(r["H Epoch"]) for r in rows]
assert epochs==list(range(1,11)) and "torch" not in sys.modules
def series(key):return [float(r[key]) for r in rows]
plt.rcParams.update({"font.size":15,"axes.titlesize":21,"axes.labelsize":17,"legend.fontsize":14,
                     "svg.fonttype":"none","font.family":"DejaVu Sans","savefig.facecolor":"white"})
BLUE="#2563EB";ORANGE="#EA580C";GREEN="#0F766E";PURPLE="#7C3AED"
def style(ax):
    ax.set_xticks(epochs);ax.set_xlim(.7,10.3);ax.set_xlabel("H checkpoint epoch");ax.tick_params(axis="x",labelsize=12)
    ax.grid(axis="y",alpha=.22);ax.spines[["top","right"]].set_visible(False)
def save(fig,name):
    fig.savefig(root/(name+".png"),dpi=180)
    fig.savefig(root/(name+".svg"))
    p=root/(name+".svg")
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
    plt.close(fig)
fig,ax=plt.subplots(figsize=(12.8,7.2),layout="constrained")
style(ax);ax.plot(epochs,series("G10 Joint mAP"),"o-",color=BLUE,label="G10 + H (historical)")
ax.plot(epochs,series("G60 Joint mAP"),"s-",color=ORANGE,label="G60 + H (new)")
ax.axvline(7,color="#64748B",linestyle=":",alpha=.7,label="Historical H7 selection")
ax.scatter([9],[series("G60 Joint mAP")[8]],s=130,facecolors="none",edgecolors=ORANGE,linewidths=2.2,zorder=4)
ax.annotate("H9: 40.43%",(9,series("G60 Joint mAP")[8]),xytext=(8.0,46),arrowprops={"arrowstyle":"->","color":ORANGE},color=ORANGE)
ax.set_ylabel("Joint mAP (%) — lower is stronger")
ax.set_title("H candidate adaptation to fixed G10 and G60")
ax.legend(loc="center right");fig.supxlabel("128 reused internal Validation queries; exploratory diagnostic, not Test selection",fontsize=12)
save(fig,"joint_map_g10_g60")
fig,ax=plt.subplots(figsize=(12.8,7.2),layout="constrained")
style(ax);y=series("G60 Joint extra drop pp")
ax.plot(epochs,y,"o-",color=GREEN)
for e,v in zip(epochs,y):ax.annotate(f"{v:.2f}",(e,v),xytext=(0,11),textcoords="offset points",ha="center",fontsize=12,color=GREEN)
ax.set_ylim(4.8,9.1);ax.set_ylabel("Additional mAP reduction\n(percentage points)")
ax.set_title("Additional mAP reduction from H under G60")
ax.axvline(7,color="#64748B",linestyle=":",alpha=.7,label="Historical H7")
ax.legend();fig.supxlabel("Shared G60 Image-only mAP = 48.80861282348633%; higher reduction is stronger",fontsize=12)
save(fig,"joint_extra_drop_g60")
fig,axes=plt.subplots(1,2,figsize=(12.8,7.2),layout="constrained",sharex=True)
for ax in axes:style(ax)
axes[0].plot(epochs,series("Text-only mAP"),"o-",color=PURPLE)
axes[0].set_title("H Text-only (historical)",fontsize=19)
axes[0].set_ylabel("Text-only mAP (%) — separate scale");axes[0].set_ylim(95.4,97.9)
axes[1].plot(epochs,series("G60 Joint mAP"),"s-",color=ORANGE)
axes[1].set_title("G60 + H Joint (new)",fontsize=19)
axes[1].set_ylabel("Joint mAP (%) — separate scale");axes[1].set_ylim(39.8,44.1)
for ax,i in zip(axes,[3,9]):ax.axvline(i,linestyle=":",color="#64748B")
fig.suptitle("Text-only and Joint rank candidates differently",fontsize=22)
fig.supxlabel("Panels use different, explicitly labeled y-axis ranges; all 10 measured candidates retained",fontsize=12)
save(fig,"text_only_vs_g60_joint")
print("THREE_FIGURES_PNG_SVG_PASS")
