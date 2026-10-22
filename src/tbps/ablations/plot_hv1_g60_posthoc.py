"""仅从本轮派生CSV/JSON绘图，不导入Torch或读取模型/原始缓存。"""
from pathlib import Path
import csv,json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"docs/framework/results/hv1_g60_posthoc_analysis";FIG=OUT/"figures"
FIG.mkdir(exist_ok=True)
def rows(name):return list(csv.DictReader((OUT/name).open()))
s=json.loads((OUT/"statistics.json").read_text())
t=[x for x in s["training_epochs"] if x["epoch"]>=11];ep=[x["epoch"] for x in t]
v=s["validation"];ve=[x["H_epoch"] for x in v]
actions=rows("action_epoch_statistics.csv");hist=rows("action_distribution.csv")
C={"sampled":"#0072B2","greedy":"#D55E00","Image-only":"#777777","Joint":"#0072B2","Text-only":"#009E73"}
plt.rcParams.update({"font.size":11,"axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":10,
                     "svg.fonttype":"none","figure.dpi":150,"savefig.dpi":150})
def frame(nr=1,nc=1):
    fig,ax=plt.subplots(nr,nc,figsize=(12.8,7.2),layout="constrained",squeeze=False)
    for a in ax.flat:a.grid(alpha=.20);a.set_axisbelow(True)
    return fig,ax
def save(fig,name):
    fig.savefig(FIG/(name+".png"));fig.savefig(FIG/(name+".svg"))
    p=FIG/(name+".svg");p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines())+"\n")
    plt.close(fig)
def trainline(ax,key,label=None,color=None):
    ax.plot(ep,[x[key] for x in t],label=label or key,color=color,linewidth=1.8)
    ax.set_xlabel("Absolute H epoch (G60 adaptation)")
def valphase(ax):
    ax.axvspan(10,60,alpha=.035,color="black")
    ax.axvline(10,linestyle=":",color="#555555",linewidth=1)
    ax.set_xticks([1,5,10,20,30,40,50,60]);ax.set_xlabel("H epoch; training guide changes after H10")
fig,ax=frame(2,2)
for a,term,title in zip(ax.flat,["total","R_sim","R_rank","synergy_increment"],
                       ["Training reward","Paired similarity drop","Best-positive margin drop","Diagnostic synergy (not optimized)"]):
    for mode in ("sampled","greedy"):trainline(a,mode+"_"+term+"_mean",mode,C[mode])
    a.set_ylabel("Epoch mean (similarity units)");a.set_title(title);a.legend()
save(fig,"training_rewards")
fig,ax=frame(2,2)
a=ax[0,0];trainline(a,"SCST_loss_mean","mean",C["sampled"])
a.fill_between(ep,[x["SCST_loss_p25"] for x in t],[x["SCST_loss_p75"] for x in t],alpha=.15,label="IQR")
a.axhline(0,color="#777777",linewidth=.8);a.set_ylabel("SCST loss");a.legend()
a=ax[0,1];trainline(a,"advantage_std","Advantage std",C["sampled"]);a.set_ylabel("Reward units")
a=ax[1,0];a.plot(ep,[100*x["zero_gradient_rate"] for x in t],label="Zero gradient")
a.plot(ep,[100*x["zero_advantage_rate"] for x in t],"--",label="Zero advantage")
a.set_xlabel("Absolute H epoch");a.set_ylabel("Eligible updates (%)");a.legend()
a=ax[1,1];trainline(a,"gradient_norm_mean","mean");trainline(a,"gradient_norm_p95","95th percentile")
a.set_ylabel("H gradient L2 norm");a.legend()
save(fig,"training_loss_gradient")
fig,ax=frame(2,2)
for mode in ("sampled","greedy"):
    trainline(ax[0,0],mode+"_mean_edits",mode+" edits",C[mode])
ax[0,0].plot(ep,[x["greedy_mean_budget"] for x in t],":",color="#555555",label="K budget")
ax[0,0].set_ylabel("Characters / caption");ax[0,0].legend()
for mode in ("sampled","greedy"):
    ax[0,1].plot(ep,[100*x[mode+"_NOOP_rate"] for x in t],label=mode+" NOOP")
ax[0,1].set_ylabel("Eligible captions (%)");ax[0,1].set_xlabel("Absolute H epoch");ax[0,1].legend()
for mode in ("sampled","greedy"):
    ax[1,0].plot(ep,[100*x[mode+"_STOP_rollout_rate"] for x in t],label=mode+" STOP")
ax[1,0].set_ylabel("Eligible rollouts (%)");ax[1,0].set_xlabel("Absolute H epoch");ax[1,0].legend()
ax[1,1].plot(ep,[100*x["same_text_rate"] for x in t],label="Same final text")
ax[1,1].plot(ep,[100*x["same_actions_rate"] for x in t],label="Same edit/decision trace")
ax[1,1].set_ylabel("Sampled vs greedy agreement (%)");ax[1,1].set_xlabel("Absolute H epoch");ax[1,1].legend()
save(fig,"training_policy")
fig,ax=frame(2,2)
for a,key,title in [(ax[0,0],"Text-only mAP","Text-only mAP"),(ax[0,1],"Joint mAP","Joint mAP"),
                    (ax[1,0],"Text-only R@1","Text-only R@1"),(ax[1,1],"Joint R@1","Joint R@1")]:
    a.plot(ve,[x[key] for x in v],"o-",color=C["Joint" if "Joint" in key else "Text-only"])
    if key=="Joint mAP":a.axhline(s["controls"]["G60_Image_only"]["mAP"],color=C["Image-only"],linestyle="--",label="G60 Image-only");a.legend()
    if key=="Text-only mAP":a.axhline(100,color=C["Image-only"],linestyle="--",label="Clean");a.legend()
    a.set_title(title);a.set_ylabel("Retrieval metric (%)");valphase(a)
save(fig,"validation_metrics")
va={int(x["H_epoch"]):x for x in actions if x["scope"]=="Validation"}
fig,ax=frame(2,2)
ax[0,0].plot(ve,[float(va[e]["mean_edits"]) for e in ve],"o-",label="Actual edits")
ax[0,0].plot(ve,[float(va[e]["mean_budget"]) for e in ve],"o--",label="K budget")
ax[0,0].set_ylabel("Characters / all 128 captions");ax[0,0].legend()
for key,label in [("NOOP_rate","Caption NOOP"),("STOP_rollout_rate","Rollout STOP"),("position_NOOP_rate","Position NOOP / position decisions")]:
    ax[0,1].plot(ve,[100*float(va[e][key]) for e in ve],"o-",label=label)
ax[0,1].set_ylabel("Rate (%; denominators in legend)");ax[0,1].legend()
cats=["0","1","2","3+"]
for e in [9,20,40,60]:
    counts={x["value"]:int(x["count"]) for x in hist if x["scope"]=="Validation" and int(x["H_epoch"])==e and x["dimension"]=="edits"}
    ys=[counts.get("0",0),counts.get("1",0),counts.get("2",0),sum(n for k,n in counts.items() if int(k)>=3)]
    ax[1,0].plot(cats,[100*n/128 for n in ys],"o-",label="H"+str(e))
ax[1,0].set_ylabel("All validation captions (%)");ax[1,0].set_xlabel("Actual edited characters");ax[1,0].legend()
ax[1,1].plot(ve,[float(va[e]["mean_token_length_delta"]) for e in ve],"o-")
ax[1,1].set_ylabel("BPE content-token length change");ax[1,1].set_title("127 audited captions / candidate")
for a in [ax[0,0],ax[0,1],ax[1,1]]:valphase(a)
save(fig,"validation_behavior")
fig,ax=frame(1,2)
for a,dimension,n in [(ax[0,0],"replacement",5),(ax[0,1],"axis_region",5)]:
    rr=[x for x in hist if x["scope"]=="G60 adaptation" and x["rollout"]=="greedy" and x["dimension"]==dimension]
    top=Counter()
    for x in rr:top[x["value"]]+=int(x["count"])
    names=[x for x,_ in top.most_common(n)];series=[]
    for name in names:
        series.append([100*sum(float(x["fraction"]) for x in rr if int(x["H_epoch"])==e and x["value"]==name) for e in ep])
    series.append([100-sum(v[i] for v in series) for i in range(len(ep))])
    a.stackplot(ep,*series,labels=names+["Other"],alpha=.85)
    a.set_ylim(0,100);a.set_xlabel("Absolute H epoch");a.set_ylabel("Share of actual edits (%)")
    a.set_title("Unicode replacements" if dimension=="replacement" else "Project attribute labels")
    a.legend(loc="upper center",ncol=2,fontsize=9)
save(fig,"replacement_attribute_distribution")
qrows=rows("per_query_analysis.csv")
by={(x["candidate"],x["group"],int(x["query_index"])):x for x in qrows}
fig,ax=frame(2,2)
for col,(a,b) in enumerate([("H9","H20"),("H20","H60")]):
    d=[100*(float(by[(b,"Joint",i)]["AP"])-float(by[(a,"Joint",i)]["AP"])) for i in range(128)]
    ax[0,col].hist(d,bins=np.linspace(-100,100,41),color="#0072B2",edgecolor="white")
    ax[0,col].set_xlabel("AP("+b+") - AP("+a+") (pp)");ax[0,col].set_ylabel("Query count")
    ax[1,col].scatter(range(128),d,s=16,color=np.where(np.asarray(d)>0,"#D55E00","#0072B2"))
    ax[1,col].axhline(0,color="black",lw=.7);ax[1,col].set_xlabel("Original query index")
    ax[1,col].set_ylabel("AP difference (pp); positive = weaker attack")
save(fig,"per_query_AP_changes")
fig,ax=frame();a=ax[0,0]
names=["H7","H9","H10","H20","H40","H60"]
heat=np.array([[100*(float(by[(name,"Joint",i)]["AP"])-float(by[("Image-only","Image-only",i)]["AP"])) for i in range(128)] for name in names])
im=a.imshow(heat,aspect="auto",cmap="coolwarm",vmin=-100,vmax=100,interpolation="none")
a.set_yticks(range(len(names)),names);a.set_xlabel("Original validation query index");a.grid(False)
a.set_title("Joint AP - Image-only AP (negative = stronger attack)")
fig.colorbar(im,ax=a,label="AP difference (percentage points)")
save(fig,"per_query_AP_heatmap")
fig,ax=frame(1,2)
pairs=["H9_vs_H20","H20_vs_H60"]
for col,metric in enumerate(["bootstrap_AP","bootstrap_R1"]):
    a=ax[0,col]
    for i,name in enumerate(pairs):
        z=s["query_comparisons"][name][metric];point=z["difference_b_minus_a_pp"]
        a.errorbar(point,i,xerr=[[point-z["CI95_lower_pp"]],[z["CI95_upper_pp"]-point]],fmt="o",capsize=5)
    a.set_yticks(range(2),["H20 - H9","H60 - H20"]);a.axvline(0,color="#555555",linestyle="--")
    a.set_xlabel(("mAP" if metric=="bootstrap_AP" else "R@1")+" difference (pp)")
    a.set_title("Exploratory PID-cluster bootstrap, 20,000 draws")
save(fig,"paired_PID_bootstrap")
fig,ax=frame(1,2)
for mode in ("sampled","greedy"):trainline(ax[0,0],mode+"_total_mean",mode,C[mode])
ax[0,0].set_title("Online training reward");ax[0,0].set_ylabel("Mean reward");ax[0,0].legend()
nv=[x for x in v if x["H_epoch"]>=20]
ax[0,1].plot([x["H_epoch"] for x in nv],[x["Joint mAP"] for x in nv],"o-",color=C["Joint"])
ax[0,1].set_xlabel("Fixed validation checkpoint epoch");ax[0,1].set_ylabel("Joint mAP (%; lower = stronger)")
ax[0,1].set_title("Saved checkpoint validation; no monotonic improvement")
save(fig,"reward_validation_alignment")
fig,ax=frame(1,2)
old=[x for x in v if x["H_epoch"]<=10]
ax[0,0].plot([x["H_epoch"] for x in old],[x["G10 Joint mAP"] for x in old],"o-",label="G10 + H")
ax[0,0].plot([x["H_epoch"] for x in old],[x["Joint mAP"] for x in old],"o-",label="G60 + H")
ax[0,0].set_xlabel("H epoch (all trained with G10 guide)");ax[0,0].set_ylabel("Joint mAP (%)");ax[0,0].legend()
ax[0,1].plot(ve,[x["Joint_extra_drop_pp"] for x in v],"o-")
ax[0,1].set_ylabel("G60 Image-only mAP - Joint mAP (pp)");valphase(ax[0,1])
save(fig,"G10_G60_validation_context")
fig,ax=frame(1,2)
ax[0,0].plot(ep,[x["gradient_norm_mean"]/(1-x["zero_gradient_rate"]) for x in t],label="Mean conditional on nonzero gradient")
ax[0,0].set_ylabel("H gradient L2 norm");ax[0,0].set_xlabel("Absolute H epoch");ax[0,0].legend()
for key,label in [("parameter_delta_l2_mean","Mean"),("parameter_delta_l2_median","Median")]:
    trainline(ax[0,1],key,label)
ax[0,1].set_ylabel("Actual H parameter change L2 / Adam step");ax[0,1].legend()
save(fig,"parameter_updates")
print("H60_POSTHOC_FIGURES_PASS",len(list(FIG.glob("*.png"))))
