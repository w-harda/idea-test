"""G/H属性审计的既有CSV绘图，PNG/SVG，无模型。"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parents[3]/"docs/framework/results/gh_attribute_alignment_audit"
F=D/"figures"
COLORS={9:"#0072B2",20:"#D55E00",60:"#009E73"}
def rows(n):return list(csv.DictReader((D/n).open()))
def save(fig,name):
    fig.set_size_inches(12.8,7.2);fig.savefig(F/(name+".png"),dpi=150,bbox_inches=None)
    fig.savefig(F/(name+".svg"),bbox_inches=None)
    p=F/(name+".svg");p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines())+"\n")
    plt.close(fig)
def style(ax,x,y,title):
    ax.set(xlabel=x,ylabel=y,title=title);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
def main():
    F.mkdir(exist_ok=False)
    s=json.loads((D/"statistics.json").read_text());u=json.loads((D/"supplementary_statistics.json").read_text())
    h=rows("h_attribute_statistics.csv");g=rows("g_attribute_statistics.csv")
    plt.rcParams.update({"font.size":11,"axes.titlesize":13,"legend.fontsize":10,"svg.fonttype":"none"})
    fig,axs=plt.subplots(1,2,constrained_layout=True)
    for ax,role,scope in zip(axs,("train","validation"),("Train","Validation")):
        gg=s["G"][role];base=[gg["candidates"]["mean"],gg["positive"]["mean"],gg["K"]["mean"]]
        x=np.arange(3);ax.bar(x-.27,base,.18,color="#777777",label="G")
        for i,ep in enumerate((9,20,60)):
            z=s["H"][f"{scope}_H{ep}"];v=[z["candidate_count"]["mean"],z["legal_count"]["mean"],z["H_K"]["mean"]]
            ax.bar(x-.09+i*.18,v,.18,color=COLORS[ep],label=f"H{ep}")
        ax.set_xticks(x,["Resolved / raw*","Eligible / positive*","Natural K"])
        style(ax,"*G raw, H resolved; eligibility differs","Mean attributes / image",f"{scope}: G vs H budget stages")
        ax.legend()
    save(fig,"candidate_and_budget_stages")
    fig,axs=plt.subplots(1,2,constrained_layout=True)
    for ax,scope,role in zip(axs,("Train","Validation"),("train","validation")):
        x=np.arange(8);gg=s["G"][role]["K_counts"];n=s["G"][role]["samples"]
        ax.plot(x,[100*gg.get(str(k),0)/n for k in x],"o-",color="#555555",label="G K")
        for ep in (9,20,60):
            z=s["H"][f"{scope}_H{ep}"];cnt=z["H_K_counts"]
            ax.plot(x,[100*cnt.get(str(k),0)/z["samples"] for k in x],"o-",label=f"H{ep} K",color=COLORS[ep])
        ax.set_xticks(x);style(ax,"K","Samples (%)",scope);ax.legend()
    save(fig,"natural_K_distributions")
    fig,axs=plt.subplots(1,2,constrained_layout=True)
    for ax,scope in zip(axs,("Train","Validation")):
        x=np.arange(3);z=[s["H"][f"{scope}_H{ep}"] for ep in (9,20,60)]
        e=np.array([v["edit_count"]["mean"] for v in z]);p=np.array([v["position_NOOP"]["mean"] for v in z]);t=np.array([v["STOP_unused_slots"]["mean"] for v in z])
        ax.bar(x,e,color="#0072B2",label="Actual edits");ax.bar(x,p,bottom=e,color="#E69F00",label="Position NOOP slots")
        ax.bar(x,t,bottom=e+p,color="#CC79A7",label="STOP unused slots");ax.set_xticks(x,["H9","H20","H60"])
        style(ax,"Logged greedy policy","Mean slots / caption",scope+": exact K accounting");ax.legend()
    save(fig,"K_slot_accounting")
    fig,axs=plt.subplots(1,2,constrained_layout=True)
    xx=[9,20,60]
    for scope,linestyle in (("Train","--"),("Validation","-")):
        z=[s["H"][f"{scope}_H{ep}"] for ep in xx]
        axs[0].plot(xx,[v["G_selected_H_edited_jaccard"]["mean"] for v in z],"o"+linestyle,label=scope+" Jaccard")
        axs[1].plot(xx,[u["budget_decomposition_and_rank"][f"{scope}_H{ep}"]["spearman"]["mean"] for ep in xx],"o"+linestyle,label=scope+" mean Spearman")
    for ax in axs:ax.set_xticks(xx);ax.legend()
    style(axs[0],"H epoch (different training guide)","Mean Jaccard","G selected vs H actually edited")
    style(axs[1],"H epoch","Mean per-caption Spearman","Strict same-candidate ranking only")
    save(fig,"selection_and_ranking_alignment")
    fig,axs=plt.subplots(1,2,constrained_layout=True)
    for ax,scope in zip(axs,("Train","Validation")):
        x=np.arange(3);zs=[u["budget_decomposition_and_rank"][f"{scope}_H{ep}"] for ep in (9,20,60)]
        keys=["candidate_component_G_to_common","importance_component_common","candidate_component_common_to_H"]
        labels=["G set -> common set","Score shape on common set","Common set -> H set"]
        for i,(k,l) in enumerate(zip(keys,labels)):ax.bar(x+(i-1)*.23,[z[k]["mean"] for z in zs],.23,label=l)
        ax.axhline(0,color="black",lw=.7);ax.set_xticks(x,["H9","H20","H60"])
        style(ax,"H epoch","Mean K contribution","Algebraic G K - H K: "+scope);ax.legend(fontsize=8)
    save(fig,"K_gap_algebra")
    cat=rows("attribute_category_statistics.csv");cats=["gender/person","hair_length/head","garment_type/upper","garment_type/lower","color/upper","color/lower","bag/carried","backpack/carried"]
    fig,ax=plt.subplots(constrained_layout=True);x=np.arange(len(cats))
    for j,(cohort,ep,label,color) in enumerate([("G_validation","","G selected","#777777")]+[("Validation_H_edited",str(e),f"H{e} edits",COLORS[e]) for e in (9,20,60)]):
        lookup={v["axis"]+"/"+v["region"]:100*float(v["fraction"]) for v in cat if v["scope"]==cohort and v["H_epoch"]==ep}
        ax.bar(x+(j-1.5)*.19,[lookup.get(c,0) for c in cats],.19,label=label,color=color)
    ax.set_xticks(x,cats,rotation=18,ha="right");style(ax,"Recorded semantic axis / region","Share of selected / edited actions (%)","Validation categories: no image region localization claim");ax.legend()
    save(fig,"attribute_category_comparison")
    fig,axs=plt.subplots(1,2,constrained_layout=True)
    x=np.arange(3);rx=[u["canonical_reextraction"][str(ep)] for ep in (9,20,60)]
    found=np.array([z["semantic_refound"] for z in rx]);own=np.array([z["own_edit_not_refound"] for z in rx]);other=np.array([z["unedited_not_refound"] for z in rx])
    axs[0].bar(x,found,label="Original semantic anchor re-found",color="#009E73")
    axs[0].bar(x,own,bottom=found,label="Edited attr not re-found",color="#D55E00")
    axs[0].bar(x,other,bottom=found+own,label="Unedited attr not re-found",color="#CC79A7")
    axs[0].set_xticks(x,["H9","H20","H60"]);style(axs[0],"Existing adversarial captions","Original attributes (612 / candidate)","Frozen rule re-extraction");axs[0].legend(fontsize=8)
    axs[1].bar(x,[z["UTF8_start_shifted_attributes"] for z in rx],color=[COLORS[e] for e in (9,20,60)])
    axs[1].set_xticks(x,["H9","H20","H60"]);style(axs[1],"Existing adversarial captions","Attributes with shifted UTF-8 start offset","Character spans stay fixed; byte spans move")
    save(fig,"unicode_anchor_mapping")
    fig,axs=plt.subplots(1,2,constrained_layout=True)
    for ep in (9,20,60):
        rr=[z for z in h if z["scope"]=="Validation" and int(z["H_epoch"])==ep and z["G_selected_H_edited_jaccard"]!=""]
        axs[0].scatter([float(z["G_selected_H_edited_jaccard"]) for z in rr],[float(z["AP_delta_vs_image_only_pp"]) for z in rr],s=16,alpha=.45,label=f"H{ep}",color=COLORS[ep])
        rr=[z for z in h if z["scope"]=="Validation" and int(z["H_epoch"])==ep]
        for slot in sorted({int(z["unused_K"]) for z in rr}):
            vals=[float(z["AP_delta_vs_image_only_pp"]) for z in rr if int(z["unused_K"])==slot]
            axs[1].scatter(np.full(len(vals),slot)+(ep-30)*.003,vals,s=15,alpha=.4,color=COLORS[ep])
    style(axs[0],"G-selected / H-edited Jaccard","AP - Image-only AP (percentage points)","Observed association; no K intervention")
    style(axs[1],"Unused K slots (offset for visibility)","AP - Image-only AP (percentage points)","Existing Validation samples; negative = stronger")
    axs[0].legend()
    save(fig,"overlap_budget_and_AP")
    print("GH_ATTRIBUTE_PLOTS_PASS 8 PNG 8 SVG")
if __name__=="__main__":main()
