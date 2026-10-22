"""从既有严格属性表派生类别重合，绝不替代Span级属性匹配。"""
from collections import defaultdict
import csv,json
from tbps.ablations import gh_attribute_audit as a
def group_attributes(samples,attributes):
 group=defaultdict(list)
 for x in samples:group[(x["scope"],int(x["H_epoch"]),x["row_id"])]=[]
 for x in attributes:
  k=(x["scope"],int(x["H_epoch"]),x["row_id"])
  a.need(k in group,"category row outside sample membership")
  group[k].append(x)
 return group
def main():
 d=a.OUT
 group=group_attributes(csv.DictReader((d/"h_attribute_statistics.csv").open()),csv.DictReader((d/"attribute_alignment.csv").open()))
 rows=[]
 for (scope,ep,rid),xs in group.items():
  gg={(x["semantic_axis"],x["region"]) for x in xs if x["G_selected"]=="1"}
  hh={(x["semantic_axis"],x["region"]) for x in xs if x["H_edited"]=="1"}
  he=[x for x in xs if x["H_edited"]=="1"]
  z=a.overlap(gg,hh)
  rows.append(dict(scope=scope,H_epoch=ep,row_id=rid,category_intersection=z["intersection"],category_union=z["union"],
   category_jaccard=z["jaccard"],H_edits=len(he),
   H_edits_in_G_selected_category=sum((x["semantic_axis"],x["region"]) in gg for x in he)))
 a.need(len(rows)==6528,"category cohorts complete");summary={}
 for scope in ("Train","Validation"):
  for ep in a.EPOCHS:
   rr=[x for x in rows if x["scope"]==scope and x["H_epoch"]==ep]
   matched=sum(x["H_edits_in_G_selected_category"] for x in rr);den=sum(x["H_edits"] for x in rr)
   summary[f"{scope}_H{ep}"]=dict(samples=len(rr),category_jaccard=a.dist(x["category_jaccard"] for x in rr if x["category_jaccard"] is not None),
      H_edited_same_category_count=matched,H_edit_denominator=den,H_edited_same_category_fraction=matched/den if den else None)
 a.writec("attribute_category_overlap.csv",rows);a.writej("attribute_category_overlap_summary.json",summary)
 text="\n## 附录：类别层级重合（不是同一原文属性）\n\n"
 text+="类别只取semantic_axis/region；不同词、Span、canonical即使类别相同，也不改标为EXACT。\n\n"
 text+="| 范围 | H | n | 平均类别Jaccard | H编辑类别在G选中类别中的次数/全部编辑 |\n| --- | --- | --- | --- | --- |\n"
 for k,v in summary.items():
  text+="| "+k.split("_H")[0]+" | H"+k.split("_H")[1]+" | "+str(v["samples"])+" | "+format(v["category_jaccard"]["mean"],".6f")+" | "+str(v["H_edited_same_category_count"])+"/"+str(v["H_edit_denominator"])+" |\n"
 with (d/"REPORT.md").open("a") as f:f.write(text)
 print("GH_CATEGORY_OVERLAP_DERIVATION_PASS",json.dumps(summary),flush=True)
if __name__=="__main__":main()
