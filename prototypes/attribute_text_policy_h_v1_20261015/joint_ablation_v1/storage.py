"""复用Phase5已验证原子文本单元；仅允许独立adv_text，不复制图库。"""
from pathlib import Path
from phase5_joint.storage import Store as Phase5Store

class TrialStore(Phase5Store):
    def audit(self,complete=True):
        self.check_pending()
        features=self.root/"features"
        if features.exists() and any(p.name!="adv_text" for p in features.iterdir()):
            raise ValueError("ablation cache only permits adv_text")
        expected={self.path("adv_text",i).name for i,_ in self.parts("adv_text")}
        folder=features/"adv_text"
        actual={p.name for p in folder.iterdir() if p.is_dir()} if folder.exists() else set()
        if actual-expected:raise ValueError("unknown ablation text unit")
        for i,rows in self.parts("adv_text"):
            data=self.load("adv_text",i,rows)
            if complete and data is None:raise ValueError("incomplete ablation text units")
        return {"adv_text":len(actual)}
