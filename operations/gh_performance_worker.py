"""只在训练动作应用已固定的加速兼容桥；审计及Test仍调用原入口。"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116")
sys.path.insert(0,str(SOURCE/"src"))
import tbps
tbps.__path__.append(str(ROOT/"src/tbps"))
from tbps.performance.runtime import verify_version,install,audit_binding
def main():
    verify_version()
    action=sys.argv[sys.argv.index("--action")+1]
    if action=="train":install()
    from tbps.attribute_response.run import main as native
    native()
    if action=="audit":
        endpoint=int(sys.argv[sys.argv.index("--round")+1])
        control="/home/lzf/ldx/tmp/gh_performance_v1_20261119"
        print(json.dumps(audit_binding(endpoint,control),ensure_ascii=False))
if __name__=="__main__":main()
