"""Stage 4B-2预先冻结条件，只校验计划；本模块不启动重放或排名。"""
from ..artifacts import ArtifactError,read_json
FIXED={"schema":"tbps-historical-replay-acceptance-1","execution_authorized":False,
"torch":"2.5.1+cu124","device":"cpu","dtype":"torch.float32","cpu_threads":2,"query_chunk":128,
"gallery_count":3074,"query_count":6156,"max_rss_bytes":2147483648,"max_seconds":1800,
"feature_values":"exact","score_max_abs_error":1e-6,"official_ranking_indices":"exact",
"metric_atol_percentage_points":1e-5,"DR_atol_percentage_points":1e-5,
"ranking_change_or_tolerance_relaxation":False,"adversarial_pixel_sha256":"UNKNOWN","new_attack_generated":False}

def validate_plan(path):
    plan=read_json(path,"tbps-historical-replay-acceptance-1")
    if plan!=FIXED:raise ArtifactError("frozen replay environment/resource/equivalence plan changed")
    return plan
