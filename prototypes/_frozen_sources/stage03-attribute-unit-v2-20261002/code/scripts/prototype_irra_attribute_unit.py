#!/usr/bin/env python3
"""9-caption frozen IRRA hidden-state 等预算验证；不接入正式阶段。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from pathlib import Path
from functools import partial
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('TORCH_HOME', '/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME', '/home/lzf/ldx/cache/huggingface')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from attributes.ap_gallery_baseline import atomic_json, sha256, write_path
from attributes.irra_attribute_unit_prototype import AttributeUnitProbe
from attributes.irra_paired_rank_drop import load_cuhk_paired_split
from attributes.irra_victim import FrozenIRRA


ROWS = ('cuhk:0:0', 'cuhk:0:1', 'cuhk:1:0', 'cuhk:18:0', 'cuhk:21:0',
        'cuhk:326:0', 'cuhk:3828:0', 'cuhk:99:1')


def run(args):
    output = write_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    protocol = load_cuhk_paired_split(args.annotation, split='train')
    previous = json.loads(args.reference.read_text(encoding='utf-8'))
    payload = torch.load(args.gallery_cache, map_location='cpu', weights_only=True)
    spec = payload['spec']
    assert sha256(args.gallery_cache) == previous['gallery_cache_sha256']
    assert spec == previous['gallery_spec']
    assert spec['split'] == 'train' and spec['gallery_paths'] == list(protocol.gallery_paths)
    assert spec['annotation_sha256'] == sha256(args.annotation)
    assert spec['checkpoint_sha256'] == sha256(args.checkpoint)
    assert spec['config_sha256'] == sha256(args.config)
    assert spec['torch_version'] == str(torch.__version__)
    root = Path(__file__).resolve().parents[1]
    assert spec['encoder_adapter_sha256'] == sha256(root / 'src/attributes/irra_victim.py')
    for name, digest in spec['source_sha256'].items():
        assert digest == sha256(args.irra_repo / name)
    # schoolbag 的第一个真实train caption预先按文字条件选取，不按梯度/结果择优。
    bag_query = next(q for q in protocol.captions if re.search(r'\bschoolbag\b', q.caption, re.I))
    row_ids = (*ROWS, bag_query.row_id)
    queries = {q.row_id: q for q in protocol.captions if q.row_id in row_ids}
    assert len(queries) == 9 and len(protocol.gallery_paths) == 34054
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(args.irra_repo.resolve()))
    official = importlib.import_module('model.build')
    official.build_CLIP_from_openai_pretrained = partial(
        official.build_CLIP_from_openai_pretrained,
        download_root=str(write_path(Path('/home/lzf/ldx/cache/clip'))))
    victim = FrozenIRRA(args.irra_repo, args.checkpoint, args.config, device='cuda').eval()
    gallery = payload['features'].to('cuda')
    assert tuple(victim.model.base_model.positional_embedding.shape) == (77, 512)
    assert len(victim.model.base_model.transformer.resblocks) == 12
    results = []
    # 官方fp16真实注入预算审计；fp32用于区分量化与算法/非线性问题。
    # fp32转换只提升已加载权重的表示精度，不更新任何参数；同一既有图库保持不变。
    for precision in ('official_fp16', 'fp32_control'):
        if precision == 'fp32_control':
            victim.model.float()
        for row_id in row_ids:
            query = queries[row_id]
            probe = AttributeUnitProbe(victim, gallery, query.paired_gallery_index, tau=.02)
            result = probe.run(query.caption, layer=8, epsilons=(.01, .05, .1))
            if row_id in ('cuhk:99:1', 'cuhk:326:0'):
                attrs = {r['slot']: r for r in result['attribute_results']}
                hair, clothing = attrs['hair_length'], attrs['upper_clothing_length']
                assert hair['canonical'] == clothing['canonical'] == 'short'
                assert hair['support_token_indices'] != clothing['support_token_indices']
                if row_id == 'cuhk:99:1':
                    assert set(hair['evidence_token_indices']).isdisjoint(clothing['evidence_token_indices'])
            if row_id == bag_query.row_id:
                backpack = next(r for r in result['attribute_results'] if r['slot'] == 'backpack')
                words = {}
                for token in backpack['support_tokens']:
                    words.setdefault((token['word_start'], token['word_end']), []).append(token)
                assert any(len(tokens) > 1 for tokens in words.values()), 'schoolbag must use genuine multi-subword BPE'
            result.update(row_id=row_id, paired_image=query.image,
                          paired_gallery_index=query.paired_gallery_index, precision=precision)
            results.append(result)
            print(f'{precision} {row_id} F={result["F"]:.8f} cosine={result["paired_cosine"]:.8f}', flush=True)
            for attr in result['attribute_results']:
                errors = [(r['epsilon'], round(r['effective_budget_relative_error'], 5),
                           round(r['relative_prediction_error'], 5)) for r in attr['interventions']]
                print(f'  {attr["slot"]}={attr["canonical"]} support={attr["support_token_indices"]} '
                      f'S={attr["S"]} (epsilon,budget_error,prediction_error)={errors}', flush=True)
    comparisons = []
    for row_id in ('cuhk:99:1', 'cuhk:326:0'):
        query = queries[row_id]
        probe = AttributeUnitProbe(victim, gallery, query.paired_gallery_index)
        for layer in (6, 10):
            result = probe.run(query.caption, layer=layer, epsilons=(.05,))
            result.update(row_id=row_id, precision='fp32_control')
            comparisons.append(result)
    report = {
        'prototype_only': True, 'objective': 'paired_soft_rank', 'tau': .02,
        'tau_status': 'prototype setting, not frozen; no tau search',
        'query_count': len(queries), 'row_ids': row_ids, 'primary_layer': 8,
        'epsilons': [.01, .05, .1], 'precision_modes': ['official_fp16', 'fp32_control'],
        'gallery_count': len(protocol.gallery_paths), 'gallery_cache_sha256': sha256(args.gallery_cache),
        'gallery_spec': spec, 'image_encodings': 0, 'new_removal_evaluations': 0,
        'precision_policy': 'native fp16 plus frozen model float32 arithmetic control; same fixed fp16-derived image cache',
        'support_policy': 'binary union of mapped minimal recorded evidence and recorded linked_objects; no inferred object',
        'score_policy': 'Frobenius norm of masked hidden-state gradient; no weights or length normalization',
        'code_sha256': {name: sha256(root / name) for name in
                        ('src/attributes/irra_attribute_unit_prototype.py',
                         'scripts/prototype_irra_attribute_unit.py',
                         'src/attributes/irra_token_alignment.py',
                         'src/attributes/irra_soft_rank_gradient.py')},
        'results': results, 'layer_comparisons': comparisons,
    }
    atomic_json(output, report)
    print(f'report: {output}', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('annotation', 'irra-repo', 'checkpoint', 'config', 'gallery-cache', 'reference', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    run(cli.parse_args())
