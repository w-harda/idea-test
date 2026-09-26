"""Stage 06：图库级联合排名训练与冻结测试流程。生成器与合法字符规则由后续阶段注入。"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from .attack_scheduler import (
    ImageAttackResult, SelectedAttribute, TextAttackResult, plan_attack, run_attack,
)
from .rank_objective import query_utility, soft_first_hit_rank

IMAGE_EPSILON = 8 / 255


@dataclass(frozen=True)
class QueryInput:
    """生成器可见字段；不含 person ID、配对图片或 split。"""
    row_id: str
    caption: str
    plan: tuple[SelectedAttribute, ...]
    targets: Mapping[str, dict[str, Any]]


def prepare_query(record: Mapping[str, Any]) -> QueryInput:
    """复用 Stage 05 的顺序及位置校验，不对配对记录运行攻击。"""
    plan = plan_attack(record)
    run_attack(
        record, image=None,
        image_attack=lambda request: ImageAttackResult(request.state.image),
        text_attack=lambda request: TextAttackResult(request.state.text),
    )
    provenance = record.get("provenance", {})
    return QueryInput(
        record["row_id"], record["caption"], plan,
        {item.slot: deepcopy(provenance[item.slot]) for item in plan},
    )


@dataclass(frozen=True)
class CharacterEdit:
    slot: str
    offset: int  # 原始 caption 的绝对字符位置
    replacement: str


@dataclass(frozen=True)
class TextDistribution:
    """完整最终文本的候选编辑序列与可训练 logits；包含跳过选项。"""
    options: tuple[tuple[CharacterEdit, ...], ...]
    logits: Tensor


class TextGenerator(Protocol):
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def distribution(self, query: QueryInput) -> TextDistribution: ...
    def select(self, query: QueryInput) -> tuple[CharacterEdit, ...]: ...


class SourceEncoder(Protocol):
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...
    def encode_images(self, images: Tensor) -> Tensor: ...
    def encode_texts(self, texts: Sequence[str]) -> Tensor: ...


AllowedCharacters = Callable[[QueryInput, str, int], Sequence[str]]


def apply_text_edits(query: QueryInput, edits: Sequence[CharacterEdit],
                     allowed_characters: AllowedCharacters) -> str:
    """按最终文本累计计量：仅 A* 映射词内、每个槽位至多一字符。"""
    selected = {item.slot for item in query.plan}
    seen: set[str] = set()
    offsets: set[int] = set()
    characters = list(query.caption)
    for edit in edits:
        if not isinstance(edit, CharacterEdit) or edit.slot not in selected or edit.slot in seen:
            raise ValueError("text edit must target a distinct selected attribute")
        seen.add(edit.slot)
        if not isinstance(edit.offset, int) or isinstance(edit.offset, bool):
            raise ValueError("character offset must be an integer")
        if edit.offset in offsets:
            raise ValueError("two attribute edits cannot target the same character")
        offsets.add(edit.offset)
        mentions = query.targets[edit.slot]["mentions"]
        if not any(mention["start"] <= edit.offset < mention["end"] for mention in mentions):
            raise ValueError("text edit lies outside the selected attribute word")
        legal = allowed_characters(query, edit.slot, edit.offset)
        if (not isinstance(edit.replacement, str) or len(edit.replacement) != 1
                or edit.replacement == query.caption[edit.offset]
                or edit.replacement not in legal):
            raise ValueError("text edit is not a legal single-character replacement")
        characters[edit.offset] = edit.replacement
    return "".join(characters)


def project_image(original: Tensor, proposed: Tensor) -> Tensor:
    """对原图统一投影，避免多轮重置 8/255 预算。"""
    if original.shape != proposed.shape or original.ndim != 4 or original.shape[1] != 3:
        raise ValueError("image tensors must have matching [N,3,H,W] shapes")
    if (not torch.isfinite(original).all() or not torch.isfinite(proposed).all()
            or original.min() < 0 or original.max() > 1):
        raise ValueError("image pixels must be finite and original in [0,1]")
    return torch.maximum(torch.minimum(proposed, original + IMAGE_EPSILON),
                         original - IMAGE_EPSILON).clamp(0, 1)


def validate_gallery_subset(query_ids: Sequence[int | str],
                            subset_paths: Sequence[str],
                            subset_ids: Sequence[int | str],
                            full_paths: Sequence[str],
                            full_ids: Sequence[int | str]) -> None:
    """近似图库可采负图，但必须保留查询 ID 的全部原始图片。"""
    if (not query_ids or not subset_paths or not full_paths
            or len(subset_paths) != len(subset_ids)
            or len(full_paths) != len(full_ids)
            or len(set(subset_paths)) != len(subset_paths)
            or len(set(full_paths)) != len(full_paths)):
        raise ValueError("gallery paths and IDs must be unique, aligned and nonempty")
    full = dict(zip(full_paths, full_ids, strict=True))
    subset = dict(zip(subset_paths, subset_ids, strict=True))
    if any(path not in full or full[path] != person_id
           for path, person_id in subset.items()):
        raise ValueError("subset image or ID is absent from the full gallery")
    for query_id in query_ids:
        positives = {path for path, person_id in full.items() if person_id == query_id}
        if not positives or not positives.issubset(subset):
            raise ValueError("subset must retain every same-ID gallery image")
        if len(subset) == len(positives):
            raise ValueError("subset needs negative-ID gallery images")


@dataclass(frozen=True)
class TrainingBatch:
    images: Tensor  # [G,3,H,W]，全部图库图片同一预处理尺寸
    gallery_paths: tuple[str, ...]
    gallery_ids: tuple[int | str, ...]
    records: tuple[Mapping[str, Any], ...]
    query_ids: tuple[int | str, ...]
    full_gallery_paths: tuple[str, ...]
    full_gallery_ids: tuple[int | str, ...]

    def validate(self) -> None:
        if (len(self.gallery_paths) != len(self.gallery_ids)
                or len(self.gallery_paths) != len(self.images)
                or len(set(self.gallery_paths)) != len(self.gallery_paths)
                or len(self.records) != len(self.query_ids)):
            raise ValueError("batch gallery/query lengths or paths are inconsistent")
        validate_gallery_subset(self.query_ids, self.gallery_paths, self.gallery_ids,
                                self.full_gallery_paths, self.full_gallery_ids)


class GeneratorTrainer:
    """每步先更新 G_I，再以更新后图库优化 G_T 的候选期望最终排名。"""

    def __init__(self, source: SourceEncoder, image_generator: nn.Module,
                 text_generator: TextGenerator, allowed_characters: AllowedCharacters,
                 image_optimizer: torch.optim.Optimizer,
                 text_optimizer: torch.optim.Optimizer, tau: float):
        if not 0 < tau < float("inf"):
            raise ValueError("tau must be finite and positive")
        if any(parameter.requires_grad for parameter in source.parameters()):
            raise ValueError("source surrogate must be frozen")
        self.source = source.eval()
        self.image_generator = image_generator
        self.text_generator = text_generator
        self.allowed_characters = allowed_characters
        self.image_optimizer = image_optimizer
        self.text_optimizer = text_optimizer
        self.tau = tau

    def _generate_images(self, images: Tensor) -> Tensor:
        # 强制每张图片独立调用 G_I，防止批内其他图片影响测试输出。
        return torch.cat([self.image_generator(images[index:index + 1])
                          for index in range(len(images))], dim=0)

    def _scores(self, texts: Sequence[str], images: Tensor) -> Tensor:
        self.source.eval()
        return self.source.encode_texts(texts) @ self.source.encode_images(images).T

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        batch.validate()
        queries = tuple(prepare_query(record) for record in batch.records)
        with torch.no_grad():
            clean_scores = self._scores([query.caption for query in queries], batch.images)
            clean_rank = soft_first_hit_rank(clean_scores, batch.query_ids,
                                             batch.gallery_ids, self.tau)

        # G_I 的输入只有图片；另一支只输出当前查询的最终文本。
        self.image_generator.train()
        self.text_generator.eval()
        with torch.no_grad():
            fixed_texts = [apply_text_edits(query, self.text_generator.select(query),
                                           self.allowed_characters) if query.plan else query.caption
                           for query in queries]
            text_embeddings = self.source.encode_texts(fixed_texts)
        self.image_optimizer.zero_grad(set_to_none=True)
        adv_images = project_image(batch.images, self._generate_images(batch.images))
        image_scores = text_embeddings @ self.source.encode_images(adv_images).T
        image_rank = soft_first_hit_rank(image_scores, batch.query_ids,
                                         batch.gallery_ids, self.tau)
        image_loss = -query_utility(image_rank - clean_rank).mean()
        image_loss.backward()
        self.image_optimizer.step()

        # G_T 评估完整候选文本在已扰动图库中的收益，梯度只经 logits。
        self.image_generator.eval()
        self.text_generator.train()
        self.text_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            final_images = project_image(batch.images, self._generate_images(batch.images))
            gallery_embeddings = self.source.encode_images(final_images)
        expected_utilities = []
        for index, query in enumerate(queries):
            if not query.plan:
                texts = [query.caption]
                logits = None
            else:
                distribution = self.text_generator.distribution(query)
                if not distribution.options or distribution.logits.shape != (len(distribution.options),):
                    raise ValueError("text distribution needs one logit per option")
                if not torch.isfinite(distribution.logits).all():
                    raise ValueError("text logits must be finite")
                texts = [apply_text_edits(query, edits, self.allowed_characters)
                         for edits in distribution.options]
                logits = distribution.logits
            with torch.no_grad():
                scores = self.source.encode_texts(texts) @ gallery_embeddings.T
                ranks = soft_first_hit_rank(scores, [batch.query_ids[index]] * len(texts),
                                             batch.gallery_ids, self.tau)
                utilities = query_utility(ranks - clean_rank[index])
            if logits is None:
                expected_utilities.append(utilities[0])
            else:
                expected_utilities.append((torch.softmax(logits, dim=0) * utilities).sum())
        text_loss = -torch.stack(expected_utilities).mean()
        if text_loss.requires_grad:
            text_loss.backward()
            self.text_optimizer.step()
        return {"image_loss": float(image_loss.detach()),
                "text_loss": float(text_loss.detach())}

    @torch.no_grad()
    def infer_gallery(self, images: Tensor, gallery_paths: Sequence[str]) -> Tensor:
        """测试：每张图库图片独立输入一次；调用方可分批处理唯一图库路径。"""
        if len(gallery_paths) != len(images) or len(set(gallery_paths)) != len(gallery_paths):
            raise ValueError("gallery paths must be unique and match image count")
        self.image_generator.eval()
        self.text_generator.eval()
        return project_image(images, self._generate_images(images))

    @torch.no_grad()
    def infer_queries(self, records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        """测试：直接选择，不使用 ID、正确图片、source 或 victim 候选打分。"""
        self.image_generator.eval()
        self.text_generator.eval()
        queries = (prepare_query(record) for record in records)
        return tuple(apply_text_edits(query, self.text_generator.select(query),
                                      self.allowed_characters) if query.plan else query.caption
                     for query in queries)
