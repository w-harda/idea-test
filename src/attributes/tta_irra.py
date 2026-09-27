"""Use official TTA attacks with IRRA features and image gradients."""
from __future__ import annotations

from math import isfinite
from pathlib import Path

import torch
from torch import nn

from .attack_scheduler import ImageAttackRequest, ImageAttackResult
from .generator_framework import IMAGE_EPSILON, project_image
from .irra_surrogate import IRRASurrogate
from .irra_victim import IMAGE_SIZE, MEAN, STD
from .tta_adapter import _tokenize, load_official_attack
from .tta_full import load_official_full_classes


def _check_source(source: IRRASurrogate, original: torch.Tensor) -> None:
    if not isinstance(source, IRRASurrogate):
        raise TypeError("TTA IRRA bridge requires IRRASurrogate")
    if (original.ndim != 4 or original.shape != (1, 3, *IMAGE_SIZE)
            or not original.is_floating_point() or not original.is_cuda
            or not torch.isfinite(original).all()
            or original.min() < 0 or original.max() > 1):
        raise ValueError("TTA IRRA image must be one finite CUDA [0,1] 384x128 image")
    if not next(source.victim.model.parameters()).is_cuda:
        raise RuntimeError("TTA IRRA source must be on CUDA")
    mean = torch.tensor(MEAN, device=original.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=original.device).view(1, 3, 1, 1)
    if not torch.equal(source.victim.mean, mean) or not torch.equal(source.victim.std, std):
        raise ValueError("TTA image normalization differs from official IRRA")


class IRRATTAImageCallback:
    """Stage 05 Image callback; only the feature/gradient source changes."""

    def __init__(self, source: IRRASurrogate, official_root: str | Path,
                 original: torch.Tensor, *, steps: int = 10,
                 transforms_per_scale: int = 6, step_size: float = 2 / 255,
                 scales: tuple[float, ...] | None = (0.5, 0.75, 1.25, 1.5)):
        _check_source(source, original)
        if (steps < 1 or transforms_per_scale < 1
                or not 0 < step_size <= IMAGE_EPSILON
                or (scales is not None and any(
                    not isfinite(scale) or scale <= 0 for scale in scales))):
            raise ValueError("invalid TTA IRRA attack parameters")
        self.original = original.detach().clone()
        self.bridge = source.eval()
        attack_type = load_official_attack(official_root)
        self.attacker = attack_type(None, _tokenize, imgs_eps=IMAGE_EPSILON,
                                    step_size=step_size)
        self.attacker.N_trans = transforms_per_scale
        self.momentum = torch.zeros_like(original)
        self.steps = steps
        self.scales = scales

    def __call__(self, request: ImageAttackRequest) -> ImageAttackResult:
        current = request.state.image
        if not isinstance(current, torch.Tensor) or current.shape != self.original.shape:
            raise ValueError("current IRRA image does not match original")
        if (current - self.original).abs().max() > IMAGE_EPSILON + 1e-6:
            raise ValueError("incoming IRRA image exceeds cumulative budget")
        word = request.state.targets[request.attribute.slot]["mentions"][0]["raw"]
        with torch.enable_grad():
            proposed, momentum = self.attacker.img_attack(
                self.bridge, [request.state.text, word], current.detach(),
                self.original, [0, 0], self.steps, self.momentum,
                current.device, scales=self.scales,
            )
        image = project_image(self.original, proposed.detach()).detach()
        self.momentum = momentum.detach()
        return ImageAttackResult(
            image, guidance={"tta_steps": self.steps, "attribute_word": word})


def vanilla_tta_irra_image_attack(
    source: IRRASurrogate, official_root: str | Path,
    original: torch.Tensor, caption: str,
) -> torch.Tensor:
    """Two official image updates without Stage 04/05 or text replacement."""
    _check_source(source, original)
    if not isinstance(caption, str) or not caption:
        raise ValueError("vanilla TTA needs a nonempty original caption")
    attack_type = load_official_attack(official_root)
    attacker = attack_type(None, _tokenize, imgs_eps=IMAGE_EPSILON,
                           step_size=2 / 255)
    attacker.N_trans = 6
    momentum = torch.zeros_like(original)
    current = original.detach().clone()
    for _ in range(2):
        with torch.enable_grad():
            proposed, momentum = attacker.img_attack(
                source, [caption], current, original, [0], 10, momentum,
                original.device, scales=(0.5, 0.75, 1.25, 1.5),
            )
        current = project_image(original, proposed.detach()).detach()
        momentum = momentum.detach()
    return current


class _FullIRRABridge(nn.Module):
    """Decode official BERT attack tokens, then encode with IRRA tokens."""

    def __init__(self, source: IRRASurrogate, bert_tokenizer):
        super().__init__()
        self.source = source
        self.bert_tokenizer = bert_tokenizer

    def inference_image(self, normalized_images):
        return self.source.inference_image(normalized_images)

    def inference_text(self, text_input):
        texts = [
            self.bert_tokenizer.decode(ids.tolist())
            .replace("[PAD]", "").replace("[CLS]", "")
            .replace("[SEP]", "").strip()
            for ids in text_input.input_ids
        ]
        return {"text_feat": self.source.encode_texts(texts)}


class IRRAFullVanillaTTA:
    """Official Image_1 -> Text_1 -> Image_2 with IRRA source features."""

    def __init__(self, source: IRRASurrogate, official_root: str | Path,
                 bert_path: str | Path, glove_path: str | Path):
        from gensim.models import KeyedVectors
        from transformers import BertForMaskedLM, BertTokenizer

        bert_path = Path(bert_path)
        glove_path = Path(glove_path)
        if not bert_path.is_dir() or not glove_path.is_file():
            raise FileNotFoundError("local BERT and GloVe resources are required")
        self.source = source
        tokenizer = BertTokenizer.from_pretrained(
            str(bert_path), local_files_only=True)
        device = source.victim.mean.device
        ref_net = BertForMaskedLM.from_pretrained(
            str(bert_path), local_files_only=True).to(device).eval()
        ref_net.requires_grad_(False)
        self.ref_net = ref_net
        glove_model = KeyedVectors.load(str(glove_path), mmap="r")
        self.glove_model = glove_model
        tta_class, attack_class = load_official_full_classes(
            official_root, glove_model)
        bridge = _FullIRRABridge(source, tokenizer).eval()
        multi_attack = attack_class(
            ref_net, tokenizer, cls=False, max_length=77,
            number_perturbation=1, topk=10, threshold_pred_score=0.3,
            imgs_eps=IMAGE_EPSILON, step_size=2 / 255,
        )
        self.attacker = tta_class(bridge, multi_attack)

    def __call__(self, original: torch.Tensor, caption: str):
        _check_source(self.source, original)
        if not isinstance(caption, str) or not caption:
            raise ValueError("full TTA needs a nonempty original caption")
        with torch.enable_grad():
            image, texts = self.attacker.attack(
                original, [caption], [0], device=original.device,
                max_length=77, scales=(0.5, 0.75, 1.25, 1.5),
            )
        if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
            raise ValueError("official TTA returned invalid text")
        return project_image(original, image.detach()).detach(), texts[0]
