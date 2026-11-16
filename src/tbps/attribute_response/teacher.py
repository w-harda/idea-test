"""共享语义/历史编码器及双向更新收益头；独立CPU Adam与有界Replay。"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from tbps.alternating.state import clone_tree, tree_sha
from .memory import AttributeMemory, key_bytes, finite_vector, STATE_DIM, ACTION_DIM
from .settings import Settings, require

class ResponseNetwork(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.semantic = nn.Embedding(256, width)
        self.history = nn.GRUCell(STATE_DIM + 3, width)
        self.trunk = nn.Sequential(nn.Linear(2 * width + STATE_DIM + ACTION_DIM, width),
                                   nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.g_to_h = nn.Linear(width, 1)
        self.h_to_g = nn.Linear(width, 1)
        for head in (self.g_to_h, self.h_to_g):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, keys, states, actions, histories, versions):
        device = self.semantic.weight.device
        require(device.type == "cpu" and states.device.type == "cpu", "training-only CPU response teacher")
        require(states.shape == (len(keys), STATE_DIM) and actions.shape == (len(keys), ACTION_DIM),
                "teacher feature shapes")
        semantic = self.semantic(torch.tensor([key_bytes(k) for k in keys])).mean(1)
        rows = []
        for events, version in zip(histories, versions):
            hidden = states.new_zeros((1, self.semantic.embedding_dim))
            for event in events:
                age = float(max(0, version - event["version"])) / (1 + max(0, version - event["version"]))
                value = states.new_tensor(event["state"] + event["response"] + [age])[None]
                hidden = self.history(value, hidden)
            rows.append(hidden)
        history = torch.cat(rows) if rows else states.new_empty((0, self.semantic.embedding_dim))
        hidden = self.trunk(torch.cat((semantic, history, states, actions), 1))
        return torch.cat((self.g_to_h(hidden), self.h_to_g(hidden)), 1)

class ResponseTeacher:
    def __init__(self, settings):
        require(isinstance(settings, Settings), "explicit response Settings")
        self.settings = settings
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
            torch.manual_seed(settings.seed)
            self.model = ResponseNetwork(settings.width)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings.teacher_lr)
        self.memory = AttributeMemory(settings.memory_capacity, settings.history)
        self.replay = []
        self.sampler = torch.Generator().manual_seed(settings.seed + 701)
        self.updates = 0

    def features(self, keys, states, actions=None, *, version):
        require(len(keys) == len(states), "one state per semantic attribute")
        if actions is None:
            actions = [[0.] * ACTION_DIM for _ in keys]
        states = [finite_vector(x, STATE_DIM) for x in states]
        actions = [finite_vector(x, ACTION_DIM) for x in actions]
        return dict(keys=list(keys), states=states, actions=actions,
                    histories=[self.memory.read(k) for k in keys], versions=[version] * len(keys))

    def predict(self, batch):
        if not batch["keys"]:
            return torch.empty((0, 2))
        with torch.no_grad():
            out = self.model(batch["keys"], torch.tensor(batch["states"]), torch.tensor(batch["actions"]),
                             batch["histories"], batch["versions"])
        require(torch.isfinite(out).all(), "finite teacher response")
        return out.detach().clone()

    def weights(self, keys, states, prior, *, version):
        batch = self.features(keys, states, version=version)
        response = self.predict(batch)[:, 0]
        prior = torch.as_tensor(prior, dtype=torch.float32).detach().cpu()
        require(prior.shape == response.shape and torch.isfinite(prior).all()
                and (prior > 0).all(), "positive original attribute prior")
        if not len(prior):
            return prior, {"features": batch, "prediction": [], "weights": []}
        logits = prior.log() + response / self.settings.weight_temperature
        scaled = len(prior) * logits.softmax(0)
        weights = 1 - self.settings.weight_mix + self.settings.weight_mix * scaled
        require(torch.isfinite(weights).all() and (weights > 0).all(), "positive mean-one weights")
        return weights.detach(), {"features": batch, "prediction": response.tolist(),
                                  "prior": prior.tolist(), "weights": weights.tolist()}

    def observe(self, batch, targets, direction, *, role="train"):
        require(role == "train" and direction in (0, 1), "Train-only directional feedback")
        targets = finite_vector(list(targets), len(batch["keys"]))
        for i, target in enumerate(targets):
            item = {k: copy.deepcopy(v[i]) for k, v in batch.items()}
            item.update(target=target, direction=direction)
            self.replay.append(item)
        self.replay = self.replay[-self.settings.replay_capacity:]
        if not self.replay:
            return {"updated": False}
        count = min(self.settings.teacher_batch, len(self.replay))
        ids = torch.randperm(len(self.replay), generator=self.sampler)[:count].tolist()
        items = [self.replay[i] for i in ids]
        keys, states, actions, histories, versions = ([x[k] for x in items] for k in
            ("keys", "states", "actions", "histories", "versions"))
        before = tree_sha(self.model.state_dict())
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(keys, torch.tensor(states), torch.tensor(actions), histories, versions)
        directions = torch.tensor([x["direction"] for x in items])
        selected = prediction[torch.arange(count), directions]
        target = torch.tensor([x["target"] for x in items]).detach()
        loss = F.smooth_l1_loss(selected, target)
        require(torch.isfinite(loss), "finite response regression")
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        require(grads and all(torch.isfinite(g).all() for g in grads), "finite teacher gradients")
        self.optimizer.step(); self.updates += 1
        require(all(torch.isfinite(p).all() for p in self.model.parameters()), "finite teacher weights")
        return {"updated": True, "update": self.updates, "teacher_before": before,
                "teacher_after": tree_sha(self.model.state_dict()), "predictions": selected.detach().tolist(),
                "targets": target.tolist(), "directions": directions.tolist(), "loss": float(loss.detach()),
                "replay_size": len(self.replay), "sample_indices": ids}

    def remember(self, key, state, response, *, version):
        self.memory.append(key, state, response, version=version)

    def state_dict(self):
        return clone_tree({"schema": "attribute-response-teacher-1", "settings": self.settings.record(),
                           "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                           "memory": self.memory.state_dict(), "replay": self.replay,
                           "sampler": self.sampler.get_state(), "updates": self.updates})

    def load_state_dict(self, value):
        require(value["schema"] == "attribute-response-teacher-1"
                and value["settings"] == self.settings.record(), "teacher schema/settings mismatch")
        require(len(value["replay"]) <= self.settings.replay_capacity
                and type(value["updates"]) is int and value["updates"] >= 0, "bounded complete teacher state")
        self.model.load_state_dict(value["model"], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(value["optimizer"]))
        self.memory.load_state_dict(value["memory"])
        for row in value["replay"]:
            key_bytes(row["keys"]); finite_vector(row["states"], STATE_DIM)
            finite_vector(row["actions"], ACTION_DIM)
            require(row["direction"] in (0, 1), "replay direction")
            finite_vector([row["target"]], 1)
        self.replay = copy.deepcopy(value["replay"]); self.sampler.set_state(value["sampler"])
        self.updates = value["updates"]
        require(tree_sha(self.state_dict()) == tree_sha(value), "full response teacher restore exact")
