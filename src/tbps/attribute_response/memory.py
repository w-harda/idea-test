"""以语义属性共享有界历史；禁止把PID/样本身份作为策略输入。"""
from collections import OrderedDict
import copy
import hashlib
import json
import math
from .settings import require

FIELDS = ("semantic_axis", "canonical_value", "region")
STATE_DIM = 12
ACTION_DIM = 6

def semantic_key(attribute):
    if not isinstance(attribute, dict):
        attribute = vars(attribute)
    require(all(isinstance(attribute.get(k), str) and attribute[k] for k in FIELDS),
            "complete canonical attribute required")
    return json.dumps({k: attribute[k] for k in FIELDS}, ensure_ascii=False, sort_keys=True)

def key_bytes(key):
    require(isinstance(key, str), "semantic key")
    data = json.loads(key)
    require(set(data) == set(FIELDS) and all(isinstance(v, str) and v for v in data.values()),
            "semantic-only memory key; no sample/PID")
    return list(hashlib.sha256(key.encode()).digest()[:16])

def finite_vector(value, size):
    require(isinstance(value, (list, tuple)) and len(value) == size,
            "exact numeric feature dimension")
    require(all(type(v) in (float, int) and math.isfinite(v) for v in value),
            "finite numeric features")
    return [float(v) for v in value]

class AttributeMemory:
    def __init__(self, capacity, horizon):
        self.capacity, self.horizon = capacity, horizon
        self.records = OrderedDict()
        self.clock = 0

    def read(self, key):
        key_bytes(key)
        return copy.deepcopy(self.records.get(key, []))

    def append(self, key, state, response, *, version, role="train"):
        require(role == "train", "attribute memory accepts Train feedback only")
        key_bytes(key)
        state = finite_vector(state, STATE_DIM)
        response = finite_vector(response, 2)
        require(type(version) is int and version >= 0, "model round version")
        self.clock += 1
        record = {"state": state, "response": response, "version": version, "clock": self.clock}
        current = self.records.pop(key, [])
        require(not current or version >= current[-1]["version"], "memory cannot travel to older model version")
        self.records[key] = (current + [record])[-self.horizon:]
        while len(self.records) > self.capacity:
            self.records.popitem(last=False)

    def state_dict(self):
        return {"schema": "attribute-memory-1", "capacity": self.capacity, "horizon": self.horizon,
                "clock": self.clock, "records": list(copy.deepcopy(self.records).items())}

    def load_state_dict(self, value):
        require(value["schema"] == "attribute-memory-1"
                and value["capacity"] == self.capacity and value["horizon"] == self.horizon,
                "memory configuration/source mismatch")
        require(type(value["clock"]) is int and value["clock"] >= 0, "memory clock")
        records = OrderedDict()
        for key, events in value["records"]:
            key_bytes(key)
            require(key not in records and 0 < len(events) <= self.horizon, "unique bounded memory")
            previous = -1
            for e in events:
                require(set(e) == {"state", "response", "version", "clock"}, "complete memory event")
                finite_vector(e["state"], STATE_DIM); finite_vector(e["response"], 2)
                require(type(e["clock"]) is int and 0 < e["clock"] <= value["clock"]
                        and type(e["version"]) is int and e["version"] >= previous,
                        "memory time/version continuity")
                previous = e["version"]
            records[key] = copy.deepcopy(events)
        require(len(records) <= self.capacity, "bounded semantic memory")
        self.clock, self.records = value["clock"], records
