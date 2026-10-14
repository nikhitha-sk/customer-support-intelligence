"""Multi-task DistilBERT: one shared encoder, three classification heads (queue, priority, type).

Sharing the encoder means one forward pass serves all three predictions at inference time,
and the related tasks regularise each other during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from csi.config import settings

TASK_LABELS: dict[str, list[str]] = {
    "queue": list(settings.queues),
    "priority": list(settings.priorities),
    "type": list(settings.ticket_types),
}


class MultiTaskClassifier(nn.Module):
    def __init__(self, encoder_name_or_path: str, task_labels: dict[str, list[str]], dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.task_labels = task_labels
        self.dropout = nn.Dropout(dropout)
        # keys are namespaced: "type" would clash with nn.Module.type()
        self.heads = nn.ModuleDict({f"{t}_head": nn.Linear(hidden, len(lbls)) for t, lbls in task_labels.items()})

    def forward(self, input_ids, attention_mask) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return {task: self.heads[f"{task}_head"](cls) for task in self.task_labels}

    def save(self, directory: Path, tokenizer) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory / "encoder")
        tokenizer.save_pretrained(directory / "encoder")
        torch.save(self.heads.state_dict(), directory / "heads.pt")
        (directory / "labels.json").write_text(json.dumps(self.task_labels, indent=2))

    @classmethod
    def load(cls, directory: Path) -> tuple["MultiTaskClassifier", AutoTokenizer]:
        task_labels = json.loads((directory / "labels.json").read_text())
        model = cls(str(directory / "encoder"), task_labels)
        model.heads.load_state_dict(torch.load(directory / "heads.pt", map_location="cpu"))
        tokenizer = AutoTokenizer.from_pretrained(directory / "encoder")
        model.eval()
        return model, tokenizer
