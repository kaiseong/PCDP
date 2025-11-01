# SPEC_EAG_stack_dataset.py
# -*- coding: utf-8 -*-
"""
SPEC_EAG_stack_dataset
----------------------
A thin wrapper around your existing SPEC stack dataset that *adds* a
`teacher_tail` field used for EAG training.

Why a wrapper?
- We don't change your proven data pipeline.
- We only compute the previous-tail of length Hc from the already-loaded `action`.
- If `action` is not available in a sample, we simply skip (policy will fallback to actions[:, :Hc]).

Usage:
  inner = SPEC_RealStackPointCloudDataset(...)
  ds = SPEC_EAG_Wrapper(inner, Hc=2)

Returned sample (dict) gets an extra key:
  - 'teacher_tail': (Hc, Da) float32

Author: PCDP
"""
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

# Import your original dataset
try:
    from pcdp.dataset.SPEC_stack_dataset import SPEC_RealStackPointCloudDataset
except Exception:
    SPEC_RealStackPointCloudDataset = None


class SPEC_EAG_Wrapper(Dataset):
    def __init__(self, inner_dataset: Dataset, Hc: int = 2):
        assert isinstance(inner_dataset, Dataset), "inner_dataset must be a torch.utils.data.Dataset"
        self.inner = inner_dataset
        self.Hc = int(Hc)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.inner[idx]
        try:
            actions = sample.get('action', None)
        except AttributeError:
            # If inner returns a tuple, convert to dict-like (not expected in your codebase)
            actions = None

        if actions is not None:
            # Expect (H, Da)
            arr = actions
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr)
            H = arr.shape[0]
            hc = min(self.Hc, max(0, H))
            if hc > 0:
                teacher_tail = arr[:hc, :].astype(np.float32)  # teacher forcing: first Hc steps
                sample['teacher_tail'] = teacher_tail
        return sample