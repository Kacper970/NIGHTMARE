import io
#import os
import sys
import json
import h5py
import torch
import random
# =============================================================================
# import numpy as np
# =============================================================================
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

UNDERSCORE_SORT = lambda s: list(map(try_int, s.split("_")))


def try_int(s):
    try:
        return int(s)
    except ValueError:
        return int.from_bytes(s.encode(), "little")


def generate_candidate_sampling_key(
    data_dir, split="train", source="features", sampling_level="panel_level", modality='text'
):
    data_source = "feat_data"
    if source == "images":
        data_source = "img_data"

    if (
        sampling_level == "intensity_level"
        or sampling_level == "ratio_level"
        or sampling_level == "saturation_level"  or sampling_level == "panel_level"
    ):
        metainfo = pd.read_csv(f"metainfo_{split}.csv")
        metainfo = metainfo.set_index("Unnamed: 0")
    elif sampling_level == "character_level":
        character_indexing = pd.read_csv(f"{data_dir}/characters_indexing.csv")

    with open(f"{split}_{source}_{modality}_indexing.json", "r") as f:
        indexing = json.load(f)[f"{split}_{source}_{modality}"]
    key_data = {}

    comics = indexing.keys()
    for comic in tqdm(comics):
        key_data[comic] = {}
        if data_source not in indexing[comic]:
            continue

        panels = indexing[comic][data_source]
        for i, panel in enumerate(panels):
            key_data[comic][panel] = []

            if sampling_level == "comic_level":
                key_data[comic][panel].append(f"{comic}/*")
            elif sampling_level == "panel_level":
                key_data[comic][panel].append(f"{comic}/{panel}")
            elif sampling_level == "ratio_level":
                ratio_bin = metainfo.loc[f"{comic}/{panel}"]["ratio_bin"]
                key_data[comic][panel] = list(
                    metainfo[metainfo["ratio_bin"] == ratio_bin].index
                )
            elif sampling_level == "intensity_level":
                intensity_bin = metainfo.loc[f"{comic}/{panel}"]["intensity_bin"]
                key_data[comic][panel] = list(
                    metainfo[metainfo["intensity_bin"] == intensity_bin].index
                )
            elif sampling_level == "saturation_level":
                saturation_bin = metainfo.loc[f"{comic}/{panel}"]["saturation_bin"]
                key_data[comic][panel] = list(
                    metainfo[metainfo["saturation_bin"] == saturation_bin].index
                )
            elif sampling_level == "character_level":
                character_panels = character_indexing[
                    character_indexing["panel_id"] == f"{comic}/{panel}"
                ]["panel_id"].tolist()

                if len(character_panels) == 0:
                    continue

                character_panels.remove(f"{comic}/{panel}")
                for character_panel in character_panels:
                    key_data[comic][panel].append(character_panel)
            else:
                raise ValueError("Invalid sampling key")

    with open(f"{split}_{source}_{modality}_{sampling_level}_sampling.json", "w+") as f:
        json.dump(key_data, f)
        
# =============================================================================
# generate_candidate_sampling_key("Data/comics/raw_panel_images/raw_panel_images", split="train", source="features", sampling_level="panel_level")
# generate_candidate_sampling_key("Data/comics/raw_panel_images/raw_panel_images", split="test", source="features", sampling_level="panel_level")
# generate_candidate_sampling_key("Data/comics/raw_panel_images/raw_panel_images", split="val", source="features", sampling_level="panel_level")
# =============================================================================

class CandidateSampler:
    def __init__(
        self,
        data_dir,
        split="train",
        modality='image_only',
        source="features",
        sampling_key="panel_level",
        sampling_strategy="random",
        negate_sampling=False,
        transform=None,
        num_candidates=9,
    ):
        self.split = split
        self.source = source
        self.data_source = "feat_data"
        if source == "images":
            self.data_source = "img_data"
        self.modality=modality
        self.transform = transform
        self.num_candidates = num_candidates
        self.panels_file = h5py.File(f"{data_dir}/{split}_{source}_{modality}.hdf5", "r")
        self.idx_to_panel = []

        self.sampling_strategy = sampling_strategy
        self.negate_sampling = negate_sampling

        self.return_metadata = False

        with open(
            f"{data_dir}/{split}_{source}_{modality}_{sampling_key}_sampling.json", "r"
        ) as f:
            self.key_data = json.load(f)

        with open(f"{data_dir}/{split}_{source}_{modality}_indexing.json", "r") as f:
            indexing = json.load(f)[f"{split}_{source}_{modality}"]

        self.panel_to_idx = {}
        self.idx_to_panel = []

        print("Indexing panels for sequence sampling...")
        comics = indexing.keys()
        for comic in tqdm(comics):
            if self.data_source not in indexing[comic]:
                continue

            self.panel_to_idx[comic] = {}
            panels = indexing[comic][self.data_source]

            for i, panel in enumerate(panels):
                self.panel_to_idx[comic][panel] = i
                self.idx_to_panel.append((comic, panel))

        if self.num_candidates < 0:
            self.all_candidates = []
            self.panel_to_candidate = {}

            print("Loading all candidates...")

            for comic, panel in tqdm(self.idx_to_panel):
                panel_idx = self.panel_to_idx[comic][panel]
                data = self.panels_file[f"{split}_{source}_{self.modality}/{comic}/{self.data_source}"][
                    panel_idx
                ]

                if self.source == "images":
                    data = Image.open(io.BytesIO(data))

                self.panel_to_candidate[f"{comic}/{panel}"] = len(self.all_candidates)
                self.all_candidates.append(self.transform(data))

            if len(self.all_candidates) > 10_000:
                self.all_candidates = self.all_candidates[:10_000]

    def random_partial_sample(self, idx):
        comic, panel = self.idx_to_panel[idx]
        non_candidate_data = self.key_data[comic][panel]

        sents = []
        candidates = []
        candidate_panels = []
        attemps = 0
        while len(candidates) < self.num_candidates:
            comic, panel = random.choice(self.idx_to_panel)

            condition = (
                f"{comic}/{panel}" in non_candidate_data
                or f"{comic}/*" in non_candidate_data
            )

            if self.negate_sampling and self.sampling_strategy == "random":
                condition = (
                    f"{comic}/{panel}" not in non_candidate_data
                    and f"{comic}/*" not in non_candidate_data
                )

            if condition:
                continue

            panel_idx = self.panel_to_idx[comic][panel]
            data = self.panels_file[
                f"{self.split}_{self.source}_{self.modality}/{comic}/{self.data_source}"
            ][panel_idx]
            sen = self.panels_file[
                f"{self.split}_{self.source}_{self.modality}/{comic}/sen_data"
            ][panel_idx]

            if self.source == "images":
                data = Image.open(io.BytesIO(data))
            candidates.append(self.transform(data))
            candidate_panels.append((comic, panel))
            sents.append(sen)

            attemps += 1

            if attemps > 5000:
                raise RuntimeError("Could not sample enough candidates")

        if self.return_metadata:
            return candidates, list(zip(*candidate_panels)), sents

        return candidates, sents

    def retrieve_partial_sample(self, idx):
        comic, panel = self.idx_to_panel[idx]
        non_candidate_data = self.key_data[comic][panel]

        can_filter = []
        if not self.negate_sampling:
            can_filter = [
                (c, p)
                for c, p in self.idx_to_panel
                if (
                    f"{c}/{p}" not in non_candidate_data
                    and f"{c}/*" not in non_candidate_data
                )
            ]
        else:
            can_filter = [
                (e.split("/")[0], e.split("/")[1]) for e in non_candidate_data
            ]

        # Expand {c}/* terms into all panels
        for i, (c, p) in enumerate(can_filter):
            if p == "*":
                can_filter[i] = [(c, p) for p in self.panel_to_idx[c].keys()]

        can_filter = [e if isinstance(e, list) else [e] for e in can_filter]
        can_filter = [e for s in can_filter for e in s]

        sents = []
        candidates = []
        candidate_panels = []

        for _ in range(min(self.num_candidates, len(can_filter))):
            comic, panel = random.choice(can_filter)

            can_filter.remove((comic, panel))
            panel_idx = self.panel_to_idx[comic][panel]

            data = self.panels_file[
                f"{self.split}_{self.source}_{self.modality}/{comic}/{self.data_source}"
            ][panel_idx]
            sen = self.panels_file[
                f"{self.split}_{self.source}_{self.modality}/{comic}/sen_data"
            ][panel_idx]

            if self.source == "images":
                data = Image.open(io.BytesIO(data))
            candidates.append(self.transform(data))
            candidate_panels.append((comic, panel))
            sents.append(sen)

        # Add 0-padding
        if len(candidates) < self.num_candidates:
            if len(candidates) == 0:
                candidates.append(torch.zeros(384))  # TODO: Fix hard-coded value
            padding = torch.zeros_like(candidates[0])
            num_padding = self.num_candidates - len(candidates)
            candidates += [padding] * num_padding
            sents += [torch.zeros_like(sents[0])] * num_padding


        if self.return_metadata:
            return candidates, list(zip(*candidate_panels)), sents

        return candidates, sents

    def full_sample(self, idx):
        candidates = self.all_candidates.copy()
        comic, panel = self.idx_to_panel[idx]
        pop_idx = self.panel_to_candidate.get(f"{comic}/{panel}", -1)

        if pop_idx >= 0 and pop_idx < len(candidates):
            del candidates[pop_idx]

        if self.return_metadata:
            candidate_panels = [
                (c, p) for c, p in self.idx_to_panel if (c, p) != (comic, panel)
            ]

            return candidates, list(zip(*candidate_panels))

        return candidates

    def sample(self, idx):
        if self.num_candidates > 0:
            if self.sampling_strategy == "random":
                return self.random_partial_sample(idx)
            elif self.sampling_strategy == "retrieve":
                return self.retrieve_partial_sample(idx)
            elif self.sampling_strategy == "mix":
                if self.return_metadata:
                    candidates1, panels1, sents1 = self.retrieve_partial_sample(idx)
                    candidates2, panels2, sents2 = self.random_partial_sample(idx)

                    return candidates1 + candidates2, panels1 + panels2, sents1 + sents2
                else:
                    candidates1, sents1 = self.retrieve_partial_sample(idx)
                    candidates2, sents2 = self.random_partial_sample(idx)
                    
                    return candidates1 + candidates2, sents1 + sents2
        else:
            return self.full_sample(idx)

    def close(self):
        self.panels_file.close()


class SequenceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        modality='text',
        source="images",
        transform=None,
        pre_seq_len=2,
        suf_seq_len=2,
    ):
        super().__init__()

        self.split = split
        self.modality = modality
        self.source = source
        self.data_source = "feat_data"
        if source == "images":
            self.data_source = "img_data"

        self.transform = transform
        self.pre_seq_len = pre_seq_len
        self.suf_seq_len = suf_seq_len
        self.tot_seq_len = pre_seq_len + 1 + suf_seq_len

        self.panels_file = h5py.File(f"{data_dir}/{split}_{source}_{modality}.hdf5", "r")
        
# =============================================================================
#         if self.modality == 'text':
#             self.sents = h5py.File(f"{data_dir}/{split}_{source}_{modality}.hdf5", "r")
# =============================================================================

        with open(f"{data_dir}/{split}_{source}_{modality}_indexing.json", "r") as f:
           # print(json.load(f).keys())
            indexing = json.load(f)[f"{split}_{source}_{modality}"]
            

        self.panel_to_idx = {}
        self.idx_to_panel = []
        self.idx_to_start_panel = []

        print("Indexing panels for sequence sampling...")
        comics = indexing.keys()
        for comic in tqdm(comics):
            if self.data_source not in indexing[comic]:
                continue

            self.panel_to_idx[comic] = {}
            panels = indexing[comic][self.data_source]

            for i, panel in enumerate(panels):
                if (
                    i >= pre_seq_len and i < len(panels) - suf_seq_len
                ):
                    self.idx_to_start_panel.append(len(self.idx_to_panel))
                self.idx_to_panel.append((comic, panel))
                self.panel_to_idx[comic][panel] = i

    def __len__(self):
        return len(self.idx_to_start_panel)

    def _fetch_panel(self, idx):
        comic, panel = self.idx_to_panel[idx]
        panel_idx = self.panel_to_idx[comic][panel]
        
        if self.modality == 'text':
            
            data = self.panels_file[f"{self.split}_{self.source}_{self.modality}/{comic}/{self.data_source}"][panel_idx]
            
            sen = self.panels_file[f"{self.split}_{self.source}_{self.modality}/{comic}/sen_data"][panel_idx]
            
            if self.source == "images":
                data = Image.open(io.BytesIO(data))
    
            if self.transform:
                data = self.transform(data)
            
            return data, sen
            
            
        else:
    
            data = self.panels_file[f"{self.split}_{self.source}/{comic}/{self.data_source}"][panel_idx]
            
            if self.source == "images":
                data = Image.open(io.BytesIO(data))
    
            if self.transform:
                data = self.transform(data)
            
            return data

    def __getitem__(self, idx):
        seq1, seq2, seq3 = [], [], []
        sen1, sen2, sen3 = [], [], []
        start_idx = self.idx_to_start_panel[idx]
        
        if self.modality == 'text':
            
            for i in range(-self.pre_seq_len, self.suf_seq_len + 1):
                
                seq_1, sen_1 = self._fetch_panel(start_idx - self.tot_seq_len + i)
                seq1.append(seq_1)
                sen1.append(sen_1)
    
            for i in range(-self.pre_seq_len, self.suf_seq_len + 1):
                
                seq_2, sen_2 = self._fetch_panel(start_idx - self.tot_seq_len + i)
                seq2.append(seq_2)
                sen2.append(sen_2)
    
            for i in range(-self.pre_seq_len, self.suf_seq_len + 1):
                
                seq_3, sen_3 = self._fetch_panel(start_idx - self.tot_seq_len + i)
                seq3.append(seq_3)
                sen3.append(sen_3)
                
                
    
            return torch.stack(seq1), sen1, torch.stack(seq2), sen2,  torch.stack(seq3), sen3
            
        else:

            for i in range(-self.pre_seq_len, self.suf_seq_len + 1):
                seq1.append(self._fetch_panel(start_idx - self.tot_seq_len + i))
    
            for i in range(-self.pre_seq_len, self.suf_seq_len + 1):
                seq2.append(self._fetch_panel(start_idx - self.tot_seq_len + i))
    
            for i in range(-self.pre_seq_len, self.suf_seq_len + 1):
                seq3.append(self._fetch_panel(start_idx - self.tot_seq_len + i))
    
            return torch.stack(seq1), torch.stack(seq2), torch.stack(seq3)

    def close(self):
        self.panels_file.close()


class SequenceCandidateDataset(SequenceDataset):
    def __init__(
        self,
        data_dir,
        candidate_sampler,
        split="train",
        modality='text',
        source="images",
        transform=None,
        pre_seq_len=2,
        suf_seq_len=2,
        return_metadata=False,
    ):
        super().__init__(
            data_dir,
            split=split,
            modality=modality,
            source=source,
            transform=transform,
            pre_seq_len=pre_seq_len,
            suf_seq_len=suf_seq_len,
        )

        self.candidate_sampler = candidate_sampler
        self.return_metadata = return_metadata
        self.candidate_sampler.return_metadata = return_metadata

    def __getitem__(self, idx):
        
        if self.modality == 'text':
            seq1, sen1, seq2, sen2, seq3, sen3 = super().__getitem__(idx)
            
        else: 
            seq1, seq2, seq3 = super().__getitem__(idx)

        start_idx = self.idx_to_start_panel[idx]

        if self.return_metadata:
            candidates, candidate_panels, sents = self.candidate_sampler.sample(start_idx)
        else:
            candidates, sents = self.candidate_sampler.sample(start_idx)
            #candidates = (x, y)
        candidates.append(seq2[self.pre_seq_len])
        sents.append(sen2[self.pre_seq_len])

        candidates = torch.stack(candidates)
        true_candidate_index = len(candidates) - 1
        
        if self.modality == 'text':
            
            if self.return_metadata:
                return (
                    seq1, sen1,
                    seq2, sen2,
                    seq3, sen3,
                    candidates,
                    true_candidate_index,
                    self.idx_to_panel[idx],
                    candidate_panels, sents
                )

            return seq1, sen1, seq2, sen2, seq3, sen3, candidates, true_candidate_index, sents
        
        else:
            
                
    
            if self.return_metadata:
                return (
                    seq1,
                    seq2,
                    seq3,
                    candidates,
                    true_candidate_index,
                    self.idx_to_panel[idx],
                    candidate_panels,
                )
    
            return seq1, seq2, seq3, candidates, true_candidate_index

    def close(self):
        super().close()
        self.candidate_sampler.close()


class SequenceClassifcationDataset(SequenceDataset):
    def __init__(
        self,
        data_dir,
        split="train",
        modality='text',
        source="images",
        transform=None,
        pre_seq_len=2,
        suf_seq_len=2,
        return_metadata=False,
    ):
        super().__init__(
            data_dir,
            split=split,
            modality=modality,
            source=source,
            transform=transform,
            pre_seq_len=pre_seq_len,
            suf_seq_len=suf_seq_len,
        )

# =============================================================================
#         with open(f"{data_dir}/classification.json", "r") as f:
#             self.classification = json.load(f)
# =============================================================================

# =============================================================================
#         self.num_classes = max(self.classification["comic_to_cls"].values())
# =============================================================================
        self.return_metadata = return_metadata

    def __getitem__(self, idx):
        
        if self.modality == 'text':
            seq, sen, _, _, _, _ = super().__getitem__(idx)
            
        else: 
            _, seq, _ = super().__getitem__(idx)
        
        
        #seq = super().__getitem__(idx)[1]
        start_idx = self.idx_to_start_panel[idx]

        comic, panel = self.idx_to_panel[start_idx + self.pre_seq_len]
        #cls_id = self.classification["comic_to_cls"].get(comic, self.num_classes)

        if self.return_metadata:
            #return seq, cls_id, comic, panel
            return seq, 4, comic, panel, sen
        #return seq, cls_id
        return seq

if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
