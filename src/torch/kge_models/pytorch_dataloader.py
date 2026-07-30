#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.py.load.kg import parse_triples
from src.py.util.util import to_tensor_cpu

# === 全局累加器：4 阶段 Micro-benchmarking ===
global_phase_1_time = 0.0
global_phase_3_time = 0.0

def get_global_phase_times():
    return global_phase_1_time, global_phase_3_time

def reset_global_phase_times():
    global global_phase_1_time, global_phase_3_time
    global_phase_1_time = 0.0
    global_phase_3_time = 0.0

# === Per-Batch Profiling Globals ===
global_collate_time_ms = 0.0
global_neg_sampling_time_ms = 0.0
global_tensor_build_time_ms = 0.0
global_neg_sampling_time_b1_ms = 0.0
global_neg_sampling_time_b2_ms = 0.0
global_neg_sampling_time_b3_ms = 0.0
global_neg_sampling_time_b4_ms = 0.0
global_neg_sampling_time_b5_ms = 0.0

global_retry_counts = []

GLOBAL_ENTITY_DEGREE = {}
HUB_DEGREE_THRESHOLD = 0
HUB_TOP1_PCT_THRESHOLD = 0

def init_entity_degree(kgs, hub_percentile=10, hub_top1_pct=1):
    global GLOBAL_ENTITY_DEGREE, HUB_DEGREE_THRESHOLD, HUB_TOP1_PCT_THRESHOLD
    from collections import Counter
    counter = Counter()
    for h, r, t in kgs.relation_triples_list:
        counter[h] += 1
        counter[t] += 1
    GLOBAL_ENTITY_DEGREE = dict(counter)
    if len(counter) > 0:
        sorted_degrees = sorted(counter.values(), reverse=True)
        idx10 = max(1, len(sorted_degrees) * hub_percentile // 100)
        HUB_DEGREE_THRESHOLD = sorted_degrees[idx10 - 1]
        idx1 = max(1, len(sorted_degrees) * hub_top1_pct // 100)
        HUB_TOP1_PCT_THRESHOLD = sorted_degrees[idx1 - 1]
    else:
        HUB_DEGREE_THRESHOLD = 0
        HUB_TOP1_PCT_THRESHOLD = 0

def reset_per_batch_profiling():
    global global_collate_time_ms, global_neg_sampling_time_ms, global_tensor_build_time_ms
    global global_neg_sampling_time_b1_ms, global_neg_sampling_time_b2_ms
    global global_neg_sampling_time_b3_ms, global_neg_sampling_time_b4_ms
    global global_neg_sampling_time_b5_ms
    global global_retry_counts
    global_collate_time_ms = 0.0
    global_neg_sampling_time_ms = 0.0
    global_tensor_build_time_ms = 0.0
    global_neg_sampling_time_b1_ms = 0.0
    global_neg_sampling_time_b2_ms = 0.0
    global_neg_sampling_time_b3_ms = 0.0
    global_neg_sampling_time_b4_ms = 0.0
    global_neg_sampling_time_b5_ms = 0.0
    global_retry_counts = []

def get_per_batch_profiling():
    return {
        'collate_time_ms': global_collate_time_ms,
        'neg_sampling_time_ms': global_neg_sampling_time_ms,
        'tensor_build_time_ms': global_tensor_build_time_ms,
        'neg_sampling_b1_ms': global_neg_sampling_time_b1_ms,
        'neg_sampling_b2_ms': global_neg_sampling_time_b2_ms,
        'neg_sampling_b3_ms': global_neg_sampling_time_b3_ms,
        'neg_sampling_b4_ms': global_neg_sampling_time_b4_ms,
        'neg_sampling_b5_ms': global_neg_sampling_time_b5_ms,
        'retry_counts': list(global_retry_counts),
    }


class PyTorchTrainDataset(Dataset):

    def __init__(self, triples, neg_num, kgs):
        self.head = [x[0] for x in triples]
        self.tail = [x[2] for x in triples]
        self.rel = [x[1] for x in triples]
        self.neg_num = neg_num
        self.kgs = kgs

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx]

    def collate_fn(self, data):
        global global_phase_1_time, global_phase_3_time
        global global_collate_time_ms, global_neg_sampling_time_ms, global_tensor_build_time_ms
        global global_retry_counts
        global global_neg_sampling_time_b1_ms, global_neg_sampling_time_b2_ms
        global global_neg_sampling_time_b3_ms, global_neg_sampling_time_b4_ms
        global global_neg_sampling_time_b5_ms

        # === Reset per-batch profiling accumulators at start of each collate ===
        # This ensures each batch gets its own timing data
        global_collate_time_ms = 0.0
        global_neg_sampling_time_ms = 0.0
        global_tensor_build_time_ms = 0.0
        global_neg_sampling_time_b1_ms = 0.0
        global_neg_sampling_time_b2_ms = 0.0
        global_neg_sampling_time_b3_ms = 0.0
        global_neg_sampling_time_b4_ms = 0.0
        global_neg_sampling_time_b5_ms = 0.0
        global_retry_counts = []

        collate_start = time.time()

        batch_h_list = [item[0] for item in data]
        batch_r_list = [item[1] for item in data]
        batch_t_list = [item[2] for item in data]

        # Stage B: Negative Sampling (calls deep-profiled version)
        neg_sampling_start = time.time()
        batch_neg, retry_info = self.generate_neg_triples_fast(
            data, set(self.kgs.relation_triples_list),
            self.kgs.entities_list, self.neg_num
        )
        neg_sampling_end = time.time()
        neg_sampling_ms = (neg_sampling_end - neg_sampling_start) * 1000.0
        global_neg_sampling_time_ms += neg_sampling_ms
        global_retry_counts.extend(retry_info)

        global_phase_3_time += (neg_sampling_end - neg_sampling_start)

        # Stage C: Tensor Construction
        tensor_build_start = time.time()
        batch_h = to_tensor_cpu(batch_h_list + [x[0] for x in batch_neg])
        batch_r = to_tensor_cpu(batch_r_list + [x[1] for x in batch_neg])
        batch_t = to_tensor_cpu(batch_t_list + [x[2] for x in batch_neg])
        tensor_build_end = time.time()
        tensor_build_ms = (tensor_build_end - tensor_build_start) * 1000.0
        global_tensor_build_time_ms += tensor_build_ms

        global_phase_1_time += (tensor_build_end - tensor_build_start)

        batch_data = [batch_h, batch_r, batch_t]
        batch_data = torch.stack(batch_data)

        global_collate_time_ms += (time.time() - collate_start) * 1000.0

        return batch_data

    # generate_neg_triples_fast will be OVERRIDDEN below with the deep-profiled version.
    # This placeholder is required for Python syntax, replaced at module level.
    def generate_neg_triples_fast(self, pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
        raise NotImplementedError("This will be replaced by the deep-profiled version")

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def set_ent_neg_rate(self, rate):
        self.neg_ent = rate

    def set_rel_neg_rate(self, rate):
        self.neg_rel = rate

    def set_bern_flag(self, bern_flag):
        self.bern_flag = bern_flag

    def set_filter_flag(self, filter_flag):
        self.filter_flag = filter_flag

    def get_ent_tot(self):
        return self.ent_total

    def get_rel_tot(self):
        return self.rel_total

    def get_tri_tot(self):
        return self.tri_total


# === Deep-profiled generate_neg_triples_fast: replaces the class method ===
def _deep_profiled_neg_sampling(self, pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
    """
    Deep-profiled version of generate_neg_triples_fast.
    Splits each inner operation into B1-B5 sub-stages.
    Preserves ALL original training logic exactly.
    """
    global global_neg_sampling_time_b1_ms, global_neg_sampling_time_b2_ms
    global global_neg_sampling_time_b3_ms, global_neg_sampling_time_b4_ms
    global global_neg_sampling_time_b5_ms

    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    retry_counts = []

    for head, relation, tail in pos_batch:
        neg_triples = list()
        nums_to_sample = neg_triples_num
        head_candidates = neighbor.get(head, entities_list)
        tail_candidates = neighbor.get(tail, entities_list)
        retry_this = 0
        for i in range(max_try):
            retry_this += 1

            # B1: Random Sampling (random.sample) — 只在首次尝试计时，排除重试
            if i == 0:
                t_b1 = time.perf_counter()
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
            if i == 0:
                global_neg_sampling_time_b1_ms += (time.perf_counter() - t_b1) * 1000.0

            # B2: Candidate Construction (set comprehension)
            t_b2 = time.perf_counter()
            if corrupt_head_prob:
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            global_neg_sampling_time_b2_ms += (time.perf_counter() - t_b2) * 1000.0

            if i == max_try - 1:
                # B5: list conversion from set
                t_b5 = time.perf_counter()
                neg_triples += list(i_neg_triples)
                global_neg_sampling_time_b5_ms += (time.perf_counter() - t_b5) * 1000.0
                break
            else:
                # B3: Collision Check (set difference)
                t_b3 = time.perf_counter()
                filtered = list(i_neg_triples - all_triples_set)
                global_neg_sampling_time_b3_ms += (time.perf_counter() - t_b3) * 1000.0

                # B5: extend with filtered
                t_b5 = time.perf_counter()
                neg_triples += filtered
                global_neg_sampling_time_b5_ms += (time.perf_counter() - t_b5) * 1000.0

            # B4: Retry Processing
            t_b4 = time.perf_counter()
            if len(neg_triples) == neg_triples_num:
                global_neg_sampling_time_b4_ms += (time.perf_counter() - t_b4) * 1000.0
                break
            else:
                nums_to_sample = neg_triples_num - len(neg_triples)
                global_neg_sampling_time_b4_ms += (time.perf_counter() - t_b4) * 1000.0

        retry_counts.append(retry_this)

        # B5: extend into global neg_batch
        t_b5 = time.perf_counter()
        neg_batch.extend(neg_triples)
        global_neg_sampling_time_b5_ms += (time.perf_counter() - t_b5) * 1000.0

    return neg_batch, retry_counts


# Install the deep-profiled version (replaces placeholder)
PyTorchTrainDataset.generate_neg_triples_fast = _deep_profiled_neg_sampling


class PyTorchTrainDataLoader(DataLoader):
    def __init__(self, kgs, batch_size, threads, neg_size):
        self.batch_size = batch_size
        self.kgs = kgs
        self.neg_size = neg_size
        self.data = self.__construct_dataset()
        super().__init__(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=threads,
            pin_memory=True,
            collate_fn=self.data.collate_fn,
            drop_last=False
        )

    def __construct_dataset(self):
        triples_set = self.kgs.relation_triples_set
        return PyTorchTrainDataset(list(triples_set), self.kgs.entities_num,
                                   self.kgs.relations_num, neg_ent=self.neg_size)

    def get_ent_tot(self):
        return self.data.get_ent_tot()

    def get_rel_tot(self):
        return self.data.get_rel_tot()

    def get_batch_size(self):
        return self.batch_size

    def set_sampling_mode(self, sampling_mode):
        self.dataset.set_sampling_mode(sampling_mode)

    def set_work_threads(self, work_threads):
        self.num_workers = work_threads

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches
        self.batch_size = self.tripleTotal // self.nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.dataset.set_ent_neg_rate(rate)

    def set_rel_neg_rate(self, rate):
        self.dataset.set_rel_neg_rate(rate)

    def set_bern_flag(self, bern_flag):
        self.dataset.set_bern_flag(bern_flag)

    def set_filter_flag(self, filter_flag):
        self.dataset.set_filter_flag(filter_flag)

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.dataset.get_ent_tot()

    def get_rel_tot(self):
        return self.dataset.get_rel_tot()

    def get_triple_tot(self):
        return self.dataset.get_tri_tot()


def parse_triples_list(relation_set):
    subjects, predicates, objects = list(), list(), list()
    for o, p, s in relation_set:
        objects.append(o)
        predicates.append(p)
        subjects.append(s)
    return objects, predicates, objects