import math
import time
import os
import tqdm
import csv
from collections import Counter
from torch.profiler import profile, record_function, ProfilerActivity
from src.torch.kge_models.pytorch_dataloader import PyTorchTrainDataset
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.py.base.losses import get_loss_func_torch
from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.evaluation import LinkPredictionEvaluator
from src.py.load import batch
from src.py.util.util import task_divide, early_stop, to_var, to_tensor
import ray
from typing import Dict

from src.torch.kge_models.basic_model import parallel_model

from src.torch.kge_models.pytorch_dataloader import (
    init_entity_degree, GLOBAL_ENTITY_DEGREE, HUB_DEGREE_THRESHOLD, HUB_TOP1_PCT_THRESHOLD,
    reset_per_batch_profiling, get_per_batch_profiling,
    get_global_phase_times, reset_global_phase_times,
)


class kge_trainer:
    def __init__(self):
        self.device = None
        self.valid = None
        self.batch_size = None
        self.neg_catch = None
        self.loss = None
        self.data_loader = None
        self.optimizer = None
        self.model = None
        self.kgs = None
        self.args = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        
        # === Profiling accumulators ===
        self.profiling_rows = []       # list of dicts for profiling_summary.csv
        self.hub_rows = []             # list of dicts for hub_analysis.csv
        self.neg_sampling_cost_rows = []  # list of dicts for negative_sampling_cost.csv (Phase 2)
        self.global_step = 0

    def init(self, args, kgs, model):
        self.args = args
        self.kgs = kgs
        self.model = model
        if self.args.is_gpu:
            # torch.cuda.set_device(0)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.device = torch.device('cuda:2')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # === Precompute entity degree for Hub Analysis ===
        init_entity_degree(kgs, hub_percentile=10)

        self.valid = LinkPredictionEvaluator(model, args, kgs, is_valid=True)
        self.optimizer = get_optimizer_torch(self.args.optimizer, self.model, self.args.learning_rate)
        train_dataset = PyTorchTrainDataset(self.kgs.relation_triples_list, self.args.neg_triple_num, kgs)
        self.data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                      collate_fn=train_dataset.collate_fn,
                                      shuffle=True, pin_memory=True, num_workers=self.args.batch_threads_num,
                                      drop_last=False)

    def run_t(self):
        triples_num = self.kgs.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        start = time.time()
        print(next(self.model.parameters()).device)
        for i in range(self.args.max_epoch):
            res = 0
            tm = time.time()
            for steps_task in steps_tasks:
                mp.Process(target=batch.generate_relation_triple_batch_queue,
                           args=(self.kgs.relation_triples_list, [],
                                 self.kgs.relation_triples_set, set(),
                                 self.kgs.entities_list, [],
                                 self.args.batch_size, steps_task,
                                 training_batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
            # print('processing cost time: {:.4f}s'.format(time.time() - tm))

            start = time.time()
            length = 0
            for j in range(triple_steps):
                self.optimizer.zero_grad()
                batch_pos, batch_neg = training_batch_queue.get()
                self.batch_size = len(batch_pos)
                # print(len(batch_neg))
                # length += len(batch_pos)
                batch_pos = np.array(batch_pos)
                batch_neg = np.array(batch_neg)
                datas = np.concatenate((batch_pos, batch_neg), axis=0)
                # datas = batch_pos
                data = {
                    'batch_h': to_var(np.array([x[0] for x in datas]), self.device),
                    'batch_r': to_var(np.array([x[1] for x in datas]), self.device),
                    'batch_t': to_var(np.array([x[2] for x in datas]), self.device),
                }
                score = self.model(data)

                length += self.batch_size
                po_score = self.get_pos_score(score)
                ne_score = self.get_neg_score(score)
                loss = get_loss_func_torch(po_score, ne_score, self.args)
                loss.backward()
                self.optimizer.step()
                res += loss.item()
                """
                score.backward()
                self.optimizer.step()
                length = length + 1
                res += score.item()
                """
            print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                t1 = time.time()
                flag = self.valid.print_results()
                print('valid cost time: {:.4f}s'.format(time.time() - start))
                '''self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break'''
        self.save()

    def run(self):
        from src.torch.kge_models.pytorch_dataloader import get_global_phase_times, reset_global_phase_times
        
        print(next(self.model.parameters()).device)
        
        # === Output dir ===
        out_dir = self.args.output if hasattr(self.args, 'output') else 'output/results/'
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(self.args.max_epoch):
            # Clear GPU cache at start of each epoch to prevent OOM accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res = 0
            length = 0

            # === Epoch 级别初始化：清空全局累加器 ===
            reset_global_phase_times()

            # GPU 阶段累加器 (秒)
            acc_phase_2 = 0.0  # Embedding Lookup
            acc_phase_4 = 0.0  # Geometry & Learning

            epoch_start_time = time.time()
            data_iter = iter(self.data_loader)

            for step_idx, batch in enumerate(data_iter):
                self.global_step += 1

                self.optimizer.zero_grad()
                batch_size_pos = int(batch[0].shape[0] / (self.args.neg_triple_num + 1))
                self.batch_size = batch_size_pos
                
                data = {
                    'batch_h': batch[0].to(self.device),
                    'batch_r': batch[1].to(self.device),
                    'batch_t': batch[2].to(self.device)
                }

                # === Task 3: Hub Entity Analysis (positive triples only) ===
                # batch[0] shape: [total] where first batch_size_pos are positive heads
                pos_heads_cpu = batch[0][:batch_size_pos].cpu().numpy() if batch[0].is_cuda else batch[0][:batch_size_pos].numpy()
                pos_tails_cpu = batch[2][:batch_size_pos].cpu().numpy() if batch[2].is_cuda else batch[2][:batch_size_pos].numpy()
                
                degrees = []
                for e in pos_heads_cpu:
                    degrees.append(GLOBAL_ENTITY_DEGREE.get(int(e), 0))
                for e in pos_tails_cpu:
                    degrees.append(GLOBAL_ENTITY_DEGREE.get(int(e), 0))
                
                batch_avg_degree = float(np.mean(degrees)) if degrees else 0.0
                batch_max_degree = float(np.max(degrees)) if degrees else 0.0
                hub_count = sum(1 for d in degrees if d >= HUB_DEGREE_THRESHOLD)

                # === Stage D: Forward (Phase 2 original) ===
                torch.cuda.synchronize()
                fwd_start = time.time()
                score = self.model(data)
                torch.cuda.synchronize()
                forward_time_ms = (time.time() - fwd_start) * 1000.0
                acc_phase_2 += forward_time_ms / 1000.0

                # === Loss + Backward (Stage E) ===
                torch.cuda.synchronize()
                bwd_start = time.time()
                if self.model.__class__.__name__ == 'ConvE' or self.model.__class__.__name__ == 'TuckER':
                    loss = score
                    loss.backward()
                    res += score.item()
                    length += 1
                else:
                    length += self.batch_size
                    po_score = self.get_pos_score(score)
                    ne_score = self.get_neg_score(score)
                    loss = get_loss_func_torch(po_score, ne_score, self.args)
                    loss.backward()
                    res += loss.item()
                torch.cuda.synchronize()
                backward_time_ms = (time.time() - bwd_start) * 1000.0

                # === Stage F: Optimizer ===
                torch.cuda.synchronize()
                opt_start = time.time()
                self.optimizer.step()
                torch.cuda.synchronize()
                optimizer_time_ms = (time.time() - opt_start) * 1000.0

                # === Phase 4 (original) ===
                acc_phase_4 += (backward_time_ms + optimizer_time_ms) / 1000.0

                # === Task 1: Collect per-batch profiling data (Phase 1 + Phase 2 deep) ===
                prof_data = get_per_batch_profiling()
                collate_time_ms = prof_data['collate_time_ms']
                neg_sampling_time_ms = prof_data['neg_sampling_time_ms']
                tensor_time_ms = prof_data['tensor_build_time_ms']
                retry_counts = prof_data['retry_counts']
                avg_retry = float(np.mean(retry_counts)) if retry_counts else 0.0
                max_retry = float(np.max(retry_counts)) if retry_counts else 0.0
                total_retry = int(np.sum(retry_counts)) if retry_counts else 0

                # === Phase 2 Deep: B1-B5 sub-stage times ===
                b1_ms = prof_data['neg_sampling_b1_ms']
                b2_ms = prof_data['neg_sampling_b2_ms']
                b3_ms = prof_data['neg_sampling_b3_ms']
                b4_ms = prof_data['neg_sampling_b4_ms']
                b5_ms = prof_data['neg_sampling_b5_ms']

                # === Task 2: GPU Resource Monitoring ===
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)  # MB
                    gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)  # MB
                else:
                    gpu_mem_allocated = 0
                    gpu_mem_reserved = 0

                # === Step time ===
                step_time_ms = collate_time_ms + neg_sampling_time_ms + tensor_time_ms + forward_time_ms + backward_time_ms + optimizer_time_ms

                # === Phase 2: Batch-level Graph Structure Analysis (Task 2) ===
                # Split head/tail degree
                head_degrees = [GLOBAL_ENTITY_DEGREE.get(int(e), 0) for e in pos_heads_cpu]
                tail_degrees = [GLOBAL_ENTITY_DEGREE.get(int(e), 0) for e in pos_tails_cpu]
                avg_head_deg = float(np.mean(head_degrees)) if head_degrees else 0.0
                avg_tail_deg = float(np.mean(tail_degrees)) if tail_degrees else 0.0

                # Top 1% hub entities
                hub_top1_count = sum(1 for d in degrees if d >= HUB_TOP1_PCT_THRESHOLD)

                # Unique entities and relations in positive batch
                pos_head_set = set(int(e) for e in pos_heads_cpu)
                pos_tail_set = set(int(e) for e in pos_tails_cpu)
                unique_entities = len(pos_head_set | pos_tail_set)
                # Relations: batch_r from positive triples (first batch_size_pos entries)
                pos_rels_cpu = batch[1][:batch_size_pos].cpu().numpy() if batch[1].is_cuda else batch[1][:batch_size_pos].numpy()
                unique_relations = len(set(int(e) for e in pos_rels_cpu))

                # === Write to profiling_summary.csv (Phase 1) ===
                self.profiling_rows.append({
                    'epoch': i,
                    'step': self.global_step,
                    'collate_time': round(collate_time_ms, 3),
                    'neg_sampling_time': round(neg_sampling_time_ms, 3),
                    'tensor_time': round(tensor_time_ms, 3),
                    'forward_time': round(forward_time_ms, 3),
                    'backward_time': round(backward_time_ms, 3),
                    'optimizer_time': round(optimizer_time_ms, 3),
                    'step_time': round(step_time_ms, 3),
                    'gpu_memory_allocated': round(gpu_mem_allocated, 1),
                    'gpu_memory_reserved': round(gpu_mem_reserved, 1),
                    'hub_count': hub_count,
                    'avg_degree': round(batch_avg_degree, 2),
                    'avg_retry': round(avg_retry, 4),
                    'max_retry': round(max_retry, 2),
                })

                # === Write to hub_analysis.csv (Phase 1) ===
                self.hub_rows.append({
                    'batch_id': self.global_step,
                    'avg_degree': round(batch_avg_degree, 2),
                    'max_degree': round(batch_max_degree, 2),
                    'hub_entity_count': hub_count,
                    'neg_sampling_time': round(neg_sampling_time_ms, 3),
                    'avg_retry': round(avg_retry, 4),
                    'max_retry': round(max_retry, 2),
                })

                # === Write to negative_sampling_cost.csv (Phase 2 deep) ===
                self.neg_sampling_cost_rows.append({
                    'epoch': i,
                    'step': self.global_step,
                    'batch_size': batch_size_pos,
                    'neg_num': self.args.neg_triple_num,
                    'sampling_time': round(b1_ms, 3),
                    'candidate_build_time': round(b2_ms, 3),
                    'collision_check_time': round(b3_ms, 3),
                    'retry_time': round(b4_ms, 3),
                    'output_build_time': round(b5_ms, 3),
                    'total_neg_sampling_time': round(neg_sampling_time_ms, 3),
                    'avg_head_degree': round(avg_head_deg, 2),
                    'avg_tail_degree': round(avg_tail_deg, 2),
                    'avg_entity_degree': round(batch_avg_degree, 2),
                    'max_entity_degree': round(batch_max_degree, 2),
                    'hub_entity_count': hub_count,
                    'hub_top1_pct_count': hub_top1_count,
                    'unique_entities': unique_entities,
                    'unique_relations': unique_relations,
                    'avg_retry': round(avg_retry, 4),
                    'max_retry': round(max_retry, 2),
                    'total_retry': total_retry,
                })

            # === 收集 CPU 阶段时间 ===
            phase_1_time, phase_3_time = get_global_phase_times()

            # === Epoch 结束统计与打印: 终极 4 阶段消耗权威报告 ===
            epoch_end_time = time.time()
            total_epoch_time = epoch_end_time - epoch_start_time

            # 其他框架调度时间 = 总时间 - (P1 + P2 + P3 + P4)
            other_time = total_epoch_time - (phase_1_time + acc_phase_2 + phase_3_time + acc_phase_4)

            print()
            print("=== 终极 4 阶段耗时权威报告 (Epoch {}) ===".format(i))
            print("运行环境: 单进程串行 (batch_threads_num=0)")
            print()
            print("[CPU 阶段]")
            print("第1段階 (ID Mapping):       {:.4f} 秒".format(phase_1_time))
            print("第3段階 (Negative Sampling):  {:.4f} 秒".format(phase_3_time))
            print()
            print("[GPU 阶段]")
            print("第2段階 (Embedding Lookup):   {:.4f} 秒".format(acc_phase_2))
            print("第4段階 (Geometry & Learning): {:.4f} 秒".format(acc_phase_4))
            print()
            print("其他框架调度时间:             {:.4f} 秒".format(other_time))
            print("Epoch 总挂钟时间:             {:.4f} 秒".format(total_epoch_time))
            print("=========================================")
            print('epoch {}, avg. triple loss: {:.4f}'.format(i, res / length))

            # === CRITICAL: Free memory before validation to prevent OOM ===
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                t1 = time.time()
                flag = self.valid.print_results()
                print('valid cost time: {:.4f}s'.format(time.time() - t1))
                # Clear GPU cache after validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # TODO: Add early stop for KGE here.
            
            # === Dump CSVs incrementally every epoch to prevent data loss ===
            self._write_profiling_csvs(out_dir)
        
        # === Write CSVs at end of training (final) ===
        self._write_profiling_csvs(out_dir)
        
        self.test()
        self.save()
    
    def _write_profiling_csvs(self, out_dir):
        """Write profiling_summary.csv, hub_analysis.csv, negative_sampling_cost.csv, and negative_sampling_breakdown.csv."""
        # profiling_summary.csv
        if self.profiling_rows:
            fields = [
                'epoch', 'step', 'collate_time', 'neg_sampling_time', 'tensor_time',
                'forward_time', 'backward_time', 'optimizer_time', 'step_time',
                'gpu_memory_allocated', 'gpu_memory_reserved',
                'hub_count', 'avg_degree', 'avg_retry', 'max_retry',
            ]
            path = os.path.join(out_dir, 'profiling_summary.md')
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.profiling_rows)
            print("[Profiling] Saved profiling_summary.csv with {} rows".format(len(self.profiling_rows)))
        
        # hub_analysis.csv
        if self.hub_rows:
            fields_hub = [
                'batch_id', 'avg_degree', 'max_degree', 'hub_entity_count',
                'neg_sampling_time', 'avg_retry', 'max_retry',
            ]
            path = os.path.join(out_dir, 'hub_analysis.md')
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields_hub)
                writer.writeheader()
                writer.writerows(self.hub_rows)
            print("[Profiling] Saved hub_analysis.csv with {} rows".format(len(self.hub_rows)))
        
        # === Generate training_time_breakdown.csv ===
        if self.profiling_rows:
            total_collate = sum(r['collate_time'] for r in self.profiling_rows)
            total_neg = sum(r['neg_sampling_time'] for r in self.profiling_rows)
            total_tensor = sum(r['tensor_time'] for r in self.profiling_rows)
            total_fwd = sum(r['forward_time'] for r in self.profiling_rows)
            total_bwd = sum(r['backward_time'] for r in self.profiling_rows)
            total_opt = sum(r['optimizer_time'] for r in self.profiling_rows)
            total_all = total_collate + total_neg + total_tensor + total_fwd + total_bwd + total_opt
            
            breakdown_fields = ['stage', 'time_ms', 'pct']
            path = os.path.join(out_dir, 'training_time_breakdown.md')
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=breakdown_fields)
                writer.writeheader()
                for label, val in [
                    ('Collate', total_collate),
                    ('Negative Sampling', total_neg),
                    ('Tensor Construction', total_tensor),
                    ('Forward', total_fwd),
                    ('Backward', total_bwd),
                    ('Optimizer', total_opt),
                ]:
                    pct = (val / total_all * 100) if total_all > 0 else 0.0
                    writer.writerow({'stage': label, 'time_ms': round(val, 3), 'pct': round(pct, 2)})
            print("[Profiling] Saved training_time_breakdown.csv")
        
        # === Phase 2: negative_sampling_cost.csv ===
        if self.neg_sampling_cost_rows:
            fields_cost = [
                'epoch', 'step', 'batch_size', 'neg_num',
                'sampling_time', 'candidate_build_time', 'collision_check_time',
                'retry_time', 'output_build_time', 'total_neg_sampling_time',
                'avg_head_degree', 'avg_tail_degree', 'avg_entity_degree', 'max_entity_degree',
                'hub_entity_count', 'hub_top1_pct_count',
                'unique_entities', 'unique_relations',
                'avg_retry', 'max_retry', 'total_retry',
            ]
            path = os.path.join(out_dir, 'negative_sampling_cost.md')
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields_cost)
                writer.writeheader()
                writer.writerows(self.neg_sampling_cost_rows)
            print("[Profiling] Saved negative_sampling_cost.csv with {} rows".format(len(self.neg_sampling_cost_rows)))
        
        # === Phase 2: negative_sampling_breakdown.csv (Runtime Breakdown - Task 6) ===
        if self.neg_sampling_cost_rows:
            total_b1 = sum(r['sampling_time'] for r in self.neg_sampling_cost_rows)
            total_b2 = sum(r['candidate_build_time'] for r in self.neg_sampling_cost_rows)
            total_b3 = sum(r['collision_check_time'] for r in self.neg_sampling_cost_rows)
            total_b4 = sum(r['retry_time'] for r in self.neg_sampling_cost_rows)
            total_b5 = sum(r['output_build_time'] for r in self.neg_sampling_cost_rows)
            total_all_ns = sum(r['total_neg_sampling_time'] for r in self.neg_sampling_cost_rows)
            total_all_b = total_b1 + total_b2 + total_b3 + total_b4 + total_b5
            
            ns_breakdown_fields = ['Component', 'Time_ms', 'Ratio_pct']
            path = os.path.join(out_dir, 'negative_sampling_breakdown.md')
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(ns_breakdown_fields)
                for label, val in [
                    ('Sampling', total_b1),
                    ('Candidate Build', total_b2),
                    ('Collision Check', total_b3),
                    ('Retry', total_b4),
                    ('Output Build', total_b5),
                ]:
                    pct_of_total = (val / total_all_ns * 100) if total_all_ns > 0 else 0.0
                    writer.writerow([label, round(val, 3), round(pct_of_total, 2)])
                # Summary row
                writer.writerow(['Total (B1-B5)', round(total_all_b, 3), round(total_all_b / total_all_ns * 100, 2) if total_all_ns > 0 else 0.0])
                writer.writerow(['Total (neg_sampling_time)', round(total_all_ns, 3), 100.0])
            print("[Profiling] Saved negative_sampling_breakdown.csv")

    def test(self):
        predict = LinkPredictionEvaluator(self.model, self.args, self.kgs)
        predict.print_results()

    def retest(self):
        if self.model.__class__.__name__ == 'ConvE':
            model_params = torch.load(self.model.out_folder + 'conve.pth')
            total_param_size = []
            params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
            for key, size, count in params:
                total_param_size.append(count)
                print(key, size, count)
            print(np.array(total_param_size).sum())
            self.model.load_state_dict(model_params)
            self.model.eval()
        else:
            self.model.load_embeddings()
        self.model.to(self.device)
        t1 = time.time()
        predict = LinkPredictionEvaluator(self.model, self.args, self.kgs)
        predict.print_results()
        print('test cost time: {:.4f}s'.format(time.time() - t1))

    def get_pos_score(self, score):
        tmp = score[:self.batch_size]
        return tmp.view(self.batch_size, -1)

    def get_neg_score(self, score):
        tmp = score[self.batch_size:]
        # print(tmp.view(self.batch_size, -1).shape)
        return tmp.view(self.batch_size, -1)

    def save(self):
        if self.model.__class__.__name__ == 'ConvE':
            if not os.path.exists(self.model.out_folder):
                os.makedirs(self.model.out_folder)
            # print(self.state_dict())
            torch.save(self.model.state_dict(), self.model.out_folder + 'conve.pth')
            #self.model = torch.load(self.model.out_folder + 'conve.pth')
        else:
            self.model.save()


def get_pos_score(score, batch_size):
    tmp = score[:batch_size]
    return tmp.view(batch_size, -1)


def get_neg_score(score, batch_size):
    tmp = score[batch_size:]
    return tmp.view(batch_size, -1)


def trainer(config: Dict):
    global early_stop
    args = config["args"]
    kgs = config["kgs"]
    model = config["model"]
    # model = nn.Linear(4, 1)
    # model.module.generate()

    model = train.torch.prepare_model(model)
    valid = LinkPredictionEvaluator(model.module, args, kgs, is_valid=True)
    # model = train.torch.prepare_model(model)
    optimizer = get_optimizer_torch(args.optimizer, model, args.learning_rate)
    train_dataset = PyTorchTrainDataset(kgs.relation_triples_list, args.neg_triple_num, kgs)
    worker_batch_size = args.batch_size * args.num_worker // train.world_size()
    data_loader = DataLoader(train_dataset, batch_size=worker_batch_size,
                             collate_fn=train_dataset.collate_fn, shuffle=True,
                             pin_memory=True, num_workers=10)
    data_loader = train.torch.prepare_data_loader(data_loader)
    t = time.time()
    for i in range(1, args.max_epoch + 1):
        res = 0
        start = time.time()
        length = 0
        for data in data_loader:
            optimizer.zero_grad()
            data0 = data[0]
            data1 = data[1]
            data2 = data[2]
            # print(len(data[0]))
            data = {
                'batch_h': data0,
                'batch_r': data1,
                'batch_t': data2
            }
            score = model(data)
            if model.mudule.__class__.__name__ == 'ConvE' or model.module.__class__.__name__ == 'TuckER':
                length += 1
                score.backward()
                optimizer.step()
                res += score.item()
                continue
            batch_size = int(data0.shape[0] / (args.neg_triple_num + 1))
            po_score = get_pos_score(score, batch_size)
            ne_score = get_neg_score(score, batch_size)
            loss = get_loss_func_torch(po_score, ne_score, args)
            loss.backward()
            optimizer.step()
            res += loss.item()
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(i, res / length, time.time() - start))
        if i >= args.start_valid and i % args.eval_freq == 0:
            t1 = time.time()
            flag = valid.print_results()
            # print('valid cost time: {:.4f}s'.format(time.time() - start))
    print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
    predict = LinkPredictionEvaluator(model.module, args, kgs)
    predict.print_results()
    model.module.save()

    # print(f"Loss results: {result}")


class parallel_trainer(parallel_model):
    """Provides multi-process and multi-GPU parallel training for KGE models, inheriting class parallel_model

        Parameters
        ----------
        args: dict
            A python dict from muKG.src.py.args. It stored detailed information about model
            training and testing.
        kg: muKG.src.py.KG
            Store the whole information of a KG, like h_dict, r_dict, t_dict,
            train_dataset, valid_dataset, test_dataset and so on.
    """
    def __init__(self):
        super(parallel_trainer, self).__init__()
        self.kgs = None
        self.args = None
        self.early_stop = None
        self.flag2 = -1
        self.flag1 = -1
        self.NetworkActor = None

    def run(self):
        """Initialize ray with number of GPU or CPU.
        """
        self.args.device_number = min(self.args.num_worker, self.args.device_number)
        if self.args.is_gpu:
            ray.init(num_gpus=self.args.device_number)
        else:
            ray.init(num_cpus=self.args.device_number)
        self.train_fashion_mnist()

    def train_fashion_mnist(self):
        """
        Activate ray train by allocating device number and worker number.
        """
        device_allocate = self.args.device_number / self.args.num_worker
        device_allocate = min(device_allocate, 1)
        if self.args.is_gpu:
            from ray.train.torch import TorchTrainer
            trainer1 = TorchTrainer(
                train_func=trainer,
                train_loop_config={"args": self.args, "kgs": self.kgs, "model": self.model},
                scaling_config={"num_workers": self.args.num_worker, "use_gpu": self.args.is_gpu, "resources_per_worker": {"GPU": device_allocate}}
            )
        else:
            from ray.train.torch import TorchTrainer
            trainer1 = TorchTrainer(
                train_func=trainer,
                train_loop_config={"args": self.args, "kgs": self.kgs, "model": self.model},
                scaling_config={"num_workers": self.args.num_worker, "use_gpu": self.args.is_gpu, "resources_per_worker": {"CPU": device_allocate}}
            )
        trainer1.fit()

    def test(self):
        pass