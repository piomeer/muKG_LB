"""
Phase 9 Step 2 – Main Benchmark (4 configurations)
Adapted to actual MuKG project structure (src.py.* imports).
"""
import sys, os, time, csv, random
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.load.features import FeatureExtractor
from src.py.load.cost_model import build_cost_table
from src.py.load.schedulers import (
    Scheduler, RandomSorter, CostSorter, ChunkPacker, FFDPacker,
)
from src.py.load.batch_provider import BatchProvider
from src.py.load.gpu_sampler import GPUNegativeSampler


# ==================== Faithful CPU Original ====================
def original_cpu_neg_sampling(batch_triples, neg_num, n_entities, all_triples_set):
    """MuKG original: Bernoulli(0.5) head/tail + global all_triples_set collision"""
    heads_list, tails_list = [], []
    max_try = 10
    for h, r, t in batch_triples:
        for _ in range(neg_num):
            tries = 0
            if random.random() < 0.5:  # corrupt head
                while tries < max_try:
                    cand_h = random.randint(0, n_entities - 1)
                    if (cand_h, r, t) not in all_triples_set and cand_h != h:
                        heads_list.append(cand_h); tails_list.append(t); break
                    tries += 1
                else:
                    heads_list.append(h); tails_list.append(t)
            else:  # corrupt tail
                while tries < max_try:
                    cand_t = random.randint(0, n_entities - 1)
                    if (h, r, cand_t) not in all_triples_set and cand_t != t:
                        heads_list.append(h); tails_list.append(cand_t); break
                    tries += 1
                else:
                    heads_list.append(h); tails_list.append(t)
    return (torch.tensor(heads_list, dtype=torch.long),
            torch.tensor(tails_list, dtype=torch.long))


# ==================== Simple TransE ====================
class SimpleTransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        self.ent_embeddings = torch.nn.Embedding(num_entities, dim)
        self.rel_embeddings = torch.nn.Embedding(num_relations, dim)

    def forward(self, heads, rels, tails):
        h = self.ent_embeddings(heads)
        r = self.rel_embeddings(rels)
        t = self.ent_embeddings(tails)
        return torch.norm(h + r - t, p=2, dim=-1)


# ==================== Filtered MRR Evaluation (no float('inf')) ====================
@torch.no_grad()
def evaluate_filtered_mrr(model, triples_for_eval, all_triples_set,
                          num_entities, num_samples=500):
    """
    Filtered ranking evaluation (head + tail).
    Replaces float('inf') with simple masking: known true triples get score = 1e9
    so they sort to the end.
    """
    model.eval()

    # Build dicts for known true triples
    known_tails = {}  # (h,r) -> set of tails
    known_heads = {}  # (r,t) -> set of heads
    for h, r, t in all_triples_set:
        known_tails.setdefault((h, r), set()).add(t)
        known_heads.setdefault((r, t), set()).add(h)

    all_entities = torch.arange(num_entities, dtype=torch.long, device='cuda')

    ranks = []
    hits10_count = 0
    total = 0

    # Sample from eval triples
    if len(triples_for_eval) > num_samples:
        indices = random.Random(42).sample(range(len(triples_for_eval)), num_samples)
        eval_subset = [triples_for_eval[i] for i in indices]
    else:
        eval_subset = triples_for_eval

    for h, r, t in eval_subset:
        # ---- Tail ranking ----
        h_t = torch.tensor([h], dtype=torch.long, device='cuda')
        r_t = torch.tensor([r], dtype=torch.long, device='cuda')
        scores = model(h_t.repeat(num_entities), r_t.repeat(num_entities), all_entities)
        true_score = scores[t].item()

        # Mask known true tails (except this test triple) with 1e9
        if (h, r) in known_tails:
            for known_t in known_tails[(h, r)]:
                if known_t != t:
                    scores[known_t] = 1e9
        rank = (scores < true_score).sum().item() + 1
        ranks.append(rank)
        if rank <= 10:
            hits10_count += 1
        total += 1

        # ---- Head ranking ----
        t_t = torch.tensor([t], dtype=torch.long, device='cuda')
        scores = model(all_entities, r_t.repeat(num_entities), t_t.repeat(num_entities))
        true_score = scores[h].item()

        if (r, t) in known_heads:
            for known_h in known_heads[(r, t)]:
                if known_h != h:
                    scores[known_h] = 1e9
        rank = (scores < true_score).sum().item() + 1
        ranks.append(rank)
        if rank <= 10:
            hits10_count += 1
        total += 1

    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    hits10 = hits10_count / total
    return mrr, hits10


# ==================== Run Single Config ====================
def run_config(label, use_gpu, sorter, packer,
               train_triples, all_triples_set, eval_triples,
               n_entities, n_relations, cost_table,
               epochs=5, batch_size=5000, neg_num=150):
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    random.seed(42 + hash(label) % 1000)

    out_dir = f'output/results/phase9_step2/{label}'
    os.makedirs(out_dir, exist_ok=True)

    model = SimpleTransE(n_entities, n_relations, dim=400).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler_obj = GPUNegativeSampler(n_entities, neg_num) if use_gpu else None

    scheduler = Scheduler(sorter, packer)
    provider = BatchProvider(scheduler, cost_table, batch_size, enable_logging=False)

    # CSV writers
    summary_path = os.path.join(out_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as sf:
        writer = csv.writer(sf)
        writer.writerow(['epoch', 'avg_loss', 'mrr', 'hits10', 'epoch_time_s', 'gpu_mem_mb'])

        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            t_start = time.time()

            for step, batch in enumerate(provider.iterate(train_triples)):
                optimizer.zero_grad()

                if use_gpu:
                    neg_h, neg_t = sampler_obj.generate(batch)
                else:
                    neg_h, neg_t = original_cpu_neg_sampling(
                        batch, neg_num, n_entities, all_triples_set)
                    neg_h = neg_h.cuda()
                    neg_t = neg_t.cuda()

                pos_h = torch.tensor([t[0] for t in batch], dtype=torch.long, device='cuda')
                pos_r = torch.tensor([t[1] for t in batch], dtype=torch.long, device='cuda')
                pos_t = torch.tensor([t[2] for t in batch], dtype=torch.long, device='cuda')

                pos_scores = model(pos_h, pos_r, pos_t)
                neg_scores = model(neg_h, pos_r.repeat_interleave(neg_num), neg_t)

                loss = torch.mean(torch.clamp(
                    pos_scores[:, None] - neg_scores.view(-1, neg_num) + 1.0, min=0
                ))
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

                if step % 30 == 0:
                    print(f"  [{label}] E{epoch} S{step:3d} loss={loss.item():.4f}")

            epoch_time = time.time() - t_start
            avg_loss = float(np.mean(epoch_losses))

            # Evaluate MRR
            mrr, hits10 = evaluate_filtered_mrr(
                model, eval_triples, all_triples_set, n_entities, num_samples=200
            )
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            torch.cuda.reset_peak_memory_stats()

            writer.writerow([epoch, f'{avg_loss:.6f}', f'{mrr:.4f}',
                             f'{hits10:.4f}', f'{epoch_time:.1f}', f'{gpu_mem:.0f}'])
            print(f"  [{label}] Epoch {epoch}: loss={avg_loss:.4f} "
                  f"MRR={mrr:.4f} Hits@10={hits10:.4f} "
                  f"time={epoch_time:.1f}s mem={gpu_mem:.0f}MB")

    print(f"  {label} done → {summary_path}")


# ==================== Main ====================
def main():
    print("Loading FB15k-237 dataset...")
    args_path = os.path.join(os.path.dirname(__file__), 'args_kge', 'transe_fb15k237_args.json')
    cmd_args = load_args(args_path)
    cmd_args.is_torch = True
    kgs = read_kgs_from_folder(
        'lp', cmd_args.training_data, cmd_args.dataset_division,
        cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False,
    )

    train_triples = kgs.local_relation_triples_list
    n_entities = kgs.entities_num
    n_relations = kgs.relations_num
    all_triples_set = set(train_triples)

    # Use 5k as eval subset (from train, shuffled)
    random.seed(42)
    shuffled = list(train_triples)
    random.shuffle(shuffled)
    eval_triples = shuffled[:5000]
    train_for_training = shuffled[5000:]

    print(f"  Entities: {n_entities}, Relations: {n_relations}")
    print(f"  Train: {len(train_for_training)}, Eval: {len(eval_triples)}")

    # Build cost table (shared)
    extractor = FeatureExtractor(train_for_training, n_entities)
    features = extractor.build()
    cost_table = build_cost_table(features, neg_num=150)
    print(f"  Cost table: mean={cost_table.mean():.2f}ms")

    # Configs (sorted by expected speed)
    configs = [
        ('CBP', False, CostSorter(), FFDPacker()),
        ('BL',  False, RandomSorter(seed=42), ChunkPacker()),
        ('CBP+GPU', True, CostSorter(), FFDPacker()),
        ('GPU', True, RandomSorter(seed=42), ChunkPacker()),
    ]

    for cfg in configs:
        run_config(*cfg, train_for_training, all_triples_set, eval_triples,
                   n_entities, n_relations, cost_table,
                   epochs=5, batch_size=5000, neg_num=150)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    summary_file = 'output/results/phase9_step2/summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'final_loss', 'mrr', 'hits10', 'avg_epoch_time_s', 'gpu_mem_mb'])
        for label, _, _, _ in configs:
            sp = f'output/results/phase9_step2/{label}/summary.csv'
            if os.path.exists(sp):
                rows = list(csv.DictReader(open(sp)))
                last = rows[-1]
                avg_time = np.mean([float(r['epoch_time_s']) for r in rows])
                mem = rows[-1]['gpu_mem_mb']
                writer.writerow([
                    label, last['avg_loss'], last['mrr'], last['hits10'],
                    f'{avg_time:.1f}', mem
                ])
                print(f"  {label:10s} | loss={last['avg_loss']} | MRR={last['mrr']} | "
                      f"Hits@10={last['hits10']} | time={avg_time:.1f}s | mem={mem}MB")

    print(f"\n  Comprehensive summary → {summary_file}")


if __name__ == '__main__':
    main()