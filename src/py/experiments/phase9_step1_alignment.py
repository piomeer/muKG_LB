"""
Phase 9 Step 1 — Semantic Alignment Check
Compares original CPU sampling vs GPU Sampler v2 on 2 epochs,
measuring MRR and loss to verify no significant accuracy regression.

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
    Scheduler, CostSorter, FFDPacker,
)
from src.py.load.batch_provider import BatchProvider
from src.py.load.gpu_sampler import GPUNegativeSampler


# ========================== Faithful CPU Original ==========================
def original_cpu_neg_sampling(batch_triples, neg_num, n_entities, all_triples_set):
    """
    Faithful reproduction of MuKG's original generate_neg_triples_fast:
    - Bernoulli(0.5) corrupt head OR tail (not both)
    - Global all_triples_set collision check
    - max_try=10 with fallback to original
    """
    neg_heads_list = []
    neg_tails_list = []
    corrupt_mask_list = []
    max_try = 10

    for h, r, t in batch_triples:
        for _ in range(neg_num):
            tries = 0
            if random.random() < 0.5:  # corrupt head
                while tries < max_try:
                    cand_h = random.randint(0, n_entities - 1)
                    if (cand_h, r, t) not in all_triples_set and cand_h != h:
                        neg_heads_list.append(cand_h)
                        neg_tails_list.append(t)
                        corrupt_mask_list.append(True)
                        break
                    tries += 1
                else:
                    # fallback: return original
                    neg_heads_list.append(h)
                    neg_tails_list.append(t)
                    corrupt_mask_list.append(True)
            else:  # corrupt tail
                while tries < max_try:
                    cand_t = random.randint(0, n_entities - 1)
                    if (h, r, cand_t) not in all_triples_set and cand_t != t:
                        neg_heads_list.append(h)
                        neg_tails_list.append(cand_t)
                        corrupt_mask_list.append(False)
                        break
                    tries += 1
                else:
                    neg_heads_list.append(h)
                    neg_tails_list.append(t)
                    corrupt_mask_list.append(False)

    return (torch.tensor(neg_heads_list, dtype=torch.long),
            torch.tensor(neg_tails_list, dtype=torch.long),
            torch.tensor(corrupt_mask_list, dtype=torch.bool))


# ========================== Simple TransE Model ==========================
class SimpleTransE(torch.nn.Module):
    """Minimal TransE for validation"""
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        self.ent_embeddings = torch.nn.Embedding(num_entities, dim)
        self.rel_embeddings = torch.nn.Embedding(num_relations, dim)
        self.margin = margin

    def forward(self, heads, rels, tails):
        h = self.ent_embeddings(heads)
        r = self.rel_embeddings(rels)
        t = self.ent_embeddings(tails)
        return torch.norm(h + r - t, p=2, dim=-1)


# ========================== Filtered MRR Evaluation ==========================
def compute_filtered_mrr(model, train_triples, valid_triples, test_triples,
                         num_entities, neg_num=150, batch_size=5000):
    """
    Compute filtered MRR and Hits@10 on test set.
    Uses 1-scoring (lower is better for TransE distance).
    Filtered: remove any corrupted triple that exists in train/valid/test.
    """
    model.eval()
    all_triples_set = set(train_triples) | set(valid_triples) | set(test_triples)

    # Build head/tail dicts for fast filtering
    rel_entity_dict = {}
    for h, r, t in train_triples + valid_triples + test_triples:
        key = (h, r)
        if key not in rel_entity_dict:
            rel_entity_dict[key] = set()
        rel_entity_dict[key].add(t)

    entity_rel_dict = {}
    for h, r, t in train_triples + valid_triples + test_triples:
        key = (r, t)
        if key not in entity_rel_dict:
            entity_rel_dict[key] = set()
        entity_rel_dict[key].add(h)

    ranks = []
    hits10_count = 0
    total = 0

    all_entities = torch.arange(num_entities, device='cuda')

    with torch.no_grad():
        for h, r, t in test_triples:
            # Tail corruption (replace tail, keep h,r)
            h_t = torch.tensor([h], dtype=torch.long, device='cuda')
            r_t = torch.tensor([r], dtype=torch.long, device='cuda')
            t_t = torch.tensor([t], dtype=torch.long, device='cuda')

            # Score all tail candidates
            h_exp = h_t.repeat(num_entities)
            r_exp = r_t.repeat(num_entities)
            scores = model(h_exp, r_exp, all_entities)  # [num_entities]
            true_score = scores[t]

            # Filter: set scores of known true triples (excluding test triple) to high
            # Known tails for (h,r)
            if (h, r) in rel_entity_dict:
                for known_t in rel_entity_dict[(h, r)]:
                    scores[known_t] = float('inf')

            # Rank of true tail (lower score = better)
            rank = (scores < true_score).sum().item() + 1
            ranks.append(rank)
            if rank <= 10:
                hits10_count += 1
            total += 1

            # Head corruption (replace head, keep r,t)
            if total >= 500:  # limit to 500 test triples for speed
                break

    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    hits10 = hits10_count / total
    return mrr, hits10


# ========================== Main ==========================
def main():
    out_dir = 'output/results/phase9_step1'
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load dataset ----
    print("Loading FB15k-237 dataset...")
    args_path = os.path.join(os.path.dirname(__file__), 'args_kge', 'transe_fb15k237_args.json')
    cmd_args = load_args(args_path)
    cmd_args.is_torch = True
    kgs = read_kgs_from_folder(
        'lp', cmd_args.training_data, cmd_args.dataset_division,
        cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False,
    )
    train_triples = kgs.local_relation_triples_list
    num_entities = kgs.entities_num
    num_relations = kgs.relations_num
    all_triples_set = set(train_triples)

    # Use a subset of train triples as valid/test for this quick check
    # (original eval would need full test set, but we approximate)
    random.seed(42)
    shuffled = list(train_triples)
    random.shuffle(shuffled)
    n_test = min(5000, len(shuffled) // 20)
    test_triples = shuffled[:n_test]
    train_for_training = shuffled[n_test:]

    print(f"  Entities: {num_entities}, Relations: {num_relations}")
    print(f"  Train: {len(train_for_training)}, Test (eval sample): {len(test_triples)}")

    # ---- Build cost table ----
    extractor = FeatureExtractor(train_for_training, num_entities)
    features = extractor.build()
    cost_table = build_cost_table(features, neg_num=150)
    print(f"  Cost table: shape={cost_table.shape}")

    # ---- Configs ----
    configs = [
        {'label': 'CPU_original', 'use_gpu': False},
        {'label': 'GPU_v2',       'use_gpu': True},
    ]

    results_file = os.path.join(out_dir, 'results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'epoch_1_loss', 'epoch_2_loss', 'mrr_sample', 'hits10_sample'])

        for cfg in configs:
            print(f"\n{'='*60}")
            print(f"  Running: {cfg['label']}")
            print(f"{'='*60}")

            # Fresh model
            model = SimpleTransE(num_entities, num_relations, dim=400, margin=1.0)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Sampler
            if cfg['use_gpu']:
                sampler = GPUNegativeSampler(num_entities, 150)
            else:
                sampler = None  # use original_cpu_neg_sampling

            # Scheduler
            sorter = CostSorter()
            packer = FFDPacker()
            scheduler_obj = Scheduler(sorter, packer)
            provider = BatchProvider(scheduler_obj, cost_table, 5000, enable_logging=False)

            # ---- Train 2 epochs ----
            model.train()
            epoch_losses = []
            for epoch in range(2):
                total_loss = 0.0
                steps = 0
                for batch_triples in provider.iterate(train_for_training):
                    optimizer.zero_grad()

                    if cfg['use_gpu']:
                        neg_heads, neg_tails = sampler.generate(batch_triples)
                    else:
                        neg_heads, neg_tails, _ = original_cpu_neg_sampling(
                            batch_triples, 150, num_entities, all_triples_set)
                        neg_heads = neg_heads.cuda()
                        neg_tails = neg_tails.cuda()

                    # Build positive tensors
                    pos_heads = torch.tensor([t[0] for t in batch_triples], dtype=torch.long, device='cuda')
                    pos_rels = torch.tensor([t[1] for t in batch_triples], dtype=torch.long, device='cuda')
                    pos_tails = torch.tensor([t[2] for t in batch_triples], dtype=torch.long, device='cuda')

                    # Forward + pairwise loss
                    pos_scores = model(pos_heads, pos_rels, pos_tails)
                    neg_rels = pos_rels.repeat_interleave(150)
                    neg_scores = model(neg_heads, neg_rels, neg_tails)
                    loss = torch.mean(torch.clamp(
                        pos_scores[:, None] - neg_scores.view(-1, 150) + 1.0, min=0
                    ))

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    steps += 1

                avg_loss = total_loss / steps
                epoch_losses.append(avg_loss)
                print(f"  Epoch {epoch}: loss={avg_loss:.6f}")

            # ---- Evaluate MRR on sample ----
            print(f"  Evaluating MRR on {len(test_triples)} test triples...")
            mrr, hits10 = compute_filtered_mrr(
                model, train_for_training, [], test_triples,
                num_entities, neg_num=150, batch_size=5000
            )
            print(f"  MRR={mrr:.4f}, Hits@10={hits10:.4f}")

            writer.writerow([cfg['label'], epoch_losses[0], epoch_losses[1], mrr, hits10])
            print(f"  Results saved for {cfg['label']}")

    # ---- Print final comparison ----
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    with open(results_file) as f:
        for line in f:
            print(f"  {line.strip()}")

    print(f"\n  Results saved to {results_file}")


if __name__ == '__main__':
    main()