import time
from collections import Counter
import os

def solve():
    train_path = 'src/py/data/FB15K237/train2id.txt'
    entity2id_path = 'src/py/data/FB15K237/entity2id.txt'
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    # Load entity mapping
    id2entity = {}
    if os.path.exists(entity2id_path):
        with open(entity2id_path, 'r') as f:
            next(f) # skip count
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    id2entity[parts[1]] = parts[0]

    start_time = time.time()
    
    counter = Counter()
    with open(train_path, 'r') as f:
        line_count = f.readline() # read first line (total triples)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                h, t, r = parts[0], parts[1], parts[2]
                counter[h] += 1
                counter[t] += 1
                
    # Get top 20 hub nodes
    top_20 = counter.most_common(20)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Top 20 Hub Nodes in FB15k-237 (Entity ID, Frequency, Entity Name):")
    for eid, freq in top_20:
        name = id2entity.get(eid, "Unknown")
        print(f"ID: {eid:6} | Freq: {freq:5} | Name: {name}")
        
    print(f"\nTime taken to select hub nodes: {duration:.4f} seconds")

if __name__ == "__main__":
    solve()
