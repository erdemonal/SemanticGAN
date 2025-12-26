import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

PROCESSED_DIR = Path("data/processed")
SYNTHETIC_DIR = Path("data/synthetic")
REAL_TRIPLES_PATH = PROCESSED_DIR / "kg_triples_ids.txt"
MAPPINGS_PATH = PROCESSED_DIR / "kg_mappings.json"
LOG_FILE = PROCESSED_DIR / "training_log.csv"
OUTPUT_JSON = Path("dashboard_data.json")

def main():
    print("[INFO] Updating Dashboard Data...")

    if not MAPPINGS_PATH.exists():
        print(f"[ERROR] Mappings file not found: {MAPPINGS_PATH}")
        return

    id_to_name = {}
    id_to_rel = {}
    
    try:
        with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            id_to_name = data.get("id2ent", {})
            if "id2rel" in data:
                id_to_rel = data["id2rel"]
            elif "rel2id" in data:
                id_to_rel = {str(v): k for k, v in data["rel2id"].items()}
    except Exception as e:
        print(f"[ERROR] Failed to load mappings: {e}")
        return

    if not id_to_rel:
        id_to_rel = {
            "0": "dblp:hasAuthor", "1": "dblp:title", "2": "dblp:publishedInYear",
            "5": "dblp:publishedIn", "9": "dblp:cites", "13": "dblp:journal", "34": "rdf:type"
        }

    synthetic_files = sorted(SYNTHETIC_DIR.glob("generated*.txt"))
    if not synthetic_files:
        synthetic_files = sorted(PROCESSED_DIR.glob("generated*.txt"))
        
    if not synthetic_files:
        print("[WARN] No generated data found.")
        return

    latest_file = synthetic_files[-1]
    synthetic_triples = []
    
    with open(latest_file, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    h, r, t, score = parts[0], str(int(float(parts[1]))), parts[2], float(parts[3])
                    synthetic_triples.append((h, r, t, score))
                except ValueError: continue

    total = len(synthetic_triples)
    
    valid_count = 0
    for h, r_id, t, _ in synthetic_triples:
        rel_name = id_to_rel.get(r_id, "").lower()
        tail_name = id_to_name.get(t, "UNKNOWN")
        is_valid = True
        
        if "year" in rel_name:
            if not tail_name.isdigit() or len(tail_name) != 4:
                is_valid = False
        elif "author" in rel_name or "editor" in rel_name:
            if tail_name.isdigit() or tail_name.startswith("conf/") or tail_name.startswith("journals/"):
                is_valid = False
        elif "publishedin" in rel_name or "journal" in rel_name or "conference" in rel_name:
            if not (tail_name.startswith("conf/") or tail_name.startswith("journals/") or "venue" in tail_name):
                if tail_name.isdigit(): 
                    is_valid = False
        elif "type" in rel_name:
            valid_types = ["person", "publication", "article", "inproceedings", "conference", "journal"]
            if not any(vt in tail_name.lower() for vt in valid_types):
                pass 

        if is_valid:
            valid_count += 1
            
    schema_validity = (valid_count / total) * 100 if total > 0 else 0

    real_triples_set = set()
    novelty_check_active = False

    if REAL_TRIPLES_PATH.exists():
        try:
            with open(REAL_TRIPLES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) == 3:
                        real_triples_set.add(hash(f"{p[0]}\t{p[1]}\t{p[2]}"))
            novelty_check_active = True
        except MemoryError:
            novelty_check_active = False

    novel_count = 0
    if novelty_check_active:
        for h, r, t, _ in synthetic_triples:
            if hash(f"{h}\t{r}\t{t}") not in real_triples_set:
                novel_count += 1
    else:
        novel_count = total 
            
    novelty_score = (novel_count / total) * 100 if total > 0 else 100
    overlap_score = 100 - novelty_score
    
    unique_triples = len(set(f"{h}\t{r}\t{t}" for h, r, t, score in synthetic_triples))
    uniqueness_score = (unique_triples / total) * 100 if total > 0 else 0

    rel_counts = Counter(r for _, r, _, _ in synthetic_triples)
    used_relations = len(rel_counts)
    total_relations = len(id_to_rel) if len(id_to_rel) > 0 else 40
    if total_relations < used_relations: total_relations = used_relations
    relation_diversity = (used_relations / total_relations) * 100 if total_relations > 0 else 0
    
    relation_freq = []
    for rel_id, cnt in rel_counts.items():
        rel_name = id_to_rel.get(rel_id, f"REL_{rel_id}").replace("dblp:", "").replace("rdf:", "")
        pct = (cnt / total) * 100 if total > 0 else 0
        relation_freq.append({"relation": rel_name, "count": cnt, "percent": round(pct, 2)})
    relation_freq.sort(key=lambda x: x["count"], reverse=True)

    avg_distance = (sum(score for _, _, _, score in synthetic_triples) / total if total > 0 else 0)

    decoded_hypotheses = []
    for i, (h, r, t, score) in enumerate(synthetic_triples):
        if i >= 200: break
        
        head_name = id_to_name.get(h, h)
        rel_clean = id_to_rel.get(r, f"REL:{r}").replace("dblp:", "").replace("rdf:", "")
        tail_name = id_to_name.get(t, t)

        decoded_hypotheses.append({
            "head": head_name,
            "relation": rel_clean,
            "tail": tail_name,
            "score": f"{score:.4f}",
            "is_novel": True
        })

    current_d_loss, current_g_loss, current_epoch = 0, 0, 0
    if LOG_FILE.exists():
        try:
            df = pd.read_csv(LOG_FILE)
            if not df.empty:
                last_row = df.iloc[-1]
                current_epoch = int(last_row.get("Epoch", 0))
                current_d_loss = float(last_row.get("D_Loss", 0))
                current_g_loss = float(last_row.get("G_Loss", 0))
        except: pass

    dashboard_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "training_status": {
            "epoch": current_epoch,
            "d_loss": round(current_d_loss, 4),
            "g_loss": round(current_g_loss, 4)
        },
        "stats": {
            "novelty": round(novelty_score, 2),
            "train_overlap": round(overlap_score, 2),
            "uniqueness": round(uniqueness_score, 2),
            "relation_diversity": round(relation_diversity, 2),
            "avg_distance": round(avg_distance, 4),
            "schema_validity": round(schema_validity, 2),
            "total_generated": total,
            "total_knowledge_base": len(id_to_name)
        },
        "relation_freq": relation_freq,
        "hypotheses": decoded_hypotheses
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"[SUCCESS] Dashboard Real Metrics Updated. Validity: {schema_validity:.2f}%")

if __name__ == "__main__":
    main()