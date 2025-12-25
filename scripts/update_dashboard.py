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

    print(" - Loading ID Mappings (this might take a moment)...")
    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        id_to_name = data.get("id2ent", {})  

    synthetic_files = sorted(SYNTHETIC_DIR.glob("generated_*.txt"))
    if not synthetic_files:
        print("[WARN] No generated data found.")
        return

    latest_file = synthetic_files[-1]
    synthetic_triples = []
    print(f" - Analyzing latest generation: {latest_file.name}")
    
    with open(latest_file, "r", encoding="utf-8") as f:
        next(f)  
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                h, r, t, score = parts[0], parts[1], parts[2], float(parts[3])
                synthetic_triples.append((h, r, t, score))

    print(" - Loading Real Triples for Novelty Comparison...")
    real_triples_set = set()
    all_relations = set()
    
    try:
        with open(REAL_TRIPLES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) == 3:
                    real_triples_set.add(hash(f"{p[0]}\t{p[1]}\t{p[2]}"))
                    all_relations.add(p[1])
    except MemoryError:
        print("[WARN] Memory full! Novelty score might be approximate.")

    total = len(synthetic_triples)
    
    novel_count = 0
    for h, r, t, _ in synthetic_triples:
        if hash(f"{h}\t{r}\t{t}") not in real_triples_set:
            novel_count += 1
            
    overlap_count = total - novel_count
    novelty_score = (novel_count / total) * 100 if total > 0 else 0
    overlap_score = (overlap_count / total) * 100 if total > 0 else 0

    unique_triples = len(set(f"{h}\t{r}\t{t}" for h, r, t, score in synthetic_triples))
    uniqueness_score = (unique_triples / total) * 100 if total > 0 else 0

    rel_counts = Counter(r for _, r, _, _ in synthetic_triples)
    used_relations = len(rel_counts)
    total_relations = len(all_relations) if all_relations else 10
    relation_diversity = (used_relations / total_relations) * 100 if total_relations > 0 else 0
    
    relation_freq = []
    for rel, cnt in rel_counts.items():
        pct = (cnt / total) * 100 if total > 0 else 0
        relation_freq.append({"relation": rel, "count": cnt, "percent": round(pct, 2)})
    relation_freq.sort(key=lambda x: x["relation"])

    avg_distance = (sum(score for _, _, _, score in synthetic_triples) / total if total > 0 else 0)


    relation_tail_rules = {
        "dblp:hasAuthor": "author",
        "dblp:hasEditor": "author",
        "dblp:conferenceSeries": "venue/conf",
        "dblp:journalID": "venue/journal",
        "dblp:listedIn": "venue",
        "dblp:presentedAt": "venue",
        "dblp:publishedInJournal": "venue",
        "dblp:publishedInYear": "year",
        "dblp:conferenceYear": "year",
        "dblp:hasDOI": "doi",
        "dblp:hasLink": "link",
        "dblp:partOf": "collection",
        "dblp:cites": "pub",
        "dblp:coauthorWith": "author",
        "dblp:primaryName": "name",
        "dblp:variantName": "name",
        "dblp:baseName": "name",
        "dblp:homonymID": "id",
        "dblp:affiliation": "org",
        "dblp:title": "", 
        "dblp:volume": "",
        "dblp:volumeFromKey": "",
        "dblp:issueNumber": "",
        "dblp:pages": "",
        "dblp:publisher": "",
        "dblp:address": "",
        "dblp:isbn": "",
        "dblp:issn": "",
        "dblp:series": "",
        "dblp:seriesPage": "",
        "dblp:school": "",
        "dblp:cdrom": "",
        "dblp:publishedInMonth": "",
        "dblp:extractedDOI": "",
        "dblp:lastModified": "",
        "rdf:type": "dblp",
    }
    relation_head_rules = {
        "dblp:hasAuthor": "conf", 
        "dblp:hasEditor": "conf",
        "dblp:conferenceSeries": "conf",
        "dblp:journalID": "journals",
        "dblp:listedIn": "conf",
        "dblp:presentedAt": "conf",
        "dblp:publishedInJournal": "journals",
        "dblp:publishedInYear": "conf",
        "dblp:conferenceYear": "conf",
        "dblp:hasDOI": "conf",
        "dblp:hasLink": "conf",
        "dblp:partOf": "conf",
        "dblp:cites": "conf",
        "dblp:title": "conf",
        "dblp:volume": "conf",
        "dblp:volumeFromKey": "conf",
        "dblp:issueNumber": "conf",
        "dblp:pages": "conf",
        "dblp:publisher": "conf",
        "dblp:address": "conf",
        "dblp:isbn": "conf",
        "dblp:issn": "conf",
        "dblp:series": "conf",
        "dblp:seriesPage": "conf",
        "dblp:school": "conf",
        "dblp:cdrom": "conf",
        "dblp:publishedInMonth": "conf",
        "dblp:extractedDOI": "conf",
        "dblp:lastModified": "conf",
        "dblp:coauthorWith": "author",
        "dblp:primaryName": "homepages",
        "dblp:variantName": "homepages",
        "dblp:baseName": "author",
        "dblp:homonymID": "author",
        "dblp:affiliation": "homepages",
        "rdf:type": "conf",
    }
    
    valid_count = 0
    for h, r, t, _ in synthetic_triples:
        allowed_head = relation_head_rules.get(r, None)
        allowed_tail = relation_tail_rules.get(r, None)
        
        if allowed_head is None and allowed_tail is None:
            valid_count += 1
        elif allowed_head is not None and allowed_tail is not None:
            head_valid = (allowed_head == "" or h.startswith(allowed_head) or 
                         any(h.startswith(prefix) for prefix in ["conf/", "journals/", "homepages/", "venue/"]))
            tail_valid = (allowed_tail == "" or t.startswith(allowed_tail) or 
                         any(t.startswith(prefix) for prefix in ["venue/", "author", "dblp:"]))
            if head_valid and tail_valid:
                valid_count += 1
        else:
            valid_count += 1
            
    schema_validity = (valid_count / total) * 100 if total > 0 else 0

    decoded_hypotheses = []
    for i, (h, r, t, score) in enumerate(synthetic_triples):
        if i >= 200: break
        
        head_name = id_to_name.get(h, h)
        tail_name = id_to_name.get(t, t)
        
        rel_clean = r.replace("dblp:", "")
        
        decoded_hypotheses.append({
            "head": head_name,
            "relation": rel_clean,
            "tail": tail_name,
            "score": f"{score:.4f}",
            "is_novel": hash(f"{h}\t{r}\t{t}") not in real_triples_set
        })

    current_d_loss = 0
    current_g_loss = 0
    current_epoch = 0
    
    if LOG_FILE.exists():
        try:
            df = pd.read_csv(LOG_FILE)
            if not df.empty:
                last_row = df.iloc[-1]
                current_epoch = int(last_row["Epoch"])
                current_d_loss = float(last_row["D_Loss"])
                current_g_loss = float(last_row["G_Loss"])
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

    print(f"[SUCCESS] Dashboard JSON saved to {OUTPUT_JSON}")
    print(f" - Validity: {schema_validity:.1f}%")
    print(f" - Novelty: {novelty_score:.1f}%")

if __name__ == "__main__":
    main()