import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# ─── CONFIGURATION ──────────────────────────────────────────
INPUT_DIR  = "extracted_chunks_1"      # Where child chunk JSONs land
OUTPUT_DIR = "parent_pages"            # Parent store
STATE_FILE = "parent_state.json"       # Tracks processed doc_ids
MERGE_ADJACENT_PAGES = True
PAGES_PER_GROUP = 2

# ─── STATE MANAGEMENT ───────────────────────────────────────
def load_state() -> set:
    """Return set of doc_ids that have already been processed."""
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_state(processed: set):
    with open(STATE_FILE, "w") as f:
        json.dump(list(processed), f)

# ─── LOAD CHILD CHUNKS (only new ones) ──────────────────────
def load_new_chunks(input_dir: str, already_processed: set) -> list:
    docs = []
    for json_file in sorted(Path(input_dir).glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                doc_id = item["metadata"]["doc_id"]
                if doc_id not in already_processed:
                    docs.append(item)
    return docs

# ─── GROUP CHUNKS (only new) ────────────────────────────────
def group_chunks(docs: list) -> Dict[tuple, list]:
    groups = defaultdict(list)
    for doc in docs:
        meta = doc["metadata"]
        key = (meta["doc_id"], meta["page"])
        groups[key].append(doc)
    return groups

# ─── BUILD PARENT DOCUMENTS ─────────────────────────────────
def build_page_parents(groups: Dict[tuple, list]) -> list:
    parents = []
    for (doc_id, page), chunks in groups.items():
        chunks.sort(key=lambda c: c["metadata"]["chunk_id"])
        full_text = " ".join([c["page_content"] for c in chunks])
        parent_meta = dict(chunks[0]["metadata"])
        parent_meta["is_parent"] = True
        parent_meta["parent_type"] = f"single_page_{page}"
        parent_meta.pop("chunk_id", None)
        parent_meta.pop("heading", None)
        parents.append({
            "page_content": full_text,
            "metadata": parent_meta
        })
    return parents

# ─── (OPTIONAL) MERGE CONSECUTIVE PAGES ─────────────────────
def merge_consecutive_pages(parents: list, pages_per_group: int = 2) -> list:
    parents.sort(key=lambda p: (p["metadata"]["doc_id"], p["metadata"]["page"]))
    merged = []
    current_group = []
    for p in parents:
        if not current_group or (
            p["metadata"]["doc_id"] == current_group[-1]["metadata"]["doc_id"] and
            len(current_group) < pages_per_group
        ):
            current_group.append(p)
        else:
            merged.append(_merge_group(current_group))
            current_group = [p]
    if current_group:
        merged.append(_merge_group(current_group))
    return merged

def _merge_group(group: list) -> dict:
    first = group[0]
    last  = group[-1]
    full_text = " ".join([d["page_content"] for d in group])
    new_meta = dict(first["metadata"])
    new_meta["page"] = f"{first['metadata']['page']}-{last['metadata']['page']}"
    new_meta["parent_type"] = f"merged_pages_{new_meta['page']}"
    return {"page_content": full_text, "metadata": new_meta}

# ─── SAVE / APPEND PARENTS ──────────────────────────────────
def append_parents(new_parents: list, output_dir: str):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    all_file = out_path / "all_parents.json"
    
    # Load existing parents if the file exists
    if all_file.exists():
        with open(all_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    
    # Append new parents
    existing.extend(new_parents)
    
    # Save back
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"Appended {len(new_parents)} new parent(s) → total {len(existing)} in {all_file}")

# ─── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    processed = load_state()
    print(f"Already processed {len(processed)} document(s).")

    new_chunks = load_new_chunks(INPUT_DIR, processed)
    if not new_chunks:
        print("No new chunks found. Nothing to do.")
        exit()

    print(f"Found {len(new_chunks)} new child chunks.")

    groups = group_chunks(new_chunks)
    page_parents = build_page_parents(groups)

    if MERGE_ADJACENT_PAGES:
        final_parents = merge_consecutive_pages(page_parents, PAGES_PER_GROUP)
    else:
        final_parents = page_parents

    append_parents(final_parents, OUTPUT_DIR)

    # Mark these doc_ids as processed
    new_doc_ids = set(doc["metadata"]["doc_id"] for doc in new_chunks)
    processed.update(new_doc_ids)
    save_state(processed)
    print(f"State updated. Total processed documents: {len(processed)}")