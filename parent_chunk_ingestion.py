import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# ─── CONFIGURATION ──────────────────────────────────────────
INPUT_DIR  = "extracted_chunks_1"      # Your existing chunk JSONs
OUTPUT_DIR = "parent_pages"            # Where to save full‑page parents
MERGE_ADJACENT_PAGES = True            # Whether to merge every 2 pages
PAGES_PER_GROUP = 2                    # Group size (2 = pair, 3 = trio, etc.)

# ─── LOAD ALL CHUNKS ────────────────────────────────────────
def load_all_chunks(input_dir: str) -> list:
    docs = []
    for json_file in sorted(Path(input_dir).glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                docs.append({
                    "page_content": item["page_content"],
                    "metadata": item["metadata"]
                })
    return docs

# ─── GROUP CHUNKS BY (doc_id, page) ─────────────────────────
def group_chunks(docs: list) -> Dict[tuple, list]:
    groups = defaultdict(list)
    for doc in docs:
        meta = doc["metadata"]
        key = (meta["doc_id"], meta["page"])
        groups[key].append(doc)
    return groups

# ─── BUILD PARENT DOCUMENTS (PER PAGE) ──────────────────────
def build_page_parents(groups: Dict[tuple, list]) -> list:
    parents = []
    for (doc_id, page), chunks in groups.items():
        # Sort by chunk_id to keep original order
        chunks.sort(key=lambda c: c["metadata"]["chunk_id"])
        full_text = " ".join([c["page_content"] for c in chunks])
        # Use metadata from the first chunk, plus a few modifications
        parent_meta = dict(chunks[0]["metadata"])
        parent_meta["is_parent"] = True
        parent_meta["parent_type"] = f"single_page_{page}"
        # Remove chunk‑specific fields that no longer make sense
        parent_meta.pop("chunk_id", None)
        parent_meta.pop("heading", None)  # optional, you can keep it
        parents.append({
            "page_content": full_text,
            "metadata": parent_meta
        })
    return parents

# ─── (OPTIONAL) MERGE CONSECUTIVE PAGES ─────────────────────
def merge_consecutive_pages(parents: list, pages_per_group: int = 2) -> list:
    """
    Group parents from the same document into blocks of `pages_per_group`.
    Example: pages 1-2, 3-4, 5-6 … become one parent each.
    """
    # Sort parents by doc_id and page
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
            # Finalise the previous group
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

# ─── SAVE TO JSON ───────────────────────────────────────────
def save_parents(parents: list, output_dir: str):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    # Option 1: Save all parents into one big JSON file
    all_file = out_path / "all_parents.json"
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(parents)} parent documents → {all_file}")

    # Option 2: Also save one JSON per original PDF (grouped by doc_id)
    by_doc = defaultdict(list)
    for p in parents:
        by_doc[p["metadata"]["doc_id"]].append(p)
    for doc_id, docs in by_doc.items():
        # Find a representative filename (strip the doc_id prefix)
        filename = docs[0]["metadata"]["filename"]
        file_path = out_path / f"{doc_id}_{Path(filename).stem}_parents.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Also saved per‑document parent files in {out_path}")

# ─── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_all_chunks(INPUT_DIR)
    print(f"Loaded {len(docs)} child chunks.")

    groups = group_chunks(docs)
    print(f"Found {len(groups)} unique (doc_id, page) pairs.")

    page_parents = build_page_parents(groups)
    print(f"Created {len(page_parents)} single‑page parent documents.")

    if MERGE_ADJACENT_PAGES:
        final_parents = merge_consecutive_pages(page_parents, PAGES_PER_GROUP)
        print(f"Merged into {len(final_parents)} parent groups (size {PAGES_PER_GROUP}).")
    else:
        final_parents = page_parents

    save_parents(final_parents, OUTPUT_DIR)