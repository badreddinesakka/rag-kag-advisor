"""
index_rfp.py — Met les enfants de chunks_rfp.json dans Milvus.

Modele d'embedding : bge-m3 via Ollama (1024 dimensions).
Index : FLAT + COSINE (89 morceaux, la recherche exhaustive est instantanee
et donne le resultat exact : aucune raison d'approximer avec HNSW).

Usage :
    python index_rfp.py --recreate
    python index_rfp.py --entree chunks_rfp.json --collection rfp_criteres
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from pymilvus import DataType, MilvusClient

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")


def embed(texte: str, modele: str) -> list:
    """Un vecteur pour un texte, via Ollama."""
    corps = json.dumps({"model": modele, "prompt": texte}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=corps,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["embedding"]


def main():
    ap = argparse.ArgumentParser(description="Indexation Milvus du RFP")
    ap.add_argument("--entree", default="chunks_rfp.json")
    ap.add_argument("--collection", default="rfp_criteres")
    ap.add_argument("--modele", default="bge-m3")
    ap.add_argument("--recreate", action="store_true",
                    help="supprime la collection existante avant de la recreer")
    args = ap.parse_args()

    chemin = Path(args.entree)
    if not chemin.exists():
        sys.exit(f"{chemin} introuvable. Lance d'abord chunk_rfp.py")

    doc = json.loads(chemin.read_text(encoding="utf-8"))
    enfants = doc["enfants"]
    print(f"{len(enfants)} enfants a indexer ({doc['n_parents']} parents)")

    # --- vecteur d'essai : on decouvre la dimension au lieu de la supposer ---
    print(f"Modele d'embedding : {args.modele}")
    v0 = embed(enfants[0]["texte"], args.modele)
    dim = len(v0)
    print(f"Dimension detectee : {dim}")

    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(args.collection):
        if not args.recreate:
            sys.exit(
                f"La collection '{args.collection}' existe deja. "
                "Relance avec --recreate pour l'ecraser."
            )
        client.drop_collection(args.collection)
        print(f"Collection '{args.collection}' supprimee")

    # --- schema explicite : les varchar sont dimensionnes sur les vrais textes ---
    max_texte = max(len(e["texte"].encode("utf-8")) for e in enfants)
    max_titre = max(len(e["titre"].encode("utf-8")) for e in enfants)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vecteur", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("enfant_id", DataType.VARCHAR, max_length=16)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=16)
    schema.add_field("section", DataType.VARCHAR, max_length=16)
    schema.add_field("titre", DataType.VARCHAR, max_length=max_titre + 64)
    schema.add_field("tableau", DataType.BOOL)
    schema.add_field("texte", DataType.VARCHAR, max_length=max_texte + 512)

    index = MilvusClient.prepare_index_params()
    index.add_index(field_name="vecteur", index_type="FLAT", metric_type="COSINE")

    client.create_collection(args.collection, schema=schema, index_params=index)
    print(f"Collection '{args.collection}' creee (FLAT / COSINE / dim {dim})")

    lignes = []
    for i, e in enumerate(enfants, start=1):
        vec = v0 if i == 1 else embed(e["texte"], args.modele)
        lignes.append(
            {
                "vecteur": vec,
                "enfant_id": e["enfant_id"],
                "parent_id": e["parent_id"],
                "section": e["section"],
                "titre": e["titre"],
                "tableau": e["tableau"],
                "texte": e["texte"],
            }
        )
        if i % 20 == 0 or i == len(enfants):
            print(f"  {i}/{len(enfants)} vecteurs")

    client.insert(args.collection, lignes)
    client.flush(args.collection)

    n = client.get_collection_stats(args.collection)["row_count"]
    print(f"\n{n} morceaux indexes dans '{args.collection}'")

    trace = {
        "collection": args.collection,
        "modele_embedding": args.modele,
        "dimension": dim,
        "index": "FLAT",
        "metrique": "COSINE",
        "n_enfants": len(enfants),
        "n_parents": doc["n_parents"],
        "source": doc["document"],
    }
    Path("index_rfp_config.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("Trace ecrite dans index_rfp_config.json")


if __name__ == "__main__":
    main()
