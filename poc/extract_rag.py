"""
extract_rag.py — Extraction des criteres du RFP par RAG.

Boucle : pour chaque requete
    1. la requete est transformee en vecteur (bge-m3)
    2. Milvus renvoie les `top_k` ENFANTS les plus proches
    3. on remonte a leurs PARENTS (sections completes) : l'enfant sert a trouver,
       le parent sert a comprendre
    4. les parents deja traites pour cette categorie sont retires
       -> si tout est deja vu, on ne rappelle PAS le LLM (economie et moins de doublons)
    5. le LLM lit les parents restants et repond en JSON

Sortie : criteres_extraits.json + un resume de COUVERTURE
(combien de sections du document la recherche a reellement atteintes).

Usage :
    python extract_rag.py
    python extract_rag.py --modele qwen3:8b --top-k 5 --categorie equipe
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

from pymilvus import MilvusClient

from prompts_rfp import CATEGORIES, construire_prompt

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")

STATUTS = {"obligatoire", "recommande", "non precise"}


# ----------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------

def embed(texte: str, modele: str) -> list:
    corps = json.dumps({"model": modele, "prompt": texte}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings", data=corps,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["embedding"]


def generer(prompt: str, modele: str, timeout: int) -> str:
    corps = json.dumps({
        "model": modele,
        "prompt": prompt,
        "stream": False,
        "format": "json",          # Ollama contraint la sortie a du JSON valide
        "options": {
            "temperature": 0,      # extraction : aucune creativite souhaitee
            "num_predict": 2000,
            "num_ctx": 8192,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=corps,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "")


# ----------------------------------------------------------------------
# Nettoyage des reponses
# ----------------------------------------------------------------------

def normaliser_statut(valeur: str) -> str:
    v = (valeur or "").strip().lower()
    v = unicodedata.normalize("NFD", v)
    v = "".join(c for c in v if unicodedata.category(c) != "Mn")
    if v in STATUTS:
        return v
    if any(m in v for m in ("must", "shall", "mandatory", "required", "oblig")):
        return "obligatoire"
    if any(m in v for m in ("should", "recommend", "prefer", "recomm")):
        return "recommande"
    return "non precise"


def cle_doublon(texte: str) -> str:
    """Cle de comparaison : minuscules, sans accents, sans ponctuation."""
    t = unicodedata.normalize("NFD", (texte or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", t).split().__str__()


def lire_reponse(brut: str, categorie: str, sections_valides: set):
    """Transforme la reponse du LLM en liste de criteres propres."""
    try:
        data = json.loads(brut)
    except json.JSONDecodeError:
        return [], 1

    lignes = data.get("criteres") if isinstance(data, dict) else data
    if not isinstance(lignes, list):
        return [], 1

    propres, rejets = [], 0
    for l in lignes:
        if not isinstance(l, dict):
            rejets += 1
            continue
        critere = str(l.get("critere") or l.get("criture") or "").strip()
        if len(critere) < 8:            # ligne vide ou inutilisable
            rejets += 1
            continue
        section = str(l.get("section") or "").strip()
        if section not in sections_valides:   # section inventee -> on ne la garde pas
            section = ""
        propres.append({
            "categorie": categorie,           # impose : le modele ne choisit pas
            "critere": critere,
            "valeur": str(l.get("valeur") or "").strip(),
            "statut": normaliser_statut(str(l.get("statut") or "")),
            "section": section,
        })
    return propres, rejets


# ----------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extraction des criteres par RAG")
    ap.add_argument("--chunks", default="chunks_rfp.json")
    ap.add_argument("--collection", default="rfp_criteres")
    ap.add_argument("--modele", default="qwen3:8b", help="modele de generation")
    ap.add_argument("--modele-embedding", default="bge-m3")
    ap.add_argument("--top-k", type=int, default=5, help="enfants remontes par requete")
    ap.add_argument("--max-parents", type=int, default=3,
                    help="parents envoyes au LLM par appel")
    ap.add_argument("--categorie", default=None, help="ne traiter qu'une categorie")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--sortie", default="criteres_extraits.json")
    args = ap.parse_args()

    chemin = Path(args.chunks)
    if not chemin.exists():
        sys.exit(f"{chemin} introuvable. Lance d'abord chunk_rfp.py")

    doc = json.loads(chemin.read_text(encoding="utf-8"))
    parents = {p["parent_id"]: p for p in doc["parents"]}
    sections_valides = {p["section"] for p in doc["parents"]} | {""}

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(args.collection):
        sys.exit(f"Collection '{args.collection}' absente. Lance d'abord index_rfp.py")
    client.load_collection(args.collection)

    cats = [args.categorie] if args.categorie else list(CATEGORIES)
    for c in cats:
        if c not in CATEGORIES:
            sys.exit(f"Categorie inconnue : {c}")

    criteres, journal = [], []
    vus_global = set()
    n_appels = n_ignores = n_illisibles = 0
    t0 = time.time()

    for cat in cats:
        requetes = CATEGORIES[cat]["requetes"]
        vus_cat = set()
        print(f"\n=== {cat} : {len(requetes)} requetes ===")

        for i, req in enumerate(requetes, start=1):
            vec = embed(req, args.modele_embedding)
            hits = client.search(
                args.collection, data=[vec], limit=args.top_k,
                output_fields=["parent_id", "section"],
            )[0]

            # enfant -> parent, en gardant l'ordre de pertinence
            ordre = []
            for h in hits:
                pid = h["entity"]["parent_id"]
                if pid not in ordre:
                    ordre.append(pid)

            nouveaux = [p for p in ordre if p not in vus_cat][: args.max_parents]
            vus_cat.update(nouveaux)
            vus_global.update(ordre)

            if not nouveaux:
                n_ignores += 1
                print(f"  [{i:>2}/{len(requetes)}] deja vu -> pas d'appel LLM  | {req[:45]}")
                continue

            # texte envoye au LLM : la section entiere, tableaux compris
            morceaux = []
            for pid in nouveaux:
                p = parents[pid]
                bloc = f"[{p['section']} {p['titre']}]\n{p['texte']}"
                for t in p["tableaux"]:
                    bloc += f"\n{t}"
                morceaux.append(bloc)
            passages = "\n\n---\n\n".join(morceaux)

            prompt = construire_prompt(cat, passages)
            depart = time.time()
            try:
                brut = generer(prompt, args.modele, args.timeout)
            except Exception as e:
                print(f"  [{i:>2}/{len(requetes)}] ECHEC LLM : {e}")
                n_illisibles += 1
                continue
            duree = time.time() - depart
            n_appels += 1

            lignes, rejets = lire_reponse(brut, cat, sections_valides)
            n_illisibles += rejets
            criteres.extend(lignes)

            sections = ",".join(sorted({parents[p]["section"] for p in nouveaux}))
            print(f"  [{i:>2}/{len(requetes)}] {len(lignes):>2} criteres "
                  f"| sections {sections:<14} | {duree:>5.0f}s | {req[:40]}")

            journal.append({
                "categorie": cat, "requete": req,
                "parents": nouveaux, "n_criteres": len(lignes),
                "secondes": round(duree, 1),
            })

    # --- dedoublonnage ---
    uniques, deja = [], set()
    for c in criteres:
        k = (c["categorie"], cle_doublon(c["critere"]))
        if k in deja:
            continue
        deja.add(k)
        uniques.append(c)

    for i, c in enumerate(uniques, start=1):
        c["id"] = f"C{i:03d}"

    couverture = len(vus_global) / len(parents) if parents else 0

    sortie = {
        "document": doc["document"],
        "modele": args.modele,
        "modele_embedding": args.modele_embedding,
        "top_k": args.top_k,
        "max_parents": args.max_parents,
        "n_criteres": len(uniques),
        "n_bruts": len(criteres),
        "taux_doublons": round(1 - len(uniques) / len(criteres), 3) if criteres else 0,
        "couverture_parents": round(couverture, 3),
        "parents_atteints": len(vus_global),
        "parents_total": len(parents),
        "parents_manques": sorted(
            f"{parents[p]['section']} {parents[p]['titre']}"
            for p in parents if p not in vus_global
        ),
        "n_appels_llm": n_appels,
        "n_requetes_sans_appel": n_ignores,
        "n_lignes_illisibles": n_illisibles,
        "minutes": round((time.time() - t0) / 60, 1),
        "criteres": uniques,
        "journal": journal,
    }
    Path(args.sortie).write_text(
        json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print(f"{len(uniques)} criteres uniques ({len(criteres)} bruts, "
          f"{sortie['taux_doublons']*100:.0f} % de doublons)")
    print(f"Couverture : {len(vus_global)}/{len(parents)} sections "
          f"({couverture*100:.0f} %)")
    if sortie["parents_manques"]:
        print("Sections JAMAIS atteintes par la recherche :")
        for s in sortie["parents_manques"]:
            print(f"   - {s}")
        print("  -> ce sont des criteres perdus d'avance : ajoute des requetes "
              "dans prompts_rfp.py pour les viser.")
    par_cat = {}
    for c in uniques:
        par_cat[c["categorie"]] = par_cat.get(c["categorie"], 0) + 1
    print("Par categorie :", par_cat)
    print(f"{n_appels} appels LLM, {n_ignores} requetes sans appel, "
          f"{sortie['minutes']} min")
    print(f"Ecrit dans {args.sortie}")


if __name__ == "__main__":
    main()
