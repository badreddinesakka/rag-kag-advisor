# -*- coding: utf-8 -*-
"""
probe.py — Étage 1 bis : le SONDEUR.

Un LLM local lit un échantillon du corpus et en extrait les vraies relations.
On mesure ce qu'il trouve, au lieu de le deviner en comptant des majuscules.

Méthode des entités partagées, en deux temps :
  1. le LLM lit un échantillon et fournit le VOCABULAIRE d'entités réelles ;
  2. chaque entité est ensuite recherchée dans le TEXTE COMPLET de tous les
     documents, par simple recherche de mots (gratuit, aucun appel LLM).

On combine ainsi l'intelligence du LLM (savoir ce qu'est une entité) et la
couverture totale du corpus (savoir dans combien de documents elle apparaît).
Chercher les entités uniquement dans les morceaux lus décrirait la finesse de
l'échantillonnage plutôt que le corpus.

Garde-fous d'extraction :
- une relation n'est retenue que si son sujet et son objet se retrouvent
  réellement dans le morceau de texte analysé ;
- les nombres, pourcentages et montants ne sont pas des entités
  (« AT&T → augmente sa participation à → 60 % » n'est pas une relation) ;
- un sujet ou un objet de plus de 6 mots est refusé : une phrase entière n'est
  pas une entité et ne se répète jamais.

Dépendance : Ollama qui tourne en local.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter

from profiler import _strip_accents, extract_text

# --- réglages ----------------------------------------------------------------
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# AUCUN NOM DE MODÈLE PAR DÉFAUT.
# Un nom de modèle décrit une installation, pas un corpus. Écrit en dur,
# il devient faux dès que l'outil change de machine — et il a la forme
# d'une recommandation alors que rien ne l'a mesuré. Le modèle est donc
# un argument OBLIGATOIRE : l'utilisateur donne ce dont il dispose.
N_CHUNKS        = 20
CHUNKS_PER_DOC  = 2
CHUNK_CHARS     = 2_000
REQUEST_TIMEOUT = 120
TOKENS_PER_WORD = 1.3
MATCH_RATIO     = 0.60    # part des mots à retrouver pour valider un sujet/objet
MAX_ARG_WORDS   = 6       # au-delà, ce n'est plus une entité mais une phrase
MIN_ARG_CHARS   = 3

_ENTITY_STOP = {"le", "la", "les", "de", "du", "des", "un", "une", "au", "aux",
                "the", "of", "and", "for", "a", "an"}

_PRED_STOP = {"est", "sont", "etait", "etaient", "etre", "a", "ont", "avait",
              "the", "is", "are", "was", "were", "has", "have", "be",
              "de", "du", "des", "d", "of", "le", "la", "les", "en", "dans",
              "par", "pour", "au", "aux", "to", "in", "on", "by", "with"}

# une valeur faite uniquement de chiffres et de symboles n'est pas une entité
_NUMERIC_ONLY = re.compile(r"^[\d\s.,;:%$€£¥+\-–—/()]*$")

PROMPT = """Tu es un extracteur de relations. Lis le TEXTE et liste uniquement \
les relations qui y sont explicitement écrites.

Règles strictes :
- Le sujet et l'objet doivent être des NOMS présents dans le texte, recopiés \
tels quels : une personne, une entreprise, un organisme, un lieu, une technologie.
- Le sujet et l'objet font au maximum 6 mots. Jamais une phrase.
- Le sujet et l'objet ne sont JAMAIS un nombre, un pourcentage ou un montant.
- N'invente rien, ne déduis rien, ne traduis rien.
- Une énumération n'est PAS une relation : deux mots écrits côte à côte dans une \
liste ou un tableau ne sont pas liés.
- Utilise des noms de relation courts et réutilisables (par exemple « dirige », \
« détient », « est filiale de », « est concurrent de ») plutôt qu'une phrase.
- Respecte le sens : « X dirige Y » signifie que X est le dirigeant de Y.
- Si le texte ne contient aucune relation claire, renvoie une liste vide.

Réponds UNIQUEMENT avec du JSON, sans commentaire, au format :
{"relations": [{"sujet": "...", "relation": "...", "objet": "..."}]}

TEXTE :
\"\"\"
%s
\"\"\"
"""


# --- appel au LLM local ------------------------------------------------------
def _call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def ollama_available(model: str) -> tuple[bool, str]:
    try:
        _call_ollama('Réponds {"relations": []}', model)
        return True, ""
    except urllib.error.URLError as e:
        return False, f"Ollama injoignable sur {OLLAMA_URL} ({e.reason})."
    except Exception as e:
        return False, f"Ollama a répondu une erreur : {e}"


# --- choix des morceaux ------------------------------------------------------
def _pick_chunks(texts, n_chunks, per_doc):
    usable = [(i, t) for i, (_, t) in enumerate(texts) if len(t.strip()) > 200]
    if not usable:
        return []

    # On lit `per_doc` morceaux par document. Le budget total de morceaux fixe
    # donc combien de documents on peut couvrir.
    n_docs = max(1, n_chunks // max(per_doc, 1))

    # Si le corpus a plus de documents que ce budget, on ne peut pas tous les
    # lire. On choisit alors `n_docs` documents RÉPARTIS sur tout le corpus
    # (indices régulièrement espacés) au lieu des premiers de la liste — sinon la
    # fin d'un gros corpus n'est jamais regardée.
    if len(usable) > n_docs:
        step = len(usable) / n_docs
        selected = [usable[int(k * step)] for k in range(n_docs)]
    else:
        selected = usable

    fractions = [0.20, 0.55, 0.80, 0.35, 0.70]
    chunks, seen = [], set()
    for round_no in range(per_doc):          # `per_doc` morceaux par document…
        for doc_i, text in selected:         # …pris à des endroits différents.
            if len(chunks) >= n_chunks:
                break
            frac = fractions[round_no % len(fractions)]
            start = max(0, min(int(len(text) * frac), max(0, len(text) - CHUNK_CHARS)))
            piece = text[start:start + CHUNK_CHARS].strip()
            key = piece[:200]
            if len(piece) > 200 and key not in seen:
                seen.add(key)
                chunks.append((doc_i, piece))
        if len(chunks) >= n_chunks:
            break
    return chunks[:n_chunks]


# --- validation d'un sujet / objet ------------------------------------------
def _is_valid_arg(value: str) -> bool:
    """Un sujet ou un objet doit être un nom court, pas un nombre ni une phrase."""
    v = (value or "").strip()
    if len(v) < MIN_ARG_CHARS:
        return False
    if len(v.split()) > MAX_ARG_WORDS:
        return False
    if _NUMERIC_ONLY.match(v):
        return False
    if sum(ch.isalpha() for ch in v) < 2:
        return False
    return True


def _verify(text_norm: str, value: str) -> bool:
    """Le sujet/objet est-il réellement présent dans le morceau lu ?"""
    v = _strip_accents(value.strip().lower())
    if v in text_norm:
        return True
    words = [w for w in re.split(r"\W+", v) if len(w) >= 3 and w not in _ENTITY_STOP]
    if not words:
        return False
    return sum(1 for w in words if w in text_norm) / len(words) >= MATCH_RATIO


def _norm_predicate(pred: str) -> str:
    p = _strip_accents((pred or "").lower().strip())
    p = re.sub(r"[^a-z0-9 ]", " ", p)
    toks = [t for t in p.split() if t and t not in _PRED_STOP]
    return " ".join(toks) if toks else p.strip()


def _norm_entity(value: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(value.strip().lower()))


def _parse_relations(raw: str) -> list[dict]:
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    rels = obj.get("relations") if isinstance(obj, dict) else obj
    if not isinstance(rels, list):
        return []
    return [
        {"sujet": str(r.get("sujet", "")), "relation": str(r.get("relation", "")),
         "objet": str(r.get("objet", ""))}
        for r in rels if isinstance(r, dict)
    ]


# --- où chaque entité apparaît-elle, dans TOUT le corpus ? ------------------
def _entity_document_spread(entities: set[str],
                            full_texts: list[str]) -> dict[str, int]:
    """
    Pour chaque entité trouvée par le LLM, compte dans combien de documents
    ENTIERS elle apparaît. Recherche par mots entiers, sans appel LLM.
    """
    normed_docs = [_strip_accents(t.lower()) for t in full_texts]
    spread = {}
    for ent in entities:
        pattern = re.compile(r"(?<!\w)" + re.escape(ent) + r"(?!\w)")
        spread[ent] = sum(1 for doc in normed_docs if pattern.search(doc))
    return spread


# --- sondage complet ---------------------------------------------------------
def probe_corpus(files, model, n_chunks=N_CHUNKS,
                 per_doc=CHUNKS_PER_DOC, progress=None) -> dict:
    texts = []
    for name, data in files:
        text, _, _ = extract_text(name, data)
        texts.append((name, text or ""))

    chunks = _pick_chunks(texts, n_chunks, per_doc)
    if not chunks:
        return {"available": False, "error": "Pas assez de texte exploitable pour un sondage."}

    ok, err = ollama_available(model)
    if not ok:
        return {"available": False, "error": err}

    kept: list[dict] = []
    unverified = 0
    predicates = Counter()
    entities: set[str] = set()
    sampled_words = 0
    chunks_per_doc = Counter(doc_i for doc_i, _ in chunks)

    for k, (doc_i, piece) in enumerate(chunks):
        if progress:
            progress(k, len(chunks))
        sampled_words += len(piece.split())
        piece_norm = _strip_accents(piece.lower())

        try:
            raw = _call_ollama(PROMPT % piece, model)
        except Exception:
            continue

        for rel in _parse_relations(raw):
            sujet, predicat, objet = rel["sujet"], rel["relation"], rel["objet"]
            if not predicat.strip():
                unverified += 1
                continue
            if not (_is_valid_arg(sujet) and _is_valid_arg(objet)):
                unverified += 1
                continue
            if _norm_entity(sujet) == _norm_entity(objet):
                unverified += 1
                continue
            if not (_verify(piece_norm, sujet) and _verify(piece_norm, objet)):
                unverified += 1
                continue

            norm_pred = _norm_predicate(predicat)
            kept.append({**rel, "doc": doc_i, "_pred": norm_pred})
            predicates[norm_pred] += 1
            for side in (sujet, objet):
                entities.add(_norm_entity(side))

    if progress:
        progress(len(chunks), len(chunks))

    # --- entités partagées : recherche dans les documents ENTIERS -----------
    # le LLM a fourni le vocabulaire d'entités ; on compte dans combien de
    # documents complets chacune apparaît (recherche de mots, sans appel LLM).
    spread = _entity_document_spread(entities, [t for _, t in texts])

    distinct_entities = len(entities)
    shared = sum(1 for c in spread.values() if c >= 2)
    cross_doc = round(shared / distinct_entities, 3) if distinct_entities else 0.0

    proposed = len(kept) + unverified
    sampled_tokens = max(1, int(sampled_words * TOKENS_PER_WORD))
    reused = sum(c for c in predicates.values() if c >= 2)

    top_shared = sorted(spread.items(), key=lambda kv: -kv[1])[:20]

    return {
        "available": True,
        "model": model,
        "chunks_sampled": len(chunks),
        "docs_covered": len(chunks_per_doc),
        "docs_total": len(texts),
        "chunks_per_doc_avg": round(len(chunks) / max(len(chunks_per_doc), 1), 1),
        "sampled_tokens": sampled_tokens,

        "relations_kept": len(kept),
        "relations_unverified": unverified,
        "unverified_rate": round(unverified / proposed, 3) if proposed else 0.0,

        "relations_per_1000_tokens": round(1000 * len(kept) / sampled_tokens, 2),
        "distinct_entities": distinct_entities,
        "cross_doc_entity_share": cross_doc,
        "distinct_predicates": len(predicates),
        "relation_reuse": round(reused / len(kept), 3) if kept else 0.0,

        "top_shared_entities": [{"entite": e, "documents": c} for e, c in top_shared],
        "top_predicates": predicates.most_common(10),
        "sample_relations": [
            {k2: v2 for k2, v2 in r.items() if not k2.startswith("_")} for r in kept[:25]
        ],
    }


# --- ligne de commande -------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Sondage LLM du corpus (étage 1 bis).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True,
                    help="modèle Ollama à utiliser pour le sondage "
                         "(aucun défaut : il dépend de ton installation)")
    ap.add_argument("--chunks", type=int, default=N_CHUNKS)
    ap.add_argument("--per-doc", type=int, default=CHUNKS_PER_DOC)
    ap.add_argument("--runs", type=int, default=1,
                    help="Répète le sondage N fois pour vérifier la stabilité")
    args = ap.parse_args()

    paths = sorted(Path(args.input).iterdir())
    files = [(p.name, p.read_bytes()) for p in paths if p.is_file()]

    def show(done, total):
        print(f"  morceau {done}/{total}…   ", end="\r")

    results = []
    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n--- sondage {run + 1}/{args.runs} ---")
        r = probe_corpus(files, model=args.model, n_chunks=args.chunks,
                         per_doc=args.per_doc, progress=show)
        print()
        if not r.get("available"):
            raise SystemExit(r.get("error"))
        results.append(r)
        print(json.dumps(
            {k: v for k, v in r.items()
             if k not in ("sample_relations", "top_predicates", "top_shared_entities")},
            ensure_ascii=False, indent=2,
        ))

    last = results[-1]
    print("\n--- entités présentes dans le plus de documents ---")
    for row in last["top_shared_entities"]:
        print(f"  {row['documents']:>3} documents · {row['entite']}")

    print("\n--- exemples de relations trouvées ---")
    for rel in last["sample_relations"]:
        print(f"  {rel['sujet']} → {rel['relation']} → {rel['objet']}")

    if args.runs > 1:
        print("\n--- stabilité entre les sondages ---")
        for key in ("relations_per_1000_tokens", "cross_doc_entity_share", "relation_reuse"):
            vals = [r[key] for r in results]
            print(f"  {key:<28} {vals}   écart = {max(vals) - min(vals):.3f}")