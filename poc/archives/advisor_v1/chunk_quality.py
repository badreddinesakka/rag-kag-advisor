"""
chunk_quality.py — Étage 4 : la NOTATION des découpages.

Note un découpage SANS construire d'index et SANS poser de questions.
On regarde les morceaux produits et on juge s'ils sont bien faits.

Cinq mesures, dans l'esprit de :
    de Moura Júnior, Lelong, Blangero — « Adaptive Chunking: Optimizing
    Chunking-Method Selection for RAG », LREC 2026, p. 11535-11551.

  1. taille        — les morceaux sont-ils dans la fourchette visée ?
  2. intégrité     — se terminent-ils proprement, ou en plein milieu ?
  3. références    — combien commencent par « cette procédure », « il »,
                     « celui-ci », sans qu'on sache de quoi il s'agit ?
  4. cohésion      — les phrases d'un morceau parlent-elles du même sujet ?
  5. cohérence     — le morceau reste-t-il dans le sujet de son document ?

Les trois premières sont instantanées (aucune dépendance).
Les deux dernières demandent des embeddings (bge-m3 via Ollama) : elles sont
désactivables par --no-embed.

HONNÊTETÉ SUR CE QUE ÇA MESURE
------------------------------
Ces notes jugent la QUALITÉ DU DÉCOUPAGE, pas la qualité des réponses. Un
découpage bien noté est un découpage propre ; il reste à vérifier qu'il fait
gagner en récupération. C'est complémentaire d'une mesure de rappel, pas un
remplacement. À écrire tel quel dans le rapport.

Utilisation :
    python chunk_quality.py --input ooredoo --strategy all
    python chunk_quality.py --input ooredoo --from-router profil.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import chunker
from chunker import Chunk, chunk_corpus, split_sentences, embed_texts, _cosine

# --------------------------------------------------------------------------
# Réglages
# --------------------------------------------------------------------------

SIZE_LOW = 0.5      # un morceau vaut au moins la moitié de la taille visée
SIZE_HIGH = 1.5     # et au plus une fois et demie
EMBED_SAMPLE = 150  # nombre de morceaux échantillonnés pour les mesures lentes

WEIGHTS = {
    "taille": 0.15,
    "integrite": 0.25,
    "references": 0.25,
    "cohesion": 0.20,
    "coherence": 0.15,
}

# Écart en dessous duquel deux découpages sont déclarés équivalents.
# Sur ~100 morceaux, un écart plus petit n'est pas distinguable du bruit.
TIE_MARGIN = 0.03

# Mots qui, en tête de morceau, renvoient à quelque chose qu'on n'a pas.
_DANGLING = re.compile(
    r"^(?:"
    r"ce|cet|cette|ces|celui|celle|ceux|celles|"
    r"il|elle|ils|elles|leur|leurs|lui|"
    r"y|en|dont|lequel|laquelle|lesquels|"
    r"ci-dessus|ci-dessous|ledit|ladite|"
    r"this|that|these|those|it|they|them|their|its|"
    r"such|said|above|below|the former|the latter"
    r")\b",
    re.IGNORECASE,
)

# Un morceau bien terminé finit sur une ponctuation forte, un deux-points,
# ou une fin de ligne de liste.
_CLEAN_END = re.compile(r"[.!?…:;)\]»\"']\s*$")


# --------------------------------------------------------------------------
# Mesures rapides
# --------------------------------------------------------------------------

def score_size(chunks: list[Chunk], target: int) -> float:
    """Part des morceaux dont la taille est dans la fourchette visée."""
    if not chunks:
        return 0.0
    low, high = target * SIZE_LOW, target * SIZE_HIGH
    ok = sum(1 for c in chunks if low <= len(c.body) <= high)
    return ok / len(chunks)


def score_integrity(chunks: list[Chunk]) -> float:
    """Part des morceaux qui se terminent proprement (phrase entière)."""
    if not chunks:
        return 0.0
    ok = 0
    for c in chunks:
        body = c.body.rstrip()
        if c.is_table:            # un tableau entier est intègre par construction
            ok += 1
            continue
        if not body:
            continue
        if _CLEAN_END.search(body):
            ok += 1
        elif body[-1].isdigit() and len(body.split()) > 3:
            # une ligne de tableau ou une valeur chiffrée : acceptable
            ok += 1
    return ok / len(chunks)


def score_references(chunks: list[Chunk]) -> float:
    """Part des morceaux qui ne commencent PAS par une référence sans référent."""
    if not chunks:
        return 0.0
    ok = 0
    for c in chunks:
        body = c.body.lstrip()
        if c.is_table or not body:
            ok += 1
            continue
        ok += 0 if _DANGLING.match(body) else 1
    return ok / len(chunks)


# --------------------------------------------------------------------------
# Mesures lentes (embeddings)
# --------------------------------------------------------------------------

def _sample(chunks: list[Chunk], n: int) -> list[Chunk]:
    if len(chunks) <= n:
        return chunks
    step = len(chunks) / n
    return [chunks[int(i * step)] for i in range(n)]


def score_cohesion(chunks: list[Chunk], model: str) -> float | None:
    """Ressemblance moyenne entre les phrases d'un même morceau."""
    sample = [c for c in _sample(chunks, EMBED_SAMPLE) if not c.is_table]
    scores: list[float] = []
    for c in sample:
        sents = split_sentences(c.body)
        if len(sents) < 2:
            continue
        sents = sents[:12]        # borne le coût sur les gros morceaux
        try:
            vecs = embed_texts(sents, model=model)
        except Exception:
            return None
        sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        if sims:
            scores.append(sum(sims) / len(sims))
    return sum(scores) / len(scores) if scores else None


def score_coherence(chunks: list[Chunk], model: str) -> float | None:
    """Ressemblance de chaque morceau au sujet moyen de son document."""
    sample = _sample(chunks, EMBED_SAMPLE)
    by_doc: dict[str, list[Chunk]] = {}
    for c in sample:
        by_doc.setdefault(c.doc_id, []).append(c)

    scores: list[float] = []
    for doc_id, group in by_doc.items():
        if len(group) < 2:
            continue
        try:
            vecs = embed_texts([c.body[:2000] for c in group], model=model)
        except Exception:
            return None
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        scores.extend(_cosine(v, centroid) for v in vecs)
    return sum(scores) / len(scores) if scores else None


# --------------------------------------------------------------------------
# Note globale
# --------------------------------------------------------------------------

def evaluate(chunks: list[Chunk], target: int, model: str = chunker.EMBED_MODEL,
             use_embeddings: bool = True) -> dict:
    """Note un découpage. Renvoie les cinq mesures et la note pondérée."""
    scores: dict[str, float | None] = {
        "taille": score_size(chunks, target),
        "integrite": score_integrity(chunks),
        "references": score_references(chunks),
        "cohesion": None,
        "coherence": None,
    }

    if use_embeddings and chunks:
        scores["cohesion"] = score_cohesion(chunks, model)
        scores["coherence"] = score_coherence(chunks, model)

    # Note pondérée sur les seules mesures disponibles : une mesure absente
    # ne doit pas être comptée comme un zéro.
    available = {k: v for k, v in scores.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in available) or 1.0
    overall = sum(WEIGHTS[k] * v for k, v in available.items()) / total_w

    sizes = sorted(len(c.body) for c in chunks) or [0]
    return {
        "scores": {k: (round(v, 3) if v is not None else None)
                   for k, v in scores.items()},
        "note": round(overall, 3),
        "mesures_utilisees": sorted(available),
        "n_chunks": len(chunks),
        "n_docs": len({c.doc_id for c in chunks}),
        "taille_mediane": sizes[len(sizes) // 2],
        "taille_visee": target,
    }


def compare_candidates(input_dir: str | Path, candidates: list[dict],
                       model: str = chunker.EMBED_MODEL,
                       use_embeddings: bool = True,
                       verbose: bool = True) -> dict:
    """
    Construit chaque découpage candidat, le note, et désigne le gagnant.

    candidates : la liste produite par router.chunking_candidates()
                 -> [{"name", "strategy", "target_chars", "why"}, ...]
    """
    results = []
    for cand in candidates:
        if verbose:
            print(f"\n=== {cand['name']} "
                  f"({cand['strategy']}, {cand['target_chars']} car.) ===")
        t0 = time.time()
        try:
            chunks = chunk_corpus(
                input_dir,
                strategy=cand["strategy"],
                target_size=cand["target_chars"],
                embed_model=model,
                verbose=verbose,
            )
        except Exception as exc:
            results.append({**cand, "error": str(exc), "note": 0.0})
            if verbose:
                print(f"  ÉCHEC : {exc}")
            continue

        if not chunks:
            msg = ("aucun morceau produit — pour la stratégie « semantic », "
                   "c'est presque toujours Ollama injoignable ou le modèle "
                   "d'embedding absent (ollama pull bge-m3)")
            results.append({**cand, "error": msg, "note": None})
            if verbose:
                print(f"  NON MESURÉ : {msg}")
            continue

        report = evaluate(chunks, cand["target_chars"], model, use_embeddings)
        report.update({
            "name": cand["name"],
            "strategy": cand["strategy"],
            "target_chars": cand["target_chars"],
            "why": cand.get("why", ""),
            "duree_s": round(time.time() - t0, 1),
        })
        results.append(report)
        if verbose:
            print(f"  note {report['note']:.3f} · "
                  f"{report['n_chunks']} morceaux · {report['duree_s']} s")

    echecs = [r for r in results if r.get("error")]
    valides = [r for r in results if not r.get("error")]
    if not valides:
        return {"resultats": results, "gagnant": None,
                "verdict": "Aucun découpage n'a pu être construit."}

    valides.sort(key=lambda r: -r["note"])
    best = valides[0]

    # Règle d'égalité : un écart minuscule sur quelques dizaines de morceaux
    # n'est pas un écart. On préfère alors le découpage le plus simple.
    proches = [r for r in valides if best["note"] - r["note"] <= TIE_MARGIN]
    ordre_simplicite = {"fixed": 0, "structural": 1, "semantic": 2}
    if len(proches) > 1:
        proches.sort(key=lambda r: (ordre_simplicite.get(r["strategy"], 9),
                                    r["duree_s"]))
        gagnant = proches[0]
        verdict = (
            f"{len(proches)} découpages se tiennent en moins de {TIE_MARGIN} de "
            f"note : l'écart n'est pas distinguable du bruit. On retient « "
            f"{gagnant['name']} », le plus simple et le plus rapide des ex aequo."
        )
    else:
        gagnant = best
        second = valides[1] if len(valides) > 1 else None
        ecart = f" (+{gagnant['note'] - second['note']:.3f} sur « {second['name']} »)" \
            if second else ""
        verdict = f"« {gagnant['name'] } » l'emporte nettement{ecart}."

    return {
        "resultats": valides,
        "non_mesures": [{"name": r["name"], "raison": r["error"]} for r in echecs],
        "gagnant": {
            "name": gagnant["name"],
            "strategy": gagnant["strategy"],
            "target_chars": gagnant["target_chars"],
            "note": gagnant["note"],
            "n_chunks": gagnant["n_chunks"],
        },
        "verdict": verdict,
        "embeddings_utilises": use_embeddings,
    }


# --------------------------------------------------------------------------
# Affichage et ligne de commande
# --------------------------------------------------------------------------


def _load_json(path: str):
    """
    Lit un JSON quel que soit son encodage.

    PowerShell écrit « > fichier.json » en UTF-16 ; cmd.exe en ANSI ; Python en
    UTF-8. On renifle donc la marque d'ordre des octets au lieu d'imposer un
    encodage, sinon la commande marche chez l'un et pas chez l'autre.
    """
    raw = open(path, "rb").read()
    for bom, enc in ((b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"),
                     (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return json.loads(raw.decode(enc))
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252", errors="replace"))


_LABELS = {
    "taille": "taille respectée",
    "integrite": "fins de morceau propres",
    "references": "sans référence orpheline",
    "cohesion": "cohésion interne",
    "coherence": "cohérence documentaire",
}


def print_table(comparison: dict) -> None:
    rows = comparison["resultats"]
    if not rows:
        print("Aucun résultat.")
        return

    print("\n" + "=" * 78)
    print(f"{'découpage':<18}{'note':>7}{'morceaux':>10}{'médiane':>9}{'durée':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:<18}{r['note']:>7.3f}{r['n_chunks']:>10}"
              f"{r['taille_mediane']:>9}{r['duree_s']:>7.1f}s")

    print("\ndétail des mesures (1 = parfait) :")
    keys = [k for k in WEIGHTS if any(r["scores"].get(k) is not None for r in rows)]
    print(f"{'':<18}" + "".join(f"{_LABELS[k][:13]:>15}" for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r["scores"].get(k)
            cells.append(f"{v:>15.3f}" if v is not None else f"{'—':>15}")
        print(f"{r['name']:<18}" + "".join(cells))

    for r in comparison.get("non_mesures", []):
        print(f"\n  NON MESURÉ — {r['name']} : {r['raison']}")

    print("\n" + comparison["verdict"])
    g = comparison.get("gagnant")
    if g:
        print(f"Retenu : {g['strategy']} à {g['target_chars']} caractères "
              f"→ {g['n_chunks']} morceaux.")
    if not comparison.get("embeddings_utilises"):
        print("(cohésion et cohérence non mesurées : mode --no-embed)")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Note les découpages d'un corpus, sans index ni questions."
    )
    ap.add_argument("--input", required=True, help="dossier contenant les documents")
    ap.add_argument("--strategy", default="all",
                    choices=["fixed", "structural", "semantic", "all"],
                    help="ignoré si --from-router est utilisé")
    ap.add_argument("--size", type=int, default=chunker.DEFAULT_TARGET)
    ap.add_argument("--from-router", metavar="PROFIL_JSON",
                    help="JSON du profiler : les candidats viennent alors du routeur")
    ap.add_argument("--mutability", default="figé", choices=["figé", "vivant"])
    ap.add_argument("--no-embed", action="store_true",
                    help="saute cohésion et cohérence (instantané, sans Ollama)")
    ap.add_argument("--embed-model", default=chunker.EMBED_MODEL)
    ap.add_argument("--out", help="fichier JSON de sortie")
    args = ap.parse_args()

    if args.from_router:
        from router import decide
        data = _load_json(args.from_router)
        corpus = data.get("corpus", data)
        decision = decide(corpus, mutability=args.mutability, probe=None)
        chunking = decision["chunking"]
        candidates = chunking["candidates"]
        print(f"Architecture : {decision['architecture']} · "
              f"référence {chunking['base_chars']} caractères")
        for note in chunking["notes"]:
            print(f"  ! {note}")
    elif args.strategy == "all":
        candidates = [
            {"name": "fixe", "strategy": "fixed", "target_chars": args.size},
            {"name": "structurel", "strategy": "structural", "target_chars": args.size},
            {"name": "sémantique", "strategy": "semantic", "target_chars": args.size},
        ]
    else:
        candidates = [{"name": args.strategy, "strategy": args.strategy,
                       "target_chars": args.size}]

    comparison = compare_candidates(
        args.input, candidates,
        model=args.embed_model,
        use_embeddings=not args.no_embed,
    )
    print_table(comparison)

    if args.out:
        Path(args.out).write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nÉcrit : {args.out}")


if __name__ == "__main__":
    main()
