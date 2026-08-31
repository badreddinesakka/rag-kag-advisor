# -*- coding: utf-8 -*-
"""
mesure_topk.py — fabrique un jeu de questions, puis mesure top_k avec.

Le blocage que ce fichier lève
------------------------------
top_k, retrieve_k et le reranker ne peuvent pas être mesurés sans questions.
Faute de questions, ils restaient « réglés » — c'est-à-dire posés à la main —
quel que soit le corpus.

Ici le LLM lit le corpus et écrit lui-même les questions. Chacune est VÉRIFIÉE :
la réponse doit se retrouver mot pour mot dans le texte source. Une question dont
la réponse est inventée est jetée.

Le piège évité
--------------
Les questions sont écrites à partir du TEXTE BRUT des documents, jamais des
morceaux. Si on les tirait des morceaux, chaque découpage serait jugé sur ses
propres morceaux et gagnerait chez lui — un biais circulaire.

Le morceau « correct » est déterminé après coup : c'est celui qui contient la
réponse. Les questions restent donc valables si le découpage change.

Ce que la mesure dit, et ne dit pas
-----------------------------------
On mesure si le morceau porteur de la réponse est REMONTÉ dans les k premiers.
On ne mesure pas si le LLM s'en sert bien ensuite. Un top_k élevé peut capturer
la réponse tout en noyant le modèle : cette mesure ne verra pas ce coût-là.

Usage :
    python mesure_topk.py --input ooredoo --config advisor_mesure.json
    python mesure_topk.py --input ooredoo --questions questions_auto.json
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

from decision import mesure, regle, consequence, resume

MODELE_LLM   = "qwen2.5:7b"
MODELE_EMB   = "bge-m3"
N_QUESTIONS  = 30
VALEURS_K    = [1, 3, 5, 10, 20]
FENETRE_CAR  = 1800    # extrait de document soumis au LLM pour écrire la question
GRAINE       = 12345


# --------------------------------------------------------------------------
# 1. fabriquer les questions
# --------------------------------------------------------------------------
PROMPT = """You are given an extract from a document.

Write ONE factual question that this extract answers, and give the answer as an
EXACT quote copied from the extract (10 words maximum).

Rules:
- the question must be answerable ONLY with this extract
- do not write a yes/no question
- the answer must be copied character for character from the extract
- reply with JSON only, no other text

{{"question": "...", "reponse": "..."}}

Extract:
{extrait}"""


def _ollama(prompt: str, modele: str) -> str:
    import urllib.request
    import os
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    corps = json.dumps({"model": modele, "prompt": prompt, "stream": False,
                        "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(url, data=corps,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]


def _normaliser(t: str) -> str:
    """Pour comparer deux textes sans se faire piéger par les espaces."""
    return re.sub(r"\s+", " ", t.lower()).strip()


def fabriquer_questions(dossier: str, n: int, modele: str) -> list[dict]:
    """
    Une question par extrait de document. Le texte source est le document BRUT.

    La vérification est le point important : si la réponse rendue par le LLM ne
    se retrouve pas dans l'extrait, la question est jetée. Sans ce garde-fou on
    mesurerait la capacité à retrouver des réponses inventées.
    """
    import profiler

    textes = []
    for f in sorted(Path(dossier).iterdir()):
        if f.is_dir():
            continue
        try:
            texte, _, _ = profiler.extract_text(f.name, f.read_bytes())
        except Exception:
            continue
        if texte and len(texte) > FENETRE_CAR:
            textes.append((f.name, texte))

    if not textes:
        raise RuntimeError(f"aucun document lisible dans {dossier}")

    # Extraits répartis sur tout le corpus, et pas seulement sur les premiers
    # documents : sinon les questions ne couvrent qu'une partie du corpus.
    extraits = []
    for nom, texte in textes:
        pas = max(FENETRE_CAR, len(texte) // 6)
        for debut in range(0, len(texte) - FENETRE_CAR, pas):
            extraits.append((nom, texte[debut:debut + FENETRE_CAR]))

    random.Random(GRAINE).shuffle(extraits)

    questions, rejets = [], 0
    for nom, extrait in extraits:
        if len(questions) >= n:
            break
        try:
            brut = _ollama(PROMPT.format(extrait=extrait), modele)
            bloc = re.search(r"\{.*\}", brut, re.S)
            if not bloc:
                rejets += 1
                continue
            obj = json.loads(bloc.group(0))
            q, rep = obj.get("question", "").strip(), obj.get("reponse", "").strip()
        except Exception:
            rejets += 1
            continue

        # Le garde-fou : la réponse doit exister dans le texte source.
        if not q or not rep or _normaliser(rep) not in _normaliser(extrait):
            rejets += 1
            continue

        questions.append({"question": q, "reponse": rep, "document": nom})
        print(f"  {len(questions):3d}/{n}  {q[:70]}")

    print(f"\n{len(questions)} questions retenues, {rejets} rejetées "
          f"(réponse absente du texte ou JSON illisible).")
    return questions


# --------------------------------------------------------------------------
# 2. mesurer top_k
# --------------------------------------------------------------------------
def _cosinus(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def mesurer(dossier: str, questions: list[dict], strategie: str, taille: int,
            modele_emb: str, valeurs_k: list[int]) -> dict:
    import chunker
    from index_rag import embed_texts

    morceaux = [c.text for c in chunker.chunk_corpus(
        dossier, strategy=strategie, target_size=taille, verbose=False)]
    if not morceaux:
        return {"ok": False, "erreur": "aucun morceau construit"}

    # Le morceau « correct » est celui qui CONTIENT la réponse. Il est trouvé
    # après le découpage, donc les questions restent valables si le découpage
    # change — c'est ce qui rend cette mesure réutilisable.
    norm = [_normaliser(m) for m in morceaux]
    utilisables = []
    for q in questions:
        cible = [i for i, m in enumerate(norm) if _normaliser(q["reponse"]) in m]
        if cible:
            utilisables.append((q, set(cible)))

    if not utilisables:
        return {"ok": False, "erreur": "aucune réponse retrouvée dans les morceaux "
                                       "— le découpage a peut-être coupé au mauvais endroit"}

    t0 = time.time()
    v_morceaux = embed_texts(morceaux, modele_emb)
    v_questions = embed_texts([q["question"] for q, _ in utilisables], modele_emb)
    duree = time.time() - t0

    rangs = []
    for vq, (_, cibles) in zip(v_questions, utilisables):
        scores = sorted(((_cosinus(vq, v), j) for j, v in enumerate(v_morceaux)),
                        key=lambda s: -s[0])
        rangs.append(next((r for r, (_, j) in enumerate(scores, 1) if j in cibles),
                          len(morceaux)))

    n = len(rangs)
    par_k = [{"k": k, "rappel": round(sum(1 for r in rangs if r <= k) / n, 3)}
             for k in valeurs_k if k <= len(morceaux)]

    return {"ok": True, "n_morceaux": len(morceaux), "n_questions": n,
            "questions_sans_cible": len(questions) - n,
            "rangs": rangs, "par_k": par_k, "secondes": round(duree, 1)}


def choix_topk(res: dict, gain_min: float = 0.02) -> list:
    """
    Le meilleur top_k n'est pas celui qui a le meilleur rappel.

    Le rappel ne peut que MONTER quand k augmente : prendre le maximum
    reviendrait à toujours choisir la plus grande valeur, ce qui n'est pas un
    choix. On cherche donc le point où ça SATURE : le plus petit k tel que
    passer au suivant rapporte moins que `gain_min`.

    Au-delà de ce point, on paie des passages en plus sans rien gagner — et on
    encombre le contexte du LLM.
    """
    par_k = res["par_k"]
    if len(par_k) < 2:
        return [regle("top_k", par_k[0]["k"] if par_k else 5,
                      "une seule valeur essayée : rien à comparer")]

    # On remonte depuis la FIN : on cherche le plus petit k après lequel plus
    # aucun palier ne rapporte `gain_min`. S'arrêter au premier palier plat
    # serait une erreur — le rappel peut stagner puis repartir, et on
    # couronnerait k=1 alors que k=10 fait bien mieux.
    retenu = par_k[-1]
    for i in range(len(par_k) - 2, -1, -1):
        if par_k[i + 1]["rappel"] - par_k[i]["rappel"] >= gain_min:
            break
        retenu = par_k[i]

    detail = " · ".join(f"k={p['k']} {p['rappel']:.0%}" for p in par_k)
    ch = mesure("top_k", [(str(p["k"]), p["rappel"]) for p in par_k], ecart_min=gain_min)
    ch.valeur = int(retenu["k"])   # entier : la valeur est consommée en aval
    ch.egalite = False
    ch.raison = ""
    ch.essais = {str(p["k"]): p["rappel"] for p in par_k}
    return [
        ch,
        consequence("retrieve_k", retenu["k"] * 2,
                    f"deux fois top_k : le reranker doit avoir de quoi trier. "
                    f"Rappels mesurés — {detail}"),
    ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Mesurer top_k avec des questions générées.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", help="advisor_mesure.json (découpage mesuré)")
    ap.add_argument("--questions", help="jeu de questions déjà généré (JSON)")
    ap.add_argument("--n", type=int, default=N_QUESTIONS)
    ap.add_argument("--llm", default=MODELE_LLM)
    ap.add_argument("--out-questions", default="questions_auto.json")
    ap.add_argument("--out", default="topk_mesure.json")
    args = ap.parse_args()

    strategie, taille, modele_emb = "fixed", 1382, MODELE_EMB
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        strategie  = cfg.get("_chunk_strategy", strategie)
        taille     = int(cfg.get("_chunk_chars", taille))
        modele_emb = cfg.get("_embedding_model", modele_emb)
    print(f"Découpage {strategie} à {taille} caractères · embedding {modele_emb}\n")

    if args.questions and Path(args.questions).exists():
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        print(f"{len(questions)} questions relues depuis {args.questions}.\n")
    else:
        print(f"Le LLM écrit les questions ({args.llm}) — comptez quelques minutes.\n")
        questions = fabriquer_questions(args.input, args.n, args.llm)
        Path(args.out_questions).write_text(
            json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Questions écrites dans {args.out_questions} — réutilisables avec "
              f"--questions.\n")

    if not questions:
        raise SystemExit("aucune question vérifiée : le LLM invente ses réponses, "
                         "ou le corpus est illisible.")

    res = mesurer(args.input, questions, strategie, taille, modele_emb, VALEURS_K)
    if not res["ok"]:
        raise SystemExit(res["erreur"])

    print(f"{res['n_questions']} questions utilisables sur {len(questions)} "
          f"({res['questions_sans_cible']} dont la réponse n'a pas été retrouvée "
          f"dans un morceau).\n")
    for p in res["par_k"]:
        print(f"  top_k = {p['k']:2d}   le bon passage est remonté dans "
              f"{p['rappel']:.0%} des cas")

    print()
    print(resume(choix_topk(res)))
    print("\nRappel : on mesure si le bon passage REMONTE, pas si le LLM s'en "
          "sert bien. Un top_k élevé peut capturer la réponse et noyer le "
          "modèle — ce coût-là n'est pas mesuré ici.")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nDétail écrit dans {args.out}.")