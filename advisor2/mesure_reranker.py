# -*- coding: utf-8 -*-
"""
mesure_reranker.py — le reranker apporte-t-il quelque chose, et à quel prix ?

Ce que fait un reranker
-----------------------
Il ne récupère RIEN. Il reçoit les candidats déjà trouvés par la recherche
vectorielle et les remet dans un autre ordre, en lisant vraiment la question et
le passage ensemble — là où l'embedding les compare de loin.

Donc :
  · le rappel à retrieve_k ne peut PAS changer (mêmes candidats)
  · le rappel à top_k peut monter (meilleur ordre avant de couper)

C'est exactement ce qu'on mesure : on récupère retrieve_k candidats, on les
réordonne, on coupe à top_k, et on compare avec la coupe directe.

Le coût
-------
Le reranker tourne à CHAQUE question, pas une fois à l'indexation. Un gain de
2 points qui double le temps de réponse n'en vaut pas la peine. Le temps par
question est donc mesuré et affiché à côté du gain.

Usage :
    python mesure_reranker.py --input ooredoo --config advisor_mesure.json \
                              --questions questions_auto.json
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

from decision import mesure, regle, contraint, resume

# AUCUN NOM DE MODÈLE PAR DÉFAUT.
# Un nom de modèle décrit une installation, pas un corpus. Écrit en dur,
# il devient faux dès que l'outil change de machine — et il a la forme
# d'une recommandation alors que rien ne l'a mesuré. Le modèle est donc
# un argument OBLIGATOIRE : l'utilisateur donne ce dont il dispose.
RETRIEVE_K      = 20
COUPES          = [3, 5, 10]
GRAINE          = 12345


def _normaliser(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def _cosinus(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def charger_reranker(nom: str):
    """
    Renvoie (fonction_de_score, message). La fonction vaut None si le reranker
    n'est pas utilisable — ce qui est en soi un résultat : la configuration
    recommande alors quelque chose que la machine ne peut pas exécuter.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None, ("sentence-transformers n'est pas installé. Le reranker "
                      "recommandé par la config ne peut donc pas tourner ici. "
                      "Pour l'essayer : pip install sentence-transformers torch "
                      "(environ 2,5 Go).")
    try:
        modele = CrossEncoder(nom, max_length=512)
    except Exception as e:
        return None, f"chargement de {nom} impossible : {e}"

    try:
        import torch
        materiel = "GPU" if torch.cuda.is_available() else "processeur"
    except ImportError:
        materiel = "processeur"

    def scorer(question: str, passages: list[str]) -> list[float]:
        return list(modele.predict([(question, p) for p in passages]))

    return scorer, f"{nom} chargé (reclassement sur {materiel})."


def mesurer(dossier: str, questions: list[dict], strategie: str, taille: int,
            modele_emb: str, nom_reranker: str, retrieve_k: int) -> dict:
    import chunker
    from index_rag import embed_texts

    morceaux = [c.text for c in chunker.chunk_corpus(
        dossier, strategy=strategie, target_size=taille, verbose=False)]
    norm = [_normaliser(m) for m in morceaux]

    # Le morceau attendu est celui qui contient la réponse — déterminé après le
    # découpage, donc les questions restent valables si le découpage change.
    utilisables = []
    for q in questions:
        cibles = {i for i, m in enumerate(norm) if _normaliser(q["reponse"]) in m}
        if cibles:
            utilisables.append((q, cibles))
    if not utilisables:
        return {"ok": False, "erreur": "aucune réponse retrouvée dans les morceaux"}

    v_morceaux = embed_texts(morceaux, modele_emb)
    v_questions = embed_texts([q["question"] for q, _ in utilisables], modele_emb)

    # --- récupération vectorielle seule ------------------------------------
    candidats, t_vect = [], 0.0
    for vq in v_questions:
        t0 = time.time()
        scores = sorted(((_cosinus(vq, v), j) for j, v in enumerate(v_morceaux)),
                        key=lambda s: -s[0])[:retrieve_k]
        t_vect += time.time() - t0
        candidats.append([j for _, j in scores])

    scorer, message = charger_reranker(nom_reranker)
    print(f"  {message}")

    # --- reclassement -------------------------------------------------------
    reclasses, t_rr = None, 0.0
    if scorer:
        reclasses = []
        for (q, _), cand in zip(utilisables, candidats):
            t0 = time.time()
            s = scorer(q["question"], [morceaux[j] for j in cand])
            t_rr += time.time() - t0
            reclasses.append([j for _, j in sorted(zip(s, cand), key=lambda x: -x[0])])

    n = len(utilisables)

    def rappel(listes, k):
        return round(sum(1 for l, (_, cibles) in zip(listes, utilisables)
                         if cibles & set(l[:k])) / n, 3)

    lignes = []
    for k in COUPES:
        if k > retrieve_k:
            continue
        ligne = {"coupe": k, "sans": rappel(candidats, k)}
        if reclasses:
            ligne["avec"] = rappel(reclasses, k)
            ligne["gain"] = round(ligne["avec"] - ligne["sans"], 3)
        lignes.append(ligne)

    return {"ok": True, "n_questions": n, "n_morceaux": len(morceaux),
            "retrieve_k": retrieve_k, "reranker": nom_reranker if scorer else None,
            "message": message, "lignes": lignes,
            "ms_par_question_vectoriel": round(1000 * t_vect / n, 1),
            "ms_par_question_reranker": round(1000 * t_rr / n, 1) if scorer else None,
            # Les paires par question, pour mesurer l'incertitude comme ailleurs.
            "detail": [{"sans": [j for j in c], "avec": [j for j in r] if reclasses else None}
                       for c, r in zip(candidats, reclasses or candidats)]}


def choix_reranker(res: dict, gain_min: float = 0.03,
                   generation_ms: float = 2000.0,
                   plafond_part: float = 0.5) -> list:
    """
    generation_ms : temps que prend déjà la génération de la réponse. Le coût
    du reclassement se juge PAR RAPPORT à lui, pas contre une constante posée
    à la main — c'est le seul point de comparaison qui a un sens pour
    l'utilisateur, qui subit le total.

    plafond_part : part maximale du temps de réponse que le reclassement peut
    prendre. Au-delà, il double le temps d'attente pour réordonner des passages
    déjà trouvés.
    """
    if not res.get("reranker"):
        return [regle("reranker", "aucun",
                      f"NON MESURABLE ICI : {res['message']} La config le "
                      f"recommandait pourtant — un paramètre qu'on ne peut pas "
                      f"exécuter ne devrait pas être recommandé sans réserve.")]

    # On juge sur la coupe où le reranker sert LE PLUS, pas sur la plus grande.
    # Juger sur la plus grande masquait le résultat : à la coupe 10 le rappel
    # sature déjà sans reranker, donc le gain y est forcément nul — même quand
    # le reranker fait remonter nettement le bon passage à la coupe 3.
    ligne = max(res["lignes"], key=lambda l: l["gain"])
    grande = max(res["lignes"], key=lambda l: l["coupe"])
    cout = res["ms_par_question_reranker"] / max(res["ms_par_question_vectoriel"], 0.1)
    ms = res["ms_par_question_reranker"]

    # Le gain et le coût sont deux questions séparées : un reranker peut très
    # bien MARCHER et rester inutilisable. Les mélanger dans une note unique
    # ferait disparaître l'information utile.
    utile = ligne["gain"] >= gain_min
    plafond_ms = generation_ms * plafond_part
    payable = ms <= plafond_ms
    part = ms / max(generation_ms, 1.0)

    if utile and payable:
        ch = mesure("reranker", [("aucun", ligne["sans"]), (res["reranker"], ligne["avec"])],
                    ecart_min=gain_min)
        ch.raison = (f"coupe à {ligne['coupe']} · {ligne['gain']:+.1%} · "
                     f"{ms:.0f} ms par question")
        return [ch, contraint("retrieve_k", res["retrieve_k"],
                              "nombre de candidats soumis au reranker"),
                contraint("top_k", ligne["coupe"],
                          f"le reranker permet de couper à {ligne['coupe']} au lieu "
                          f"de {grande['coupe']} pour un rappel comparable — "
                          f"contexte plus court, génération plus rapide")]

    if utile and not payable:
        ch = mesure("reranker", [("aucun", ligne["sans"]), (res["reranker"], ligne["avec"])],
                    ecart_min=gain_min)
        ch.valeur, ch.egalite = "aucun", False
        ch.raison = (f"IL MARCHE MAIS IL COÛTE TROP CHER : {ligne['gain']:+.1%} à "
                     f"la coupe {ligne['coupe']} ({ligne['sans']:.0%} → "
                     f"{ligne['avec']:.0%}), pour {ms:.0f} ms par question, soit "
                     f"{cout:.0f} fois la recherche seule et {part:.0%} du temps "
                     f"de génération. {res.get('message', '')} Ce verdict tient au "
                     f"matériel disponible, pas au modèle : sur une carte non "
                     f"partagée avec les modèles Ollama, le reclassement serait "
                     f"bien plus rapide et le compromis pourrait s'inverser")
        return [ch]

    ch = mesure("reranker", [("aucun", ligne["sans"]), (res["reranker"], ligne["avec"])],
                ecart_min=gain_min)
    ch.raison = (f"meilleur gain {ligne['gain']:+.1%} à la coupe {ligne['coupe']} — "
                 f"sous le seuil utile, pour {ms:.0f} ms par question")
    return [ch]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Le reranker vaut-il son coût ?")
    ap.add_argument("--input", required=True)
    ap.add_argument("--questions", required=True, help="questions_auto.json")
    ap.add_argument("--config", help="advisor_mesure.json")
    ap.add_argument("--reranker", required=True,
                    help="reranker à évaluer (nom sentence-transformers)")
    ap.add_argument("--retrieve-k", type=int, default=RETRIEVE_K)
    ap.add_argument("--embedding", required=True,
                    help="modèle d'embedding (celui de l'index)")
    ap.add_argument("--out", default="reranker_mesure.json")
    args = ap.parse_args()

    strategie, taille, modele_emb = "fixed", 1382, args.embedding
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        strategie  = cfg.get("_chunk_strategy", strategie)
        taille     = int(cfg.get("_chunk_chars", taille))
        modele_emb = cfg.get("_embedding_model", modele_emb)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"Découpage {strategie} à {taille} caractères · embedding {modele_emb} · "
          f"{len(questions)} questions\n")

    res = mesurer(args.input, questions, strategie, taille, modele_emb,
                  args.reranker, args.retrieve_k)
    if not res["ok"]:
        raise SystemExit(res["erreur"])

    print(f"\n{res['n_questions']} questions · {res['retrieve_k']} candidats récupérés\n")
    for l in res["lignes"]:
        if "avec" in l:
            print(f"  coupe à {l['coupe']:2d}   sans {l['sans']:.0%}   "
                  f"avec {l['avec']:.0%}   ({l['gain']:+.1%})")
        else:
            print(f"  coupe à {l['coupe']:2d}   sans reranker {l['sans']:.0%}")

    if res["ms_par_question_reranker"] is not None:
        print(f"\n  recherche vectorielle : {res['ms_par_question_vectoriel']} ms/question")
        print(f"  reclassement          : {res['ms_par_question_reranker']} ms/question")

    print()
    print(resume(choix_reranker(res)))

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nDétail écrit dans {args.out}.")