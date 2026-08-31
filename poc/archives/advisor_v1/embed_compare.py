# -*- coding: utf-8 -*-
"""
embed_compare.py — mesurer le modèle d'embedding au lieu de le subir.

Pourquoi ce fichier
-------------------
L'Advisor écrivait « embedding_model = bge-m3 — imposé : seul modèle installé ».
Dès qu'un deuxième modèle est présent, cette phrase devient fausse : ce n'est
plus une contrainte, c'est un choix. Et un choix se mesure.

Le test, et ses limites
-----------------------
Comparer deux modèles demande normalement des questions et leurs bonnes
réponses. On n'en a pas.

À la place : on prend une phrase AU MILIEU d'un morceau, on s'en sert comme
requête, et on regarde à quel rang le modèle retrouve le morceau d'origine.

  rang 1 sur 40 essais  -> le modèle sépare bien les morceaux
  rang 12 en moyenne    -> il les confond

CE QUE CE TEST NE MESURE PAS, et il faut le dire dans le rapport :
la requête est un EXTRAIT LITTÉRAL du morceau. Une vraie question serait
reformulée, avec d'autres mots. Ce test récompense donc la ressemblance de
surface plus qu'un usage réel ne le ferait.

Le biais est le MÊME pour tous les modèles comparés : il sert à les départager,
pas à annoncer un niveau de performance absolu.

Usage :
    python embed_compare.py --input ooredoo --config advisor_mesure.json
    python embed_compare.py --input ooredoo --models bge-m3 qwen3-embedding:0.6b
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

from decision import mesure, contraint, regle, resume

MODELES_DEFAUT = ["bge-m3", "qwen3-embedding:0.6b"]
N_REQUETES     = 40
MOTS_MIN       = 8      # une requête plus courte ne porte aucune information
GRAINE         = 12345  # même échantillon pour tous les modèles : sinon on
                        # compare des modèles ET des échantillons différents


def _morceaux(dossier: str, strategie: str, taille: int) -> list[str]:
    """Découpe le corpus avec la stratégie retenue par l'Advisor."""
    import chunker
    ch = chunker.chunk_corpus(dossier, strategy=strategie, target_size=taille,
                              verbose=False)
    return [c.text for c in ch if len(c.text.split()) >= MOTS_MIN * 2]


def _requete(texte: str) -> str | None:
    """
    Une phrase prise au MILIEU du morceau.

    Au milieu, et pas au début : un morceau commence souvent par un titre ou une
    en-tête, que n'importe quel modèle retrouve sans effort. Le test serait plus
    facile et discriminerait moins.
    """
    phrases = [p.strip() for p in re.split(r"(?<=[.!?])\s+", texte)
               if len(p.split()) >= MOTS_MIN]
    return phrases[len(phrases) // 2] if phrases else None


def _cosinus(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def evaluer(modele: str, morceaux: list[str], indices: list[int]) -> dict:
    """Rang du bon morceau, pour chaque requête. Plus le rang est bas, mieux c'est."""
    from index_rag import embed_texts

    t0 = time.time()
    vecteurs = embed_texts(morceaux, modele)
    t_index = time.time() - t0

    requetes = [(_requete(morceaux[i]), i) for i in indices]
    requetes = [(q, i) for q, i in requetes if q]
    if not requetes:
        raise RuntimeError("aucune requête exploitable dans ce corpus")

    t0 = time.time()
    v_req = embed_texts([q for q, _ in requetes], modele)
    t_req = time.time() - t0

    rangs = []
    for vq, (_, bon) in zip(v_req, requetes):
        scores = [(_cosinus(vq, v), j) for j, v in enumerate(vecteurs)]
        scores.sort(key=lambda s: -s[0])
        rangs.append(next(r for r, (_, j) in enumerate(scores, 1) if j == bon))

    n = len(rangs)
    return {
        "modele": modele,
        # Les rangs requête par requête. Sans eux, impossible de savoir si un
        # écart de MRR vient d'une vraie différence ou d'une seule requête.
        "rangs": rangs,
        "dimension": len(vecteurs[0]),
        "n_requetes": n,
        "rang1": round(sum(1 for r in rangs if r == 1) / n, 3),
        "rang5": round(sum(1 for r in rangs if r <= 5) / n, 3),
        # MRR : moyenne de 1/rang. Récompense un bon rang sans être écrasée par
        # un échec isolé, contrairement au rang moyen.
        "mrr": round(sum(1 / r for r in rangs) / n, 3),
        "rang_median": sorted(rangs)[n // 2],
        "secondes_index": round(t_index, 1),
        "secondes_requetes": round(t_req, 1),
    }


def comparer(dossier: str, modeles: list[str], strategie: str = "fixed",
             taille: int = 1382, n_requetes: int = N_REQUETES) -> dict:
    morceaux = _morceaux(dossier, strategie, taille)
    if len(morceaux) < 10:
        return {"ok": False, "erreur": f"{len(morceaux)} morceaux : trop peu pour "
                                       f"que le classement veuille dire quelque chose"}

    random.Random(GRAINE).shuffle(indices := list(range(len(morceaux))))
    plafond = len(morceaux)
    if n_requetes > plafond:
        print(f"  (demandé {n_requetes} requêtes, plafonné à {plafond} : une "
              f"requête par morceau au maximum)")
    indices = indices[:min(n_requetes, plafond)]

    resultats, echecs = [], []
    for m in modeles:
        try:
            r = evaluer(m, morceaux, indices)
            r["_plafond"] = len(morceaux)
            resultats.append(r)
            print(f"  {m:26s} rang1 {r['rang1']:.0%}  rang5 {r['rang5']:.0%}  "
                  f"MRR {r['mrr']:.3f}  ({r['secondes_index']}s)")
        except Exception as e:
            echecs.append({"modele": m, "raison": str(e)})
            print(f"  {m:26s} ÉCHEC : {e}")

    return {"ok": bool(resultats), "n_morceaux": len(morceaux),
            "resultats": resultats, "echecs": echecs}


def ecart_significatif(a: dict, b: dict, tirages: int = 2000) -> dict:
    """
    L'écart de MRR entre deux modèles est-il plus grand que le bruit ?

    Un seuil fixe (« 0,03 d'écart ») ne veut rien dire : il ne sait pas combien
    de requêtes ont servi. Sur 40 requêtes, un écart de 0,03 peut tenir à UNE
    requête ; sur 500, le même écart serait solide.

    On mesure donc l'incertitude au lieu de la supposer. Les deux modèles ont
    répondu aux MÊMES requêtes : on compare donc par PAIRES, puis on retire au
    hasard des échantillons de ces paires (bootstrap) pour voir à quel point
    l'écart bouge. S'il change de signe, il n'y a pas de gagnant.
    """
    ra, rb = a["rangs"], b["rangs"]
    n = min(len(ra), len(rb))
    diffs = [1 / ra[i] - 1 / rb[i] for i in range(n)]

    alea = random.Random(GRAINE)
    echant = []
    for _ in range(tirages):
        tirage = [diffs[alea.randrange(n)] for _ in range(n)]
        echant.append(sum(tirage) / n)
    echant.sort()
    bas, haut = echant[int(0.025 * tirages)], echant[int(0.975 * tirages)]

    return {
        "ecart_mrr": round(sum(diffs) / n, 4),
        "intervalle": (round(bas, 4), round(haut, 4)),
        # Si l'intervalle contient zéro, l'écart peut être nul : pas de gagnant.
        "tranche": bas > 0 or haut < 0,
        "gagne_a": sum(1 for d in diffs if d > 0),
        "gagne_b": sum(1 for d in diffs if d < 0),
        "egalite": sum(1 for d in diffs if d == 0),
    }


def choix_embedding(comp: dict) -> list:
    """Traduit la comparaison en Choix, pour la config de l'Advisor."""
    res = comp.get("resultats") or []
    if len(res) < 2:
        seul = res[0]["modele"] if res else "bge-m3"
        return [contraint("embedding_model", seul,
                          "un seul modèle d'embedding a répondu : pas de choix "
                          "possible, donc rien à mesurer")]

    # Ordre du plus rapide au plus lent : à qualité égale, le modèle qui indexe
    # deux fois plus vite est le bon.
    res.sort(key=lambda r: r["secondes_index"])
    meilleur = max(res, key=lambda r: r["mrr"])
    autres = [r for r in res if r is not meilleur]
    test = ecart_significatif(meilleur, autres[0])

    if not test["tranche"]:
        # L'écart n'est pas distinguable du bruit : on garde le plus rapide et
        # on le DIT, au lieu de couronner sur deux millièmes de marge.
        #
        # Ce verdict reste MESURÉ, pas « réglé » : conclure à une égalité demande
        # exactement le même travail que désigner un gagnant. Le ranger parmi
        # les valeurs non vérifiées effacerait la mesure qui l'a produit.
        rapide = res[0]
        ch = mesure("embedding_model", [(r["modele"], r["mrr"]) for r in res],
                    ecart_min=0.0)
        ch.valeur, ch.egalite = rapide["modele"], False
        ch.raison = (f"AUCUN GAGNANT : écart de MRR {test['ecart_mrr']:+.3f}, "
                     f"intervalle [{test['intervalle'][0]:+.3f}, "
                     f"{test['intervalle'][1]:+.3f}] — il contient zéro, donc "
                     f"l'écart n'est pas distinguable du bruit sur "
                     f"{meilleur['n_requetes']} requêtes. On garde le plus "
                     f"rapide ({rapide['secondes_index']}s)")
        return [ch, contraint("embedding_dim", rapide["dimension"],
                              f"imposée par {rapide['modele']}")]

    gagne = mesure("embedding_model", [(r["modele"], r["mrr"]) for r in res],
                   ecart_min=0.0)
    vainqueur = next(r for r in res if r["modele"] == gagne.valeur)
    gagne.raison = (f"écart {test['ecart_mrr']:+.3f}, intervalle "
                    f"[{test['intervalle'][0]:+.3f}, {test['intervalle'][1]:+.3f}] "
                    f"— il ne contient pas zéro")
    return [
        gagne,
        contraint("embedding_dim", vainqueur["dimension"],
                  f"imposée par {vainqueur['modele']} — si elle diffère de "
                  f"l'index existant, il faut le reconstruire"),
    ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Comparer des modèles d'embedding.")
    ap.add_argument("--input", required=True, help="dossier des documents")
    ap.add_argument("--config", help="advisor_mesure.json (pour le découpage mesuré)")
    ap.add_argument("--models", nargs="+", default=MODELES_DEFAUT)
    ap.add_argument("--n", type=int, default=N_REQUETES)
    ap.add_argument("--out", default="embedding_mesure.json")
    args = ap.parse_args()

    strategie, taille = "fixed", 1382
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        strategie = cfg.get("_chunk_strategy", strategie)
        taille    = int(cfg.get("_chunk_chars", taille))
    print(f"Découpage : {strategie} à {taille} caractères.\n")

    comp = comparer(args.input, args.models, strategie, taille, args.n)
    if not comp["ok"]:
        raise SystemExit(comp.get("erreur", "aucun modèle n'a répondu"))

    n_reel = comp["resultats"][0]["n_requetes"]
    print(f"\n{comp['n_morceaux']} morceaux, {n_reel} requêtes.\n")
    print(resume(choix_embedding(comp)))

    if len(comp["resultats"]) >= 2:
        a, b = comp["resultats"][0], comp["resultats"][1]
        t = ecart_significatif(a, b)
        print(f"\nComparaison requête par requête : {a['modele']} devant sur "
              f"{t['gagne_a']}, {b['modele']} devant sur {t['gagne_b']}, "
              f"à égalité sur {t['egalite']}.")
    print("\nRappel : la requête est un extrait littéral du morceau. Ce test "
          "départage les modèles, il ne mesure pas leur niveau absolu.")

    Path(args.out).write_text(json.dumps(comp, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nDétail écrit dans {args.out}.")