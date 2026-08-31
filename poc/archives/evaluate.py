# -*- coding: utf-8 -*-
"""
evaluate.py — compare une réponse à la liste de référence.

Deux chiffres, et un troisième qui n'est pas un chiffre :

  RAPPEL     : combien des critères de la référence la réponse contient-elle ?
               Se mesure sur n'importe quelle réponse, JSON ou texte libre.

  PRÉCISION  : parmi les critères annoncés, combien correspondent à un critère
               réel ? Ne se mesure que sur du JSON — le prompt court ne produit
               pas de liste identifiable, donc sa précision reste vide.

  À VÉRIFIER : les critères annoncés que le rapprochement automatique n'a pas
               reconnus. CE NE SONT PAS FORCÉMENT DES INVENTIONS. Ce sont des
               lignes à lire à la main : soit le système a inventé, soit ta
               liste de référence est incomplète, soit c'est une reformulation
               que les mots-clés n'attrapent pas. Les trois cas arrivent.

Le rapprochement est volontairement simple et lisible : un critère de référence
est trouvé si, pour chacun de ses groupes de mots-clés, au moins un mot du
groupe apparaît dans le texte. On peut vérifier à la main pourquoi une ligne a
été comptée ou non — ce qu'aucune mesure de similarité de vecteurs ne permet.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def _normaliser(texte: str) -> str:
    """Minuscules, sans accents, espaces normalisés."""
    t = unicodedata.normalize("NFD", texte or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t


def charger_reference(chemin: str = "criteres_reference.json") -> list[dict]:
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)["criteres"]


def _critere_present(critere: dict, texte_norm: str) -> bool:
    """Chaque groupe de mots-clés doit être représenté par au moins un alias."""
    for groupe in critere.get("mots_cles", []):
        if not any(_normaliser(alias) in texte_norm for alias in groupe):
            return False
    return True


def _extraire_items_json(reponse: str) -> list[dict] | None:
    """
    Sort la liste de critères d'une réponse JSON. Renvoie None si ce n'est
    pas exploitable — auquel cas on retombe sur l'analyse en texte libre.
    """
    brut = (reponse or "").strip()
    brut = re.sub(r"^```(?:json)?|```$", "", brut, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(brut)
    except Exception:
        return None
    items = obj.get("criteres") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return None
    return [i for i in items if isinstance(i, dict)]


def evaluer(reponse: str, reference: list[dict]) -> dict:
    """Compare une réponse brute à la liste de référence."""
    texte_norm = _normaliser(reponse)

    # --- RAPPEL : sur le texte complet de la réponse ------------------------
    trouves, manques = [], []
    for c in reference:
        (trouves if _critere_present(c, texte_norm) else manques).append(c["id"])

    total = len(reference) or 1
    resultat = {
        "n_reference": len(reference),
        "n_trouves": len(trouves),
        "rappel": round(len(trouves) / total, 3),
        "ids_trouves": trouves,
        "ids_manques": manques,
        "reponse_json": False,
        "n_annonces": None,
        "n_rapproches": None,
        "precision": None,
        "a_verifier": [],
    }

    # --- PRÉCISION : seulement si la réponse est du JSON exploitable --------
    items = _extraire_items_json(reponse)
    if items is None:
        return resultat

    resultat["reponse_json"] = True
    resultat["n_annonces"] = len(items)

    rapproches = 0
    a_verifier = []
    for item in items:
        ligne = " ".join(str(item.get(k, "")) for k in
                         ("critere", "valeur", "statut", "categorie", "source"))
        ligne_norm = _normaliser(ligne)
        if any(_critere_present(c, ligne_norm) for c in reference):
            rapproches += 1
        else:
            a_verifier.append(item.get("critere", str(item))[:120])

    resultat["n_rapproches"] = rapproches
    resultat["precision"] = round(rapproches / len(items), 3) if items else 0.0
    resultat["a_verifier"] = a_verifier
    return resultat


def resume_par_categorie(reponse: str, reference: list[dict]) -> dict:
    """Rappel détaillé par famille de critères : où le système perd-il ?"""
    texte_norm = _normaliser(reponse)
    par_cat: dict[str, list[int]] = {}
    for c in reference:
        cat = c.get("categorie", "?")
        par_cat.setdefault(cat, [0, 0])
        par_cat[cat][1] += 1
        if _critere_present(c, texte_norm):
            par_cat[cat][0] += 1
    return {cat: {"trouves": t, "total": n, "rappel": round(t / n, 2) if n else 0.0}
            for cat, (t, n) in sorted(par_cat.items())}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Compare une réponse à la liste de référence.")
    ap.add_argument("--reponse", required=True,
                    help="fichier contenant la réponse à évaluer")
    ap.add_argument("--reference", default="criteres_reference.json")
    args = ap.parse_args()

    reference = charger_reference(args.reference)
    texte = Path(args.reponse).read_text(encoding="utf-8")

    r = evaluer(texte, reference)
    print(f"Rappel    : {r['n_trouves']}/{r['n_reference']} = {r['rappel']:.0%}")
    if r["precision"] is not None:
        print(f"Précision : {r['n_rapproches']}/{r['n_annonces']} = {r['precision']:.0%}")
    else:
        print("Précision : non mesurable (la réponse n'est pas du JSON)")

    print("\nRappel par catégorie :")
    for cat, v in resume_par_categorie(texte, reference).items():
        print(f"  {cat:<12} {v['trouves']:>2}/{v['total']:<3} {v['rappel']:.0%}")

    if r["ids_manques"]:
        print(f"\nCritères manqués ({len(r['ids_manques'])}) :")
        print("  " + ", ".join(r["ids_manques"]))

    if r["a_verifier"]:
        print(f"\nÀ VÉRIFIER À LA MAIN ({len(r['a_verifier'])}) — "
              f"invention, ou référence incomplète, ou reformulation :")
        for ligne in r["a_verifier"]:
            print(f"  · {ligne}")
