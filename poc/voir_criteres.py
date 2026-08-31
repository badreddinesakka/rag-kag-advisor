"""
voir_criteres.py — Affiche les criteres extraits, lisiblement.

Ne relance rien, ne calcule rien : il lit juste criteres_extraits.json.

Usage :
    python voir_criteres.py
    python voir_criteres.py --par section
    python voir_criteres.py --categorie equipe
    python voir_criteres.py --csv criteres.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SYMBOLE = {"obligatoire": "[O]", "recommande": "[R]", "non precise": "[ ]"}


def main():
    ap = argparse.ArgumentParser(description="Lecture des criteres extraits")
    ap.add_argument("--fichier", default="criteres_extraits.json")
    ap.add_argument("--par", choices=["categorie", "section"], default="categorie",
                    help="regroupement de l'affichage")
    ap.add_argument("--categorie", default=None, help="n'afficher qu'une categorie")
    ap.add_argument("--csv", default=None, help="exporter aussi en CSV (pour Excel)")
    args = ap.parse_args()

    chemin = Path(args.fichier)
    if not chemin.exists():
        sys.exit(f"{chemin} introuvable. Lance d'abord extract_rag.py")

    data = json.loads(chemin.read_text(encoding="utf-8"))
    criteres = data.get("criteres", [])
    if args.categorie:
        criteres = [c for c in criteres if c["categorie"] == args.categorie]

    if not criteres:
        sys.exit("Aucun critere a afficher.")

    cle = "categorie" if args.par == "categorie" else "section"
    groupes = {}
    for c in criteres:
        groupes.setdefault(c.get(cle, "?"), []).append(c)

    print(f"\nDocument : {data.get('document', '?')}")
    print(f"Modele   : {data.get('modele', '?')}")
    print(f"Total    : {len(criteres)} criteres\n")

    for g in sorted(groupes):
        lignes = groupes[g]
        print("=" * 78)
        print(f"{g.upper()}  ({len(lignes)} criteres)")
        print("=" * 78)
        for c in lignes:
            marque = SYMBOLE.get(c.get("statut", ""), "[ ]")
            suffixe = f"  ({c['section']})" if cle == "categorie" and c.get("section") else ""
            print(f" {marque} {c.get('id', '')}  {c['critere']}{suffixe}")
            if c.get("valeur"):
                print(f"        -> {c['valeur']}")
        print()

    print("Legende : [O] obligatoire   [R] recommande   [ ] non precise")

    # --- sections analysees qui n'ont rien donne : la vraie liste a examiner ---
    vides = data.get("sections_analysees_sans_resultat", [])
    if vides:
        print(f"\nSections analysees mais SANS aucun critere ({len(vides)}) :")
        for v in vides:
            print(f"   - {v}")
        print("  -> si une de ces sections contient visiblement des exigences,")
        print("     c'est le prompt qu'il faut corriger, pas les requetes.")

    if args.csv:
        champs = ["id", "categorie", "section", "critere", "valeur", "statut"]
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=champs, extrasaction="ignore", delimiter=";")
            w.writeheader()
            w.writerows(criteres)
        print(f"\nExport CSV : {args.csv}")


if __name__ == "__main__":
    main()
