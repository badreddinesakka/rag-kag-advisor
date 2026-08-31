"""
chunk_rfp.py — Decoupage structurel d'un RFP en parents (sections) et enfants (petits morceaux).

Principe :
  - un PARENT = une section du document (detectee par son titre numerote : 1.0, 5.3.1, 19.4 ...)
  - un ENFANT = un petit morceau (~80 tokens) decoupe DANS un parent
  - un TABLEAU n'est jamais coupe : il devient un enfant entier

L'enfant sert a la recherche vectorielle (court = precis).
Le parent sera envoye au LLM (long = comprehensible).

Aucun appel LLM, aucun embedding. Sortie : chunks_rfp.json

Usage :
    python chunk_rfp.py --pdf corpus/rfp/RFP_OT_Bill_Generation_Solution_V1_with_HR.pdf
    python chunk_rfp.py --pdf ... --mots-enfant 60 --sortie chunks_rfp.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber

# --------------------------------------------------------------------------
# 1. Nettoyage : lignes a jeter (en-tetes, pieds de page, sommaire)
# --------------------------------------------------------------------------

BRUIT = [
    re.compile(r"^classification\s*:", re.I),
    re.compile(r"ooredoo tunisia confidential", re.I),
    re.compile(r"rfp for bill generation solution", re.I),
    re.compile(r"^\d+\s+of\s+\d+$", re.I),
    re.compile(r"^version\s+\d", re.I),
]

# ligne de sommaire : "5.3 Compliance Matrix ................ 7"
SOMMAIRE = re.compile(r"\.{4,}\s*\d+\s*$")


def est_bruit(ligne: str) -> bool:
    l = ligne.strip()
    if not l:
        return True
    if SOMMAIRE.search(l):
        return True
    for motif in BRUIT:
        if motif.search(l):
            return True
    return False


# --------------------------------------------------------------------------
# 2. Detection des titres
# --------------------------------------------------------------------------

# "1.0 About Ooredoo Tunisia" / "5.3.1 Requirements Definition" / "19.4 Required Technical Skills"
TITRE_NUM = re.compile(r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+([A-Za-z].{2,80})$")

# sous-titres sans numero presents dans ce type de RFP
TITRE_NU_MAX_CAR = 45


def lire_titre(ligne: str, gras: bool):
    """Renvoie (numero, intitule) si la ligne est un titre, sinon (None, None).

    `gras` = tous les mots de la ligne sont en gras dans le PDF.
    C'est ce qui distingue un vrai sous-titre ("Standard Provisions")
    d'une puce de liste ("Bidder Company name and address").
    """
    l = ligne.strip()

    m = TITRE_NUM.match(l)
    if m:
        return m.group(1), m.group(2).strip()

    # sous-titre sans numero : court, en gras, sans ponctuation finale
    if (
        gras
        and 2 < len(l) <= TITRE_NU_MAX_CAR
        and l[0].isupper()
        and not l.endswith((".", ":", ",", ";", ")"))
        and not re.search(r"\d", l)
        and len(l.split()) <= 6
        and l.lower() not in {"yes", "no", "note", "abstract"}
    ):
        return None, l

    return None, None


# --------------------------------------------------------------------------
# 3. Lecture du PDF : blocs de texte et blocs de tableau, dans l'ordre
# --------------------------------------------------------------------------

def tableau_en_texte(rows) -> str:
    """Transforme un tableau en texte lisible par un LLM."""
    lignes = []
    for row in rows:
        cellules = [(c or "").replace("\n", " ").strip() for c in row]
        if any(cellules):
            lignes.append(" | ".join(cellules))
    return "\n".join(lignes)


def lire_blocs(chemin_pdf: Path):
    """Renvoie la liste des blocs du document, dans l'ordre de lecture."""
    blocs = []

    with pdfplumber.open(chemin_pdf) as pdf:
        for num_page, page in enumerate(pdf.pages, start=1):
            blocs_page = []
            tables = page.find_tables()
            zones = [t.bbox for t in tables]

            # --- blocs tableau ---
            for t in tables:
                try:
                    rows = t.extract()
                except Exception:
                    continue
                txt = tableau_en_texte(rows)
                if txt.strip():
                    blocs_page.append(
                        {"type": "tableau", "texte": txt, "page": num_page, "haut": t.bbox[1]}
                    )

            # --- blocs texte (mots situes HORS des tableaux) ---
            mots = page.extract_words(use_text_flow=False, extra_attrs=["fontname"])
            dehors = []
            for m in mots:
                dans_tableau = any(
                    (m["x0"] >= x0 - 2 and m["x1"] <= x1 + 2 and m["top"] >= y0 - 2 and m["bottom"] <= y1 + 2)
                    for (x0, y0, x1, y1) in zones
                )
                if not dans_tableau:
                    dehors.append(m)

            # regrouper les mots en lignes par coordonnee verticale
            lignes = {}
            for m in dehors:
                cle = round(m["top"] / 3)
                lignes.setdefault(cle, []).append(m)

            for cle in sorted(lignes):
                mots_ligne = sorted(lignes[cle], key=lambda w: w["x0"])
                texte = " ".join(w["text"] for w in mots_ligne)
                if est_bruit(texte):
                    continue
                gras = all("Bold" in w.get("fontname", "") for w in mots_ligne)
                blocs_page.append(
                    {
                        "type": "texte",
                        "texte": texte.strip(),
                        "page": num_page,
                        "haut": mots_ligne[0]["top"],
                        "gras": gras,
                    }
                )

            # ORDRE DE LECTURE : un tableau doit se placer entre les paragraphes
            # qui l'entourent, pas en tete de page
            blocs_page.sort(key=lambda b: b["haut"])
            blocs.extend(blocs_page)

    return blocs


# --------------------------------------------------------------------------
# 4. Construction des parents
# --------------------------------------------------------------------------

def construire_parents(blocs):
    parents = []
    courant = {
        "section": "0",
        "titre": "Preambule",
        "page": 1,
        "lignes": [],
        "tableaux": [],
    }
    dernier_numero = "0"

    for b in blocs:
        if b["type"] == "tableau":
            courant["tableaux"].append(b["texte"])
            continue

        numero, intitule = lire_titre(b["texte"], b.get("gras", False))

        if numero:  # vrai titre numerote -> nouvelle section
            if courant["lignes"] or courant["tableaux"]:
                parents.append(courant)
            dernier_numero = numero
            courant = {
                "section": numero,
                "titre": intitule,
                "page": b["page"],
                "lignes": [],
                "tableaux": [],
            }
        elif intitule:  # sous-titre sans numero -> sous-section rattachee
            if courant["lignes"] or courant["tableaux"]:
                parents.append(courant)
            courant = {
                "section": dernier_numero,
                "titre": intitule,
                "page": b["page"],
                "lignes": [],
                "tableaux": [],
            }
        else:
            courant["lignes"].append(b["texte"])

    if courant["lignes"] or courant["tableaux"]:
        parents.append(courant)

    # mise en forme finale
    sortie = []
    for i, p in enumerate(parents):
        texte = " ".join(p["lignes"]).strip()
        texte = re.sub(r"\s{2,}", " ", texte)
        sortie.append(
            {
                "parent_id": f"P{i:03d}",
                "section": p["section"],
                "titre": p["titre"],
                "page": p["page"],
                "texte": texte,
                "tableaux": p["tableaux"],
                "n_car": len(texte) + sum(len(t) for t in p["tableaux"]),
            }
        )
    return sortie


# --------------------------------------------------------------------------
# 5. Construction des enfants
# --------------------------------------------------------------------------

FIN_PHRASE = re.compile(r"(?<=[.:;])\s+")


def decouper_en_enfants(texte: str, mots_cible: int, recouvrement: int):
    """Decoupe un texte en morceaux d'environ `mots_cible` mots,
    en respectant les fins de phrase quand c'est possible."""
    phrases = [p.strip() for p in FIN_PHRASE.split(texte) if p.strip()]
    morceaux, courant, n = [], [], 0

    for ph in phrases:
        n_ph = len(ph.split())
        if n + n_ph > mots_cible and courant:
            morceaux.append(" ".join(courant))
            # recouvrement : on garde les derniers mots
            garde = " ".join(courant).split()[-recouvrement:] if recouvrement else []
            courant = [" ".join(garde)] if garde else []
            n = len(garde)
        courant.append(ph)
        n += n_ph

    if courant:
        morceaux.append(" ".join(courant))

    return [m.strip() for m in morceaux if m.strip()]


def construire_enfants(parents, mots_cible, recouvrement):
    enfants = []
    for p in parents:
        contexte = f"[{p['section']} {p['titre']}]"

        for txt in decouper_en_enfants(p["texte"], mots_cible, recouvrement):
            enfants.append(
                {
                    "enfant_id": f"E{len(enfants):04d}",
                    "parent_id": p["parent_id"],
                    "section": p["section"],
                    "titre": p["titre"],
                    "tableau": False,
                    # le titre voyage AVEC le morceau : la recherche vectorielle
                    # comprend mieux "at least 8 persons" si "Training Services" est colle devant
                    "texte": f"{contexte} {txt}",
                }
            )

        # un tableau n'est JAMAIS coupe
        for t in p["tableaux"]:
            enfants.append(
                {
                    "enfant_id": f"E{len(enfants):04d}",
                    "parent_id": p["parent_id"],
                    "section": p["section"],
                    "titre": p["titre"],
                    "tableau": True,
                    "texte": f"{contexte} {t}",
                }
            )

    return enfants


# --------------------------------------------------------------------------
# 6. Programme principal
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Decoupage structurel d'un RFP")
    ap.add_argument("--pdf", required=True, help="chemin du PDF")
    ap.add_argument("--sortie", default="chunks_rfp.json")
    ap.add_argument("--mots-enfant", type=int, default=60,
                    help="taille visee d'un enfant, en mots (60 mots ~ 80 tokens)")
    ap.add_argument("--recouvrement", type=int, default=10,
                    help="mots repris d'un enfant a l'autre")
    args = ap.parse_args()

    chemin = Path(args.pdf)
    if not chemin.exists():
        sys.exit(f"PDF introuvable : {chemin}")

    print(f"Lecture de {chemin.name} ...")
    blocs = lire_blocs(chemin)
    n_tab = sum(1 for b in blocs if b["type"] == "tableau")
    print(f"  {len(blocs)} blocs lus dont {n_tab} tableaux")

    parents = construire_parents(blocs)
    enfants = construire_enfants(parents, args.mots_enfant, args.recouvrement)

    doc = {
        "document": chemin.name,
        "n_parents": len(parents),
        "n_enfants": len(enfants),
        "n_enfants_tableau": sum(1 for e in enfants if e["tableau"]),
        "mots_enfant": args.mots_enfant,
        "parents": parents,
        "enfants": enfants,
    }

    Path(args.sortie).write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\n{len(parents)} parents, {len(enfants)} enfants "
          f"({doc['n_enfants_tableau']} tableaux entiers)")
    print(f"Ecrit dans {args.sortie}\n")
    print("Sections detectees :")
    for p in parents:
        marque = f" +{len(p['tableaux'])} tab" if p["tableaux"] else ""
        print(f"  {p['section']:<7} {p['titre'][:45]:<47} {p['n_car']:>6} car{marque}")


if __name__ == "__main__":
    main()
