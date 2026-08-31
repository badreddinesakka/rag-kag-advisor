# -*- coding: utf-8 -*-
"""
query_kag.py - extraction des exigences a partir du graphe Neo4j (v2).

Chaine : Neo4j -> triplets -> groupes etiquetes -> UN APPEL LLM PAR GROUPE
         -> validation -> fusion -> JSON.

La sortie a EXACTEMENT la meme forme que celle de query_rag.py : memes dix
categories, memes statuts, meme validation, meme fusion approximative. C'est ce
qui permet de comparer les deux fichiers ligne a ligne. Sans cela, le tableau
RAG contre KAG du rapport ne veut rien dire.

CE QUI CHANGE PAR RAPPORT A LA v1
=================================

1. UN APPEL LLM PAR GROUPE DE TRIPLETS, PLUS UN SEUL APPEL GLOBAL.
   La v1 collait jusqu'a 400 triplets dans un prompt et faisait un appel. Deux
   defauts mesures : le modele ne s'arretait plus (60 triplets -> 246 s,
   172 triplets -> timeout, d'ou le NUM_PREDICT de secours), et il s'arretait
   spontanement vers 17-19 exigences quand il s'arretait.
   Ici les triplets sont decoupes en groupes de taille fixe, un appel chacun.
   Chaque sortie est courte, le modele ne boucle plus, et une exigence manquee
   reste localisee a un groupe.

2. LE PROMPT KAG REMPLACE LE PROMPT RAG.
   Le KAG ne voit pas des phrases mais des triplets, qui ont perdu la phrase
   d'origine. Les consignes « ne complete jamais une phrase tronquee » et
   « ne recopie pas plus de six mots » n'ont aucun sens ici. Le risque n'est
   plus la recopie mais l'INVENTION : un triplet est pauvre, le modele est
   tente de reconstituer la phrase qu'il croit deviner.
   A ECRIRE DANS LE RAPPORT : le banc d'essai ne mesure donc plus des
   architectures toutes choses egales par ailleurs, mais des chaines completes,
   chacune avec son prompt. C'est ce qu'on deploierait en vrai, mais il faut le
   dire.

3. LE MODELE DE GENERATION PASSE A qwen3:8b.
   Comme pour le RAG. Attention : le GRAPHE, lui, a ete construit avec
   qwen2.5:7b, impose par le routeur, et index_kag.py n'offre aucune option
   pour en changer. L'extraction des triplets reste donc en 2.5 ; seule
   l'extraction des exigences passe en 3.

CE QU'IL FAUT SAVOIR POUR LIRE LES RESULTATS
============================================
Mesure du graphe le 30/08 : 11 morceaux, 179 relations, 283 noeuds. Mais
257 entites sur 283 - 91 % - ont un degre de 1 : elles n'apparaissent que dans
un seul triplet et ne se relient a rien. Ce n'est pas un reseau, c'est un sac
de faits isoles.
Deux causes mesurees : la resolution d'entites « basique » n'a fusionne que
4 entites sur 289, et cross_doc_connectivity vaut 0 puisque le corpus ne
contient qu'un document.
Consequence : il n'y a aucun chemin a parcourir dans ce graphe. Le KAG y perd
son avantage theorique, et c'est un resultat a publier tel quel.

COUT, A NE PAS OUBLIER DANS LE TABLEAU DU RAPPORT
=================================================
Le KAG a deja consomme un appel LLM par morceau A LA CONSTRUCTION
(11 morceaux, 14,5 min). Le RAG, zero. Si le KAG repond mieux, ce n'est pas
forcement le graphe qui est meilleur : c'est peut-etre simplement que le LLM a
deja lu tout le corpus une fois.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

from index_kag import _driver
from index_rag import OLLAMA_BASE
from prompts import (
    CATEGORIE_DEFAUT,
    CATEGORIES,
    FORMAT_JSON,
    MOTS_CLES_GRAPHE,
    PROMPTS,
    STATUTS,
)

REQUEST_TIMEOUT = 900
NUM_CTX = 16_384
NUM_PREDICT = 2_000

# Nombre de triplets envoyes par appel. 25 tient largement dans la fenetre et
# produit une sortie courte, donc pas de boucle. Monter ce nombre reduit le
# temps total mais ramene le probleme d'arret premature de la v1.
TAILLE_GROUPE = 25

# Seuil de fusion approximative, identique a query_rag.py pour que les deux
# sorties soient comparables. Mesure du 29/08 : les vrais et les faux doublons
# ne sont pas separables par un seuil, on reste donc conservateur.
SEUIL_DOUBLON = 0.72

MOTS_IGNORES = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "by", "for",
    "from", "in", "is", "it", "its", "of", "on", "or", "the", "their", "them",
    "there", "these", "this", "to", "with", "within", "that", "which", "all",
    "each", "every", "other", "such", "shall", "will", "would", "can", "may",
    "bidder", "vendor", "respondent", "supplier", "must", "should", "provide",
    "provided", "providing", "ensure", "ensures", "include", "includes",
    "including", "have", "has", "submit", "submits", "required", "requirement",
    "requirements", "solution", "ooredoo", "tunisia", "rfp", "proposal",
}

# Tout le graphe. Sur 179 relations, tout envoyer est le cas le PLUS FAVORABLE
# au KAG : on ne peut pas lui reprocher d'avoir mal cherche. Le filtrage par
# mots-cles reste disponible pour mesurer ce qu'il coute.
_CYPHER_TOUT = """
MATCH (a:Entity)-[r:RELATION]->(b:Entity)
RETURN a.name AS sujet, r.type AS relation, b.name AS objet,
       properties(r) AS props
LIMIT $limite
"""

_CYPHER_FILTRE = """
MATCH (a:Entity)-[r:RELATION]->(b:Entity)
WHERE any(k IN $mots WHERE toLower(a.name) CONTAINS k
                        OR toLower(b.name) CONTAINS k
                        OR toLower(r.type)  CONTAINS k)
RETURN a.name AS sujet, r.type AS relation, b.name AS objet,
       properties(r) AS props
LIMIT $limite
"""

_CYPHER_TAILLE = """
MATCH (n:Entity) WITH count(n) AS noeuds
MATCH ()-[r:RELATION]->() RETURN noeuds, count(r) AS relations
"""

# Noms possibles de la propriete qui dit d'ou vient le triplet. On ne sait pas
# ce que index_kag.py ecrit, donc on cherche parmi les candidats plausibles au
# lieu de supposer - la lecon du schema Milvus suppose puis plante.
CLES_PROVENANCE = ("sources", "source", "chunk", "chunk_index", "morceau",
                   "chunk_id", "origine")


# ---------------------------------------------------------------------------
# RECUPERATION
# ---------------------------------------------------------------------------
def recuperer(limite: int = 2000, filtrer: bool = False):
    """Sort les triplets du graphe. Retourne (triplets, informations)."""
    driver = _driver()
    infos = {}
    try:
        with driver.session() as session:
            ligne = session.run(_CYPHER_TAILLE).single()
            if ligne:
                infos["noeuds_graphe"] = ligne["noeuds"]
                infos["relations_graphe"] = ligne["relations"]

            if filtrer:
                mots = [m.lower() for m in MOTS_CLES_GRAPHE]
                res = session.run(_CYPHER_FILTRE, mots=mots, limite=limite)
                infos["mode"] = "filtre par mots-cles"
            else:
                res = session.run(_CYPHER_TOUT, limite=limite)
                infos["mode"] = "graphe entier"

            triplets = []
            for row in res:
                props = dict(row["props"] or {})
                provenance = None
                for k in CLES_PROVENANCE:
                    if k in props:
                        provenance = props[k]
                        break
                triplets.append({
                    "sujet": row["sujet"],
                    "relation": row["relation"],
                    "objet": row["objet"],
                    "provenance": provenance,
                })
    finally:
        driver.close()

    infos["cles_props_vues"] = sorted(
        {k for t in triplets for k in ([] if t["provenance"] is None else [1])}
    ) and "provenance trouvee" or "aucune provenance dans les proprietes"
    return triplets, infos


def grouper(triplets: list[dict], taille: int) -> list[dict]:
    """Decoupe les triplets en groupes etiquetes G1, G2, ..."""
    groupes = []
    for i in range(0, len(triplets), taille):
        lot = triplets[i:i + taille]
        groupes.append({
            "etiquette": f"G{len(groupes) + 1}",
            "triplets": lot,
            "texte": "\n".join(
                f"- {t['sujet']} -> {t['relation']} -> {t['objet']}"
                for t in lot),
        })
    return groupes


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------
def _generer(prompt: str, model: str, force_json: bool,
             num_ctx: int = NUM_CTX,
             num_predict: int = NUM_PREDICT) -> tuple[str, dict]:
    """Temperature 0 : on veut la reproductibilite entre runs."""
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx,
                    "num_predict": num_predict},
    }
    if force_json:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate", data=data,
        headers={"Content-Type": "application/json"},
    )
    debut = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    diag = {
        "duree_s": round(time.time() - debut, 1),
        "raison_arret": body.get("done_reason"),
        "tokens_generes": body.get("eval_count"),
    }
    return body.get("response", ""), diag


# ---------------------------------------------------------------------------
# VALIDATION (identique a query_rag.py, volontairement)
# ---------------------------------------------------------------------------
def _cle_propre(valeur) -> str:
    v = unicodedata.normalize("NFKD", str(valeur))
    v = "".join(c for c in v if not unicodedata.combining(c))
    return v.strip().lower()


def _lire_reponse(brut: str, etiquette: str, anomalies: dict) -> list[dict]:
    """
    Transforme la reponse du modele en lignes propres.

    Toute ligne hors contrat est COMPTEE, pas corrigee en silence.
    """
    try:
        donnees = json.loads(brut)
    except Exception:
        anomalies["json_illisible"] += 1
        return []

    if isinstance(donnees, list):
        lignes = donnees
    elif isinstance(donnees, dict):
        lignes = next((v for v in donnees.values() if isinstance(v, list)), None)
        if lignes is None:
            anomalies["racine_inattendue"] += 1
            return []
    else:
        anomalies["racine_inattendue"] += 1
        return []

    propres = []
    for ligne in lignes:
        if not isinstance(ligne, dict):
            anomalies["ligne_non_objet"] += 1
            continue
        norm = {_cle_propre(k): v for k, v in ligne.items()}

        texte = str(norm.get("requirement") or "").strip()
        if not texte:
            anomalies["exigence_vide"] += 1
            continue

        cat = _cle_propre(norm.get("category") or "")
        if cat not in CATEGORIES:
            anomalies["categorie_hors_liste"] += 1
            vues = anomalies["categories_vues"]
            vues[cat or "(vide)"] = vues.get(cat or "(vide)", 0) + 1
            cat = CATEGORIE_DEFAUT

        statut = _cle_propre(norm.get("status") or "")
        if statut not in STATUTS:
            anomalies["statut_hors_liste"] += 1
            statut = "unspecified"

        rendue = str(norm.get("passage") or "").strip()
        if rendue and rendue != etiquette:
            anomalies["etiquette_incorrecte"] += 1

        propres.append({"requirement": texte, "category": cat,
                        "status": statut, "passage": etiquette})
    return propres


# ---------------------------------------------------------------------------
# FUSION (identique a query_rag.py, volontairement)
# ---------------------------------------------------------------------------
def _mots_significatifs(texte: str) -> frozenset:
    t = unicodedata.normalize("NFKD", texte.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    mots = re.findall(r"[a-z0-9]+", t)
    return frozenset(m for m in mots if m not in MOTS_IGNORES and len(m) > 1)


def _recouvrement(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fusionner(lignes: list[dict], index: dict,
              seuil: float = SEUIL_DOUBLON) -> tuple[list[dict], int]:
    """Regroupe les exigences quasi identiques en conservant les provenances."""
    retenues: list[dict] = []
    empreintes: list[frozenset] = []
    doublons = 0

    for l in lignes:
        emp = _mots_significatifs(l["requirement"])
        g = index.get(l["passage"], {})
        provenance = {
            "passage": l["passage"],
            "source": f"graphe · groupe {l['passage']}",
            "n_triplets_du_groupe": len(g.get("triplets", [])),
        }

        meilleur, score_max = None, 0.0
        for i, autre in enumerate(empreintes):
            s = _recouvrement(emp, autre)
            if s > score_max:
                meilleur, score_max = i, s

        if meilleur is not None and score_max >= seuil:
            doublons += 1
            cible = retenues[meilleur]
            if provenance not in cible["provenances"]:
                cible["provenances"].append(provenance)
            if l["requirement"] not in cible["variantes"]:
                cible["variantes"].append(l["requirement"])
            continue

        retenues.append({
            "requirement": l["requirement"],
            "category": l["category"],
            "status": l["status"],
            "source": provenance["source"],
            "provenances": [provenance],
            "variantes": [],
        })
        empreintes.append(emp)

    return retenues, doublons


# ---------------------------------------------------------------------------
# CHAINE COMPLETE
# ---------------------------------------------------------------------------
def extraire(gen_model: str = "qwen3:8b", limite: int = 2000,
             filtrer: bool = False, taille_groupe: int = TAILLE_GROUPE,
             num_ctx: int = NUM_CTX, seuil_doublon: float = SEUIL_DOUBLON,
             nom_prompt: str = "kag") -> dict:
    try:
        triplets, infos = recuperer(limite, filtrer)
    except Exception as e:
        return {"ok": False, "error": f"Neo4j injoignable ou vide : {e}"}

    if not triplets:
        return {"ok": False, "error": "Le graphe ne contient aucune relation."}

    groupes = grouper(triplets, taille_groupe)
    index = {g["etiquette"]: g for g in groupes}

    anomalies = {
        "json_illisible": 0, "racine_inattendue": 0, "ligne_non_objet": 0,
        "exigence_vide": 0, "categorie_hors_liste": 0, "statut_hors_liste": 0,
        "etiquette_incorrecte": 0, "categories_vues": {},
    }

    modele_prompt = PROMPTS[nom_prompt]
    brutes, vides, duree_totale = [], 0, 0.0

    print(f"graphe : {infos.get('noeuds_graphe')} noeuds / "
          f"{infos.get('relations_graphe')} relations ({infos.get('mode')})")
    print(f"{len(triplets)} triplets -> {len(groupes)} groupes "
          f"de {taille_groupe}\n")

    for i, g in enumerate(groupes, start=1):
        prompt = modele_prompt % (g["etiquette"], g["texte"])
        print(f"  {g['etiquette']:<6} {i}/{len(groupes)}…", end=" ", flush=True)
        try:
            reponse, diag = _generer(prompt, gen_model,
                                     FORMAT_JSON[nom_prompt], num_ctx)
        except Exception as e:
            print(f"ERREUR : {e}")
            anomalies["json_illisible"] += 1
            continue
        duree_totale += diag["duree_s"]
        lignes = _lire_reponse(reponse, g["etiquette"], anomalies)
        if not lignes:
            vides += 1
        brutes.extend(lignes)
        print(f"{len(lignes)} exigences · {diag['duree_s']}s")

    exigences, doublons = fusionner(brutes, index, seuil_doublon)

    par_categorie = {c: 0 for c in CATEGORIES}
    for e in exigences:
        par_categorie[e["category"]] += 1

    return {
        "ok": True,
        "architecture": "KAG",
        "modele_generation": gen_model,
        "modele_extraction_graphe": "qwen2.5:7b (impose par le routeur)",
        "mode_recuperation": infos.get("mode"),
        "noeuds_graphe": infos.get("noeuds_graphe"),
        "relations_graphe": infos.get("relations_graphe"),
        "n_triplets_utilises": len(triplets),
        "taille_groupe": taille_groupe,
        "n_groupes": len(groupes),
        "n_groupes_vides": vides,
        "seuil_doublon": seuil_doublon,
        "n_extractions_brutes": len(brutes),
        "n_exigences": len(exigences),
        "n_fusionnees": doublons,
        "taux_doublons": round(doublons / len(brutes), 3) if brutes else 0,
        "par_categorie": par_categorie,
        "anomalies": anomalies,
        "duree_llm_s": round(duree_totale, 1),
        "appels_llm_construction": 11,   # mesure du 30/08
        "appels_llm_extraction": len(groupes),
        "exigences": exigences,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Extrait les exigences d'un RFP via le graphe KAG (Neo4j).")
    ap.add_argument("--gen-model", default="qwen3:8b")
    ap.add_argument("--limite", type=int, default=2000,
                    help="nombre maximum de triplets tires du graphe")
    ap.add_argument("--filtrer", action="store_true",
                    help="filtre le graphe par mots-cles au lieu de tout "
                         "prendre ; sert a mesurer ce que le filtrage coute")
    ap.add_argument("--taille-groupe", type=int, default=TAILLE_GROUPE,
                    help="triplets par appel LLM")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--seuil-doublon", type=float, default=SEUIL_DOUBLON)
    ap.add_argument("--out", default="exigences_kag.json")
    args = ap.parse_args()

    r = extraire(args.gen_model, args.limite, args.filtrer,
                 args.taille_groupe, args.num_ctx, args.seuil_doublon)
    if not r.get("ok"):
        raise SystemExit(f"[ECHEC] {r['error']}")

    Path(args.out).write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"{r['n_exigences']} exigences "
          f"({r['n_extractions_brutes']} brutes, "
          f"{r['n_fusionnees']} fusionnees, "
          f"{r['taux_doublons']:.0%} de doublons)")
    print(f"groupes sans exigence : {r['n_groupes_vides']}/{r['n_groupes']}")
    print(f"duree LLM : {r['duree_llm_s'] / 60:.1f} min")

    print("\npar categorie :")
    for c, n in sorted(r["par_categorie"].items(), key=lambda x: -x[1]):
        print(f"  {c:<16} {n}")

    a = r["anomalies"]
    ecarts = {k: v for k, v in a.items() if k != "categories_vues" and v}
    if ecarts:
        print("\nanomalies :")
        for k, v in ecarts.items():
            print(f"  {k:<24} {v}")
        if a["categories_vues"]:
            print(f"  categories inventees : {a['categories_vues']}")
    else:
        print("\naucune anomalie.")

    print(f"\nsortie : {args.out}")
    print(f"[!] Cout total du KAG : {r['appels_llm_construction']} appels a la "
          f"construction du graphe + {r['appels_llm_extraction']} ici. "
          f"Le RAG n'en a aucun a la construction.")


if __name__ == "__main__":
    main()