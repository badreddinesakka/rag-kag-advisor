# -*- coding: utf-8 -*-
"""
query_rag.py - extraction des exigences a partir de l'index Milvus (v5).

Chaine : requetes -> vecteurs (Ollama) -> recherche Milvus -> passages
         -> UN APPEL LLM PAR PASSAGE -> validation -> fusion -> JSON.

CE QUI CHANGE PAR RAPPORT A LA v4
=================================

1. LE CHAMP « page » VIDE DEVIENT UN CHAMP « source » RENSEIGNE.
   Les versions precedentes livraient « page: null » en attendant que
   l'indexation produise le numero de page. Un champ toujours vide est pire
   qu'un champ absent : il annonce une information qui n'existe pas.
   « source » vaut desormais la section et son titre, par exemple
   « 19.3 Required Certifications », meme forme que dans le fichier de
   reference de la camarade qui traite le meme document.

   POURQUOI LA SECTION PLUTOT QUE LA PAGE, decide avec Bader le 29/08 :
     - une page melange plusieurs sujets sans rapport (la page 11 contient
       l'eligibilite, les couts et l'implementation), la section pose le
       lecteur sur le bon paragraphe ;
     - la section survit a une reedition du document, la page non ;
     - la section 19 de ce RFP, ajoutee apres coup, n'a AUCUNE numerotation de
       page imprimee : le pied de page s'arrete a « 13 of 13 » alors que le
       fichier fait 15 pages. Un numero de page y serait deja une convention.
   Le numero de page reste souhaitable EN PLUS, pas a la place. Il demande de
   modifier l'extraction, le decoupage et le schema Milvus, puis de reindexer.
   Le champ est donc conserve dans les provenances, pret a etre rempli.

2. LA CATEGORIE DE REPLI VIENT DE prompts.CATEGORIE_DEFAUT.
   Elle etait ecrite en dur (« operational »), un mot qui n'existe plus depuis
   que les categories sont passees a dix. Une valeur de repli codee en dur dans
   un fichier et definie dans un autre finit toujours par diverger.

CE QUI VIENT DES VERSIONS PRECEDENTES
=====================================
- DEDUPLICATION APPROXIMATIVE (v4). La comparaison exacte ne marche plus depuis
  que le prompt demande une reformulation : « single PDF bill » et « single PDF
  invoice » sont la meme exigence. On compare les mots significatifs (indice de
  Jaccard) apres retrait des mots vides et des tournures imposees par le prompt.
  SEUIL VOLONTAIREMENT CONSERVATEUR. Mesure du 29/08 sur la sortie v5 : les
  vrais et les faux doublons ne sont PAS separables par un seuil. A 0,67 on
  trouve un vrai doublon (rotation du personnel) ET deux fausses paires
  (methodologie vs outils, exigences fonctionnelles vs techniques) ; a 0,50,
  CCNA vs CCNP. Baisser le seuil detruirait de vraies exigences. On garde 0,72
  et on accepte quelques doublons residuels.
- LES NOMS DE CHAMPS SONT LUS DANS LE SCHEMA (v3). La collection du RFP porte
  texte/enfant_id/parent_id/section/titre/tableau, pas text/source/chunk_index.
  Si un champ indispensable manque, le script s'arrete en le disant.
- UN APPEL LLM PAR PASSAGE (v2). Un seul appel global obligeait le modele a
  sortir toutes les exigences en une generation, situation ou il s'arretait
  vers 17-19 lignes.
- VALIDATION DE LA SORTIE (v2). Categories hors liste, cles deformees et JSON
  illisibles sont COMPTES, pas corriges en silence.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

from index_rag import OLLAMA_BASE, _milvus_client, embed_texts
from prompts import (
    CATEGORIE_DEFAUT,
    CATEGORIES,
    FORMAT_JSON,
    PROMPTS,
    REQUETES_RECHERCHE,
    STATUTS,
)

REQUEST_TIMEOUT = 900
NUM_CTX = 16_384
NUM_PREDICT = 2_000

# Noms possibles du champ qui porte le texte, par ordre de preference.
# index_rfp.py ecrit « texte », index_rag.py ecrit « text ».
NOMS_TEXTE = ("texte", "text", "contenu", "content")

# Champs facultatifs recuperes s'ils existent. Aucun n'est indispensable.
CHAMPS_FACULTATIFS = (
    "enfant_id", "parent_id", "section", "titre", "tableau",
    "source", "chunk_index", "parent_index", "page",
)

# Seuil de recouvrement au-dela duquel deux exigences sont fusionnees.
# Voir la note en tete de fichier sur pourquoi il reste haut.
SEUIL_DOUBLON = 0.72

# Mots retires avant la comparaison de doublons. Deux familles :
#   - les mots vides de l'anglais ;
#   - les mots de structure que le prompt fait ecrire a CHAQUE ligne
#     (« the bidder must provide... »), qui gonfleraient artificiellement le
#     recouvrement entre deux exigences pourtant sans rapport.
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


# ---------------------------------------------------------------------------
# LECTURE DU SCHEMA
# ---------------------------------------------------------------------------
def inspecter(client, collection: str) -> tuple[str, list[str]]:
    """
    Retourne (nom du champ texte, champs a demander a la recherche).

    Echoue bruyamment si aucun champ de texte n'est reconnu : mieux vaut un
    message clair ici qu'une erreur Milvus obscure trois fonctions plus loin.
    """
    try:
        info = client.describe_collection(collection)
    except Exception as e:
        raise SystemExit(
            f"[ECHEC] Impossible de lire le schema de « {collection} » : {e}")

    noms = [f.get("name") for f in info.get("fields", [])]

    champ_texte = next((n for n in NOMS_TEXTE if n in noms), None)
    if champ_texte is None:
        raise SystemExit(
            f"[ECHEC] Aucun champ de texte reconnu dans « {collection} ».\n"
            f"        Champs presents : {noms}\n"
            f"        Noms attendus   : {list(NOMS_TEXTE)}\n"
            f"        Ajoute le bon nom dans NOMS_TEXTE en haut de ce fichier.")

    champs = [champ_texte] + [n for n in CHAMPS_FACULTATIFS if n in noms]
    return champ_texte, champs


# ---------------------------------------------------------------------------
# RECUPERATION
# ---------------------------------------------------------------------------
def recuperer(collection: str, embed_model: str, top_k: int,
              retrieve_k: int, requetes: list[str] | None = None):
    """
    Lance une recherche par requete, puis fusionne les resultats.

    PLUSIEURS REQUETES. Une seule recherche du type « quelles sont les
    exigences ? » ressemble a tout et a rien. On en lance une par famille et on
    fusionne les passages obtenus. La deduplication se fait sur l'identifiant
    du passage : un meme passage remonte par trois requetes n'est analyse
    qu'une fois.
    """
    requetes = requetes or REQUETES_RECHERCHE
    client = _milvus_client()

    champ_texte, champs = inspecter(client, collection)
    print(f"champ de texte : « {champ_texte} » · champs lus : {champs}")

    try:
        client.load_collection(collection)
    except Exception:
        pass  # deja chargee, ou version de pymilvus qui charge automatiquement

    vecteurs = embed_texts(requetes, embed_model)

    trouves: dict[str, dict] = {}
    n_hits = 0
    for vec in vecteurs:
        res = client.search(
            collection_name=collection,
            data=[vec],
            limit=retrieve_k,
            output_fields=champs,
        )
        for hit in (res[0] if res else []):
            ent = hit.get("entity", hit) or {}
            score = float(hit.get("distance", 0.0))
            n_hits += 1

            texte = (ent.get(champ_texte) or "").strip()
            if not texte:
                continue

            cle = str(ent.get("enfant_id")
                      or ent.get("chunk_index")
                      or hash(texte))

            if cle not in trouves or score > trouves[cle]["score"]:
                trouves[cle] = {
                    "texte": texte,
                    "score": score,
                    "enfant_id": ent.get("enfant_id"),
                    "parent_id": ent.get("parent_id"),
                    "section": ent.get("section"),
                    "titre": ent.get("titre"),
                    "tableau": ent.get("tableau"),
                    "page": ent.get("page"),   # null tant que l'index n'en a pas
                }
            trouves[cle]["n_remontees"] = trouves[cle].get("n_remontees", 0) + 1

    passages = sorted(trouves.values(), key=lambda r: -r["score"])[:top_k]
    for i, p in enumerate(passages, start=1):
        p["etiquette"] = f"P{i}"
    return passages, n_hits, len(trouves)


def _source(p: dict) -> str:
    """
    Localisation lisible d'une exigence : « 19.3 Required Certifications ».

    Remplace le champ « page » qui restait vide. Voir la note en tete de
    fichier sur le choix de la section plutot que de la page.
    """
    section = str(p.get("section") or "").strip()
    titre = str(p.get("titre") or "").strip()
    if section and titre:
        return f"{section} {titre}"
    return section or titre or "?"


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
    return body.get("response", ""), {
        "duree_s": round(time.time() - debut, 1),
        "raison_arret": body.get("done_reason"),
        "tokens_generes": body.get("eval_count"),
    }


# ---------------------------------------------------------------------------
# VALIDATION DE LA SORTIE
# ---------------------------------------------------------------------------
def _cle_propre(valeur) -> str:
    v = unicodedata.normalize("NFKD", str(valeur))
    v = "".join(c for c in v if not unicodedata.combining(c))
    return v.strip().lower()


def _lire_reponse(brut: str, etiquette: str, anomalies: dict) -> list[dict]:
    """
    Transforme la reponse du modele en lignes propres.

    Toute ligne hors contrat est COMPTEE, pas corrigee en silence : un modele
    qui invente des categories est une information a garder pour le rapport.
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
            cat = CATEGORIE_DEFAUT   # repli, mais l'ecart est deja compte

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
# FUSION APPROXIMATIVE
# ---------------------------------------------------------------------------
def _mots_significatifs(texte: str) -> frozenset:
    """
    Reduit une exigence a l'ensemble de ses mots porteurs de sens.

    Sans le retrait des tournures imposees, « The bidder must provide X » et
    « The bidder must provide Y » partageraient deja quatre mots sur six.
    """
    t = unicodedata.normalize("NFKD", texte.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    mots = re.findall(r"[a-z0-9]+", t)
    return frozenset(m for m in mots if m not in MOTS_IGNORES and len(m) > 1)


def _recouvrement(a: frozenset, b: frozenset) -> float:
    """Indice de Jaccard : mots communs / mots distincts."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fusionner(lignes: list[dict], index: dict,
              seuil: float = SEUIL_DOUBLON) -> tuple[list[dict], int]:
    """
    Regroupe les exigences quasi identiques en CONSERVANT tout.

    Deux choses sont conservees a chaque fusion :
      - les PROVENANCES, car une exigence peut legitimement figurer dans deux
        passages, donc dans deux sections ;
      - les VARIANTES de formulation, pour qu'une fusion abusive reste visible
        et rattrapable a la lecture.
    """
    retenues: list[dict] = []
    empreintes: list[frozenset] = []
    doublons = 0

    for l in lignes:
        emp = _mots_significatifs(l["requirement"])
        p = index.get(l["passage"], {})
        provenance = {
            "passage": l["passage"],
            "source": _source(p),
            "section": p.get("section"),
            "titre": p.get("titre"),
            "parent_id": p.get("parent_id"),
            "enfant_id": p.get("enfant_id"),
            "tableau": p.get("tableau"),
            "page": p.get("page"),   # pret a etre rempli le jour ou l'index
        }                            # portera la page ; absent aujourd'hui

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
def extraire(collection: str, embed_model: str, gen_model: str,
             top_k: int, retrieve_k: int, num_ctx: int,
             seuil_doublon: float = SEUIL_DOUBLON,
             nom_prompt: str = "rag") -> dict:
    passages, n_hits, n_uniques = recuperer(
        collection, embed_model, top_k, retrieve_k)
    if not passages:
        return {"ok": False, "error": f"Aucun passage trouve dans « {collection} »."}

    index = {p["etiquette"]: p for p in passages}
    anomalies = {
        "json_illisible": 0, "racine_inattendue": 0, "ligne_non_objet": 0,
        "exigence_vide": 0, "categorie_hors_liste": 0, "statut_hors_liste": 0,
        "etiquette_incorrecte": 0, "categories_vues": {},
    }

    modele_prompt = PROMPTS[nom_prompt]
    brutes, vides, duree_totale = [], 0, 0.0

    print(f"\n{len(REQUETES_RECHERCHE)} requetes -> {n_hits} resultats "
          f"-> {n_uniques} passages distincts -> {len(passages)} analyses\n")

    for i, p in enumerate(passages, start=1):
        bloc = f"[{p['etiquette']}]\n{p['texte']}"
        prompt = modele_prompt % (p["etiquette"], bloc)
        etiq = f"{p['etiquette']} ({p.get('section') or '?'})"
        print(f"  {etiq:<20} {i}/{len(passages)}…", end=" ", flush=True)
        try:
            reponse, diag = _generer(prompt, gen_model,
                                     FORMAT_JSON[nom_prompt], num_ctx)
        except Exception as e:
            print(f"ERREUR : {e}")
            anomalies["json_illisible"] += 1
            continue
        duree_totale += diag["duree_s"]
        lignes = _lire_reponse(reponse, p["etiquette"], anomalies)
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
        "architecture": "RAG",
        "collection": collection,
        "modele_generation": gen_model,
        "modele_embedding": embed_model,
        "top_k": top_k,
        "retrieve_k": retrieve_k,
        "seuil_doublon": seuil_doublon,
        "n_requetes": len(REQUETES_RECHERCHE),
        "n_passages_distincts_trouves": n_uniques,
        "n_passages_analyses": len(passages),
        "n_passages_vides": vides,
        "n_extractions_brutes": len(brutes),
        "n_exigences": len(exigences),
        "n_fusionnees": doublons,
        "taux_doublons": round(doublons / len(brutes), 3) if brutes else 0,
        "par_categorie": par_categorie,
        "sections_couvertes": sorted(
            {str(p.get("section")) for p in passages if p.get("section")}),
        "anomalies": anomalies,
        "duree_llm_s": round(duree_totale, 1),
        "exigences": exigences,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Extrait les exigences d'un RFP via l'index RAG (Milvus).")
    ap.add_argument("--collection", default="rfp_criteres")
    ap.add_argument("--embed-model", default="bge-m3",
                    help="DOIT etre celui qui a construit l'index")
    ap.add_argument("--gen-model", default="qwen3:8b")
    ap.add_argument("--top-k", type=int, default=100,
                    help="passages finalement analyses (un appel LLM chacun)")
    ap.add_argument("--retrieve-k", type=int, default=10,
                    help="passages ramenes PAR requete de recherche")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--seuil-doublon", type=float, default=SEUIL_DOUBLON,
                    help="recouvrement de mots au-dela duquel deux exigences "
                         "sont fusionnees ; 1.0 = fusion exacte seulement")
    ap.add_argument("--out", default="exigences_rag.json")
    args = ap.parse_args()

    r = extraire(args.collection, args.embed_model, args.gen_model,
                 args.top_k, args.retrieve_k, args.num_ctx,
                 args.seuil_doublon)
    if not r.get("ok"):
        raise SystemExit(f"[ECHEC] {r['error']}")

    Path(args.out).write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"{r['n_exigences']} exigences "
          f"({r['n_extractions_brutes']} brutes, "
          f"{r['n_fusionnees']} fusionnees, "
          f"{r['taux_doublons']:.0%} de doublons)")
    print(f"passages sans exigence : "
          f"{r['n_passages_vides']}/{r['n_passages_analyses']}")
    print(f"sections couvertes : {len(r['sections_couvertes'])}")
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
    print("[!] Relis le champ « variantes » : il contient les formulations "
          "fusionnees, donc les fusions abusives eventuelles.")


if __name__ == "__main__":
    main()