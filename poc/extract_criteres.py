# -*- coding: utf-8 -*-
"""
extract_criteres.py — extraction EXHAUSTIVE des exigences d'un appel d'offres.

Ce n'est PAS du RAG. C'est un balayage systématique.

Pourquoi la distinction compte
------------------------------
Le RAG CHOISIT un sous-ensemble de morceaux : c'est sa définition. Lui demander
les 253 exigences d'un document, c'est lui demander de ne rien choisir — il ne
peut pas, par construction. Et même avec une récupération parfaite, un modèle 7B
ne génère pas 253 lignes en une seule réponse : le banc d'essai en a mesuré 17.

Ici, chaque morceau du document passe une fois devant le LLM. Aucun n'est
choisi, aucun n'est écarté. Le nombre d'exigences trouvées n'est plus limité par
la longueur d'une génération, mais par le nombre de morceaux.

Découpage PARENT / ENFANT
-------------------------
  - ENFANT (~80 tokens) : l'unité d'extraction. Une ligne de tableau, une puce.
    C'est la granularité d'une exigence dans une matrice de conformité.
  - PARENT : le contexte. Sans lui, la ligne
    « CCNP | Cisco | Professional | Yes » ne veut rien dire : on ne sait pas que
    la colonne « Yes » signifie « obligatoire », ni que le tableau s'intitule
    « Required Certifications ».

L'enfant dit QUOI extraire, le parent dit ce que ça VEUT DIRE.

CE QUI CHANGE EN v2 : LE PARENT N'EST PLUS UNE TRANCHE DE 650 TOKENS
--------------------------------------------------------------------
La v1 coupait le parent tous les ~3000 caractères, au compteur. Un tableau se
retrouvait donc à cheval sur deux parents : la moitié des lignes perdaient leur
en-tête de colonnes, et avec lui le sens du « Yes ».

Le parent est désormais produit par chunker.py, qui coupe AUX SECTIONS du
document. Un tableau reste entier, et chaque parent commence par
« document > titre de section ».

Trois découpages sont disponibles, et c'est un paramètre de mesure, pas une
conviction :
    --decoupage fixe         le comportement v1 (le témoin à battre)
    --decoupage structurel   coupe aux titres      (défaut)
    --decoupage semantique   coupe où le sujet change

Compare-les sur TON document. Le compte d'exigences est la mesure :
    python extract_criteres.py --input rfp --decoupage fixe       --sortie ex_fixe.json
    python extract_criteres.py --input rfp --decoupage structurel --sortie ex_struct.json

Reprise sur incident
--------------------
Le cache est indexé par EMPREINTE DU CONTENU (sha1), pas par numéro de morceau.
Changer de document ou de découpage change les empreintes : aucune reprise
silencieuse sur les mauvaises données. C'est la correction du défaut connu de
kag_triplets_cache.json.

Utilisation :
    python extract_criteres.py --input dossier_rfp
    python extract_criteres.py --input dossier_rfp --max-chunks 10   # essai
    python extract_criteres.py --input dossier_rfp --resume
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path

import chunker
from index_rag import CHARS_PER_TOKEN, split_text
from profiler import extract_text

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

# --- découpage ---------------------------------------------------------------
# 80 tokens : la taille utilisée par l'encadrant, et la fourchette « réponses
# factuelles courtes » de Bhat et al. 2025 (référence [5] de la bibliographie).
# TAILLE DE L'ENFANT — mesure du 12/08 : avec 80 tokens (~368 caracteres),
# 92 morceaux sur 159 ont rendu ZERO exigence. Une exigence coupee en deux
# n'est extractible d'aucun cote : chaque moitie est un fragment sans verbe,
# et le prompt demande alors (correctement) une liste vide.
# 80 reste disponible via --child-tokens : c'est le temoin.
CHILD_TOKENS  = 200
PARENT_TOKENS = 650          # utilisé par --decoupage fixe uniquement
DECOUPAGE_DEFAUT = "structurel"

# chunker.py parle en anglais (fixed / structural / semantic), la ligne de
# commande en français. Une seule table de correspondance, ici.
_STRATEGIES = {
    "fixe": "fixed",
    "structurel": "structural",
    "semantique": "semantic",
}
CHILD_OVERLAP_RATIO = 0.20   # recouvrement entre enfants, pour ne pas couper
                             # une ligne de tableau en deux

REQUEST_TIMEOUT = 300
NUM_CTX         = 8_192
CACHE_FILE      = "criteres_extraction_cache.json"
CACHE_EVERY     = 10

# Deux exigences dont les textes se ressemblent à ce point sont considérées
# comme la même. Le recouvrement entre enfants fait que la même ligne est vue
# deux fois : sans fusion, tout serait compté double.
SIMILARITE_DOUBLON = 0.88


# Deux prompts : le passage ordinaire, et le tableau. Un tableau ne se lit pas
# comme un paragraphe — il faut dire au modele de sortir UNE LIGNE PAR LIGNE.
_REGLES_COMMUNES = """
RULES
- ONE requirement per row. NEVER GROUP. "CCNA and CCNP required" gives TWO rows.
- Copy exact values: 5 years, 3 WD, 20k bills/hour, ISO 27001.
- Each row must be understandable on its own.
- "statut" must be EXACTLY one of: obligatoire, recommande, non precise.
  Map: must / shall / required / mandatory / Yes -> obligatoire
       should / recommended / preferably        -> recommande
       nothing clear                            -> non precise
  NEVER write the English word from the document in that field.
- "categorie" must be EXACTLY one of: entreprise, equipe, solution, dossier,
  sla, cout. Never invent another one.
- WHAT IS NOT A REQUIREMENT: the three compliance answers the supplier will
  tick ("Fully compliant", "Partially compliant", "Not compliant") and their
  definitions are ANSWER OPTIONS, not requirements. Do not extract them.
  The requirement in that section is what the BIDDER MUST DO: submit the
  compliance matrix, sign the proposal, cross-reference each clause with a
  page number, keep the offer valid for a stated period.
- Write ONLY what is in the text. Add nothing from general knowledge.
- If the passage holds no requirement (cover page, table of contents, heading
  alone), return an empty list. That is a valid answer.

OUTPUT
JSON only:
{"exigences": [{"exigence": "...", "valeur": "...", "statut": "...", \
"categorie": "..."}]}
valeur: the exact threshold, duration or name; "-" if there is none.
"""


PROMPT = """You are reading a tender (RFP). We are building a COMPLIANCE MATRIX: \
one row per requirement, which the supplier will later tick as compliant or \
non-compliant.

Extract the requirements of the PASSAGE TO ANALYSE, and of that passage only. \
The CONTEXT is there to help you understand the passage (table title, column \
headers, current section) - extract nothing from it.

Every bullet point, every obligation, every sentence containing "must", \
"shall", "should", "required" or "Yes" is a requirement.
%(regles)s
CONTEXT:
\"\"\"
%(parent)s
\"\"\"

PASSAGE TO ANALYSE:
\"\"\"
%(child)s
\"\"\"
"""


PROMPT_TABLEAU = """You are reading a TABLE extracted from a tender (RFP). \
Cells are separated by " | " and rows by line breaks.

The FIRST row is usually the column headers. Use it to read the others, but do \
not extract it.

Produce ONE REQUIREMENT PER DATA ROW. Do not summarise the table, do not merge \
rows, do not skip rows. A table of twelve data rows gives twelve requirements.

Read the headers to understand each row:
- a "Yes" in a "Required" column makes that row mandatory;
- a duration column ("3 WD", "10 WD", "2 weeks") is the "valeur" of the row;
- a severity or priority column (Emergency, Prio1, Medium, Minor) and an action
  column (Initial Response, Workaround, Final Correction) BOTH belong in the
  requirement text - "Prio2 workaround" and "Prio2 final correction" are TWO
  DIFFERENT requirements with two different deadlines.
%(regles)s
TABLE TITLE AND SURROUNDING SECTION:
\"\"\"
%(parent)s
\"\"\"

TABLE:
\"\"\"
%(child)s
\"\"\"
"""


# ===========================================================================
# 1. DÉCOUPAGE PARENT / ENFANT
# ===========================================================================
def decouper(files: list[tuple[str, bytes]],
             decoupage: str = DECOUPAGE_DEFAUT,
             embed_model: str = chunker.EMBED_MODEL,
             child_tokens: int = CHILD_TOKENS) -> list[dict]:
    """
    Produit la liste des enfants, chacun accompagné de son parent.

    Le PARENT vient de chunker.py : une section du document, pas une tranche de
    N caractères. Il porte son titre en en-tête, et un tableau n'est jamais
    coupé en deux.

    L'ENFANT est une tranche découpée à l'intérieur du parent, avec
    recouvrement. Un enfant ne franchit jamais la frontière de son parent.

    EXCEPTION — LES TABLEAUX NE SONT JAMAIS RECOUPÉS.
    chunker.py garde chaque tableau entier, puis la v2 le retaillait en enfants
    de 368 caractères : le tableau était protégé d'un côté et cassé de l'autre.
    Mesuré sur le RFP : les lignes du tableau SLA revenaient collées à leurs
    en-têtes (« Service / Case Type Category Action Response Emergency Critical
    Restoration-workaround 1 WD ») et les lignes « correction finale », qui
    portent les 3 WD / 10 WD / 2 semaines, étaient perdues.
    Un tableau est donc UN SEUL enfant, quelle que soit sa taille.
    """
    strategie = _STRATEGIES.get(decoupage)
    if strategie is None:
        raise ValueError(f"découpage inconnu : {decoupage} "
                         f"(attendu : {', '.join(_STRATEGIES)})")

    child_chars   = int(child_tokens * CHARS_PER_TOKEN)
    child_overlap = int(child_chars * CHILD_OVERLAP_RATIO)
    parent_chars  = int(PARENT_TOKENS * CHARS_PER_TOKEN)

    enfants = []
    for nom, data in files:
        # --- construction des parents --------------------------------------
        if strategie == "fixed":
            # Comportement v1, gardé comme témoin : tranches au compteur.
            texte, _, _ = extract_text(nom, data)
            if not texte or not texte.strip():
                continue
            parents = [
                {"texte": p, "section": "", "tableau": False}
                for p in split_text(texte, parent_chars, int(parent_chars * 0.10))
            ]
        else:
            # Le fichier doit exister sur disque pour pdfplumber : on écrit le
            # contenu dans un fichier temporaire quand il vient de la mémoire
            # (cas Streamlit, où `data` sont les octets d'un upload).
            chemin = Path(nom)
            temporaire = None
            if not chemin.is_file():
                temporaire = Path(tempfile.gettempdir()) / f"_extr_{Path(nom).name}"
                temporaire.write_bytes(data)
                chemin = temporaire
            try:
                morceaux = chunker.chunk_document(
                    chemin, strategy=strategie,
                    target_size=parent_chars, embed_model=embed_model,
                )
            finally:
                if temporaire and temporaire.exists():
                    temporaire.unlink()
            parents = [{"texte": c.text, "section": c.section,
                        "tableau": c.is_table} for c in morceaux]

        if not parents:
            continue

        # --- découpage de chaque parent en enfants -------------------------
        for pi, parent in enumerate(parents):
            corps = parent["texte"]

            if parent["tableau"]:
                # Un tableau part d'un bloc : le recouper detruirait les lignes.
                tranches = [corps]
            else:
                tranches = split_text(corps, child_chars, child_overlap)

            for ci, enfant in enumerate(tranches):
                enfants.append({
                    "fichier": nom,
                    "section": parent["section"],
                    "tableau": parent["tableau"],
                    "parent_index": pi,
                    "child_index": ci,
                    "parent": corps,
                    "child": enfant,
                    "empreinte": hashlib.sha1(
                        (corps + "||" + enfant).encode("utf-8")).hexdigest(),
                })
    return enfants


# ===========================================================================
# 2. APPEL AU LLM
# ===========================================================================
def _call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


# --- normalisation des champs contraints ------------------------------------
# Le prompt impose six categories et trois statuts. Le modele n'obeit pas
# toujours : mesure du 12/08, il a produit « performance », « skill/technology »,
# « duree », et « non précise » avec accent a cote de « non precise » sans.
#
# On corrige APRES COUP plutot qu'en relancant 32 minutes de calcul. La valeur
# d'origine est toujours conservee : rien n'est perdu, tout est verifiable.

CATEGORIES = {"entreprise", "equipe", "solution", "dossier", "sla", "cout"}

# Variantes deja observees, ou previsibles. Ce n'est PAS une devinette
# generale : chaque entree correspond a une famille de mots vue en sortie.
_CATEGORIE_ALIAS = {
    "performance": "solution",
    "technology": "solution",
    "technologie": "solution",
    "skill": "equipe",
    "competence": "equipe",
    "certification": "equipe",
    "personnel": "equipe",
    "staff": "equipe",
    "team": "equipe",
    "duree": "sla",
    "delai": "sla",
    "deadline": "sla",
    "temps": "sla",
    "prix": "cout",
    "price": "cout",
    "licence": "cout",
    "vendor": "entreprise",
    "fournisseur": "entreprise",
    "company": "entreprise",
    "document": "dossier",
    "livrable": "dossier",
    "submission": "dossier",
    "proposal": "dossier",
}

_STATUT_ALIAS = {
    "obligatoire": "obligatoire", "mandatory": "obligatoire",
    "must": "obligatoire", "shall": "obligatoire", "required": "obligatoire",
    "require": "obligatoire", "yes": "obligatoire", "oui": "obligatoire",
    "recommande": "recommande", "recommended": "recommande",
    "should": "recommande", "preferably": "recommande",
    "preferred": "recommande", "optional": "recommande",
    "non precise": "non precise", "unspecified": "non precise",
    "not specified": "non precise", "n a": "non precise",
}


def _sans_accents(texte: str) -> str:
    t = unicodedata.normalize("NFD", (texte or "").strip().lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", t).strip()


def _normaliser_categorie(brut: str) -> tuple[str, bool]:
    """(categorie parmi les six, faut-il relire la ligne ?)"""
    n = _sans_accents(brut)
    if n in CATEGORIES:
        return n, False
    if n in _CATEGORIE_ALIAS:
        return _CATEGORIE_ALIAS[n], False
    # Un libelle compose (« skill/technology ») contient des mots-cles de
    # familles differentes. L'ordre de recherche est donc EXPLICITE, du plus
    # specifique au plus general : « skill/technology » decrit une competence
    # exigee d'une personne, donc equipe, et non la technologie du produit.
    for famille in ("equipe", "sla", "cout", "dossier", "entreprise", "solution"):
        for mot, cible in _CATEGORIE_ALIAS.items():
            if cible == famille and mot in n:
                return cible, False
    for cat in CATEGORIES:
        if cat in n:
            return cat, False
    # Inconnu : on ne devine PAS. La ligne est marquee pour relecture.
    return "autre", True


def _normaliser_statut(brut: str) -> tuple[str, bool]:
    """(statut parmi les trois, faut-il relire la ligne ?)"""
    n = _sans_accents(brut)
    if not n:
        return "non precise", False
    if n in _STATUT_ALIAS:
        return _STATUT_ALIAS[n], False
    for mot, cible in _STATUT_ALIAS.items():
        if mot in n:
            return cible, False
    return "non precise", True


def normaliser_ligne(r: dict) -> dict | None:
    """
    Met une exigence brute aux normes : statut parmi trois, categorie parmi six.

    APPELEE SUR TOUTE LIGNE, qu'elle sorte du LLM ou du CACHE. Le cache stocke
    des exigences deja analysees : si la normalisation vivait uniquement dans
    _parse(), un --resume la court-circuiterait entierement. C'est exactement
    ce qui s'est produit le 12/08 — la reprise a rendu « performance » et
    « non précise » intacts, sans rien recalculer.

    Renvoie None si la ligne n'est pas exploitable.
    """
    if not isinstance(r, dict):
        return None
    exigence = str(r.get("exigence", "")).strip()
    if len(exigence) < 5:
        return None

    # Une ligne deja normalisee garde sa valeur d'origine : on repart de
    # celle-la, jamais du resultat d'une normalisation precedente.
    statut_brut = str(r.get("statut_original") or r.get("statut", "")).strip()
    cat_brut = str(r.get("categorie_originale") or r.get("categorie", "")).strip()
    statut, statut_douteux = _normaliser_statut(statut_brut)
    categorie, cat_douteuse = _normaliser_categorie(cat_brut)

    ligne = {
        "exigence": exigence,
        "valeur": str(r.get("valeur", "")).strip() or "—",
        "statut": statut,
        "categorie": categorie,
    }
    # La valeur d'origine n'est gardee que si elle a ete corrigee : sinon le
    # fichier double de taille pour rien.
    if _sans_accents(statut_brut) != statut:
        ligne["statut_original"] = statut_brut
    if _sans_accents(cat_brut) != categorie:
        ligne["categorie_originale"] = cat_brut
    if statut_douteux or cat_douteuse:
        ligne["a_relire"] = True
    return ligne


def _parse(brut: str) -> list[dict]:
    try:
        obj = json.loads(brut)
    except Exception:
        return []
    lignes = obj.get("exigences") if isinstance(obj, dict) else obj
    if not isinstance(lignes, list):
        return []
    sorties = []
    for r in lignes:
        ligne = normaliser_ligne(r)
        if ligne:
            sorties.append(ligne)
    return sorties


# ===========================================================================
# 3. FUSION DES DOUBLONS
# ===========================================================================
def _normaliser(texte: str) -> str:
    t = unicodedata.normalize("NFD", texte or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn").lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def fusionner(brutes: list[dict]) -> list[dict]:
    """
    Regroupe les exigences quasi identiques.

    Les enfants se recouvrent de 20 % : la même ligne de tableau est vue par
    deux morceaux consécutifs. Sans fusion, le compte final serait gonflé
    d'environ un cinquième — et un chiffre gonflé ne vaut rien face aux 253 de
    l'encadrant.

    On garde la formulation la plus longue (généralement la plus complète) et
    on additionne les sources.
    """
    retenues: list[dict] = []
    for item in sorted(brutes, key=lambda x: -len(x["exigence"])):
        norme = _normaliser(item["exigence"])
        if not norme:
            continue
        doublon = None
        for r in retenues:
            if difflib.SequenceMatcher(None, norme, r["_norme"]).ratio() \
                    >= SIMILARITE_DOUBLON:
                doublon = r
                break
        if doublon:
            doublon["occurrences"] += 1
            if item["_source"] not in doublon["sources"]:
                doublon["sources"].append(item["_source"])
        else:
            garde = {
                "exigence": item["exigence"],
                "valeur": item["valeur"],
                "statut": item["statut"],
                "categorie": item["categorie"],
                "sources": [item["_source"]],
                "occurrences": 1,
                "_norme": norme,
            }
            for champ in ("statut_original", "categorie_originale", "a_relire"):
                if champ in item:
                    garde[champ] = item[champ]
            retenues.append(garde)

    for i, r in enumerate(retenues, start=1):
        r.pop("_norme", None)
        r["id"] = f"EXG-{i:03d}"
    return retenues


# ===========================================================================
# 4. ORCHESTRATION
# ===========================================================================
def _charger_cache(chemin: str) -> dict:
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _sauver_cache(chemin: str, cache: dict) -> None:
    try:
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def extraire(files, model="qwen2.5:7b", max_chunks=0, resume=False,
             progress=None, decoupage=DECOUPAGE_DEFAUT,
             child_tokens=CHILD_TOKENS) -> dict:
    enfants = decouper(files, decoupage, child_tokens=child_tokens)
    if not enfants:
        return {"ok": False, "error": "Aucun texte exploitable."}

    tronque = False
    if max_chunks and max_chunks < len(enfants):
        enfants = enfants[:max_chunks]
        tronque = True

    cache = _charger_cache(CACHE_FILE) if resume else {}
    brutes, echecs, repris = [], 0, 0
    # Un morceau qui rend zero exigence est du temps de calcul perdu. Sur le
    # RFP en v2, 92 morceaux sur 159 etaient dans ce cas : c'est LA mesure qui
    # dit si la taille d'enfant est bien choisie.
    vides = 0
    total = len(enfants)
    debut = time.time()

    for i, e in enumerate(enfants):
        cle = e["empreinte"]
        if cle in cache:
            # Le cache peut dater d'une version anterieure : on renormalise
            # systematiquement au lieu de faire confiance a ce qui y dort.
            trouvees = [l for l in (normaliser_ligne(x) for x in cache[cle]) if l]
            repris += 1
        else:
            if progress:
                progress(i, total)
            try:
                gabarit = PROMPT_TABLEAU if e.get("tableau") else PROMPT
                trouvees = _parse(_call_ollama(
                    gabarit % {"parent": e["parent"], "child": e["child"],
                               "regles": _REGLES_COMMUNES}, model))
            except Exception:
                echecs += 1
                continue
            cache[cle] = trouvees
            if (i + 1) % CACHE_EVERY == 0:
                _sauver_cache(CACHE_FILE, cache)

        if not trouvees:
            vides += 1
        for t in trouvees:
            t["_source"] = f"{e['fichier']}#p{e['parent_index']}c{e['child_index']}"
            brutes.append(t)

    _sauver_cache(CACHE_FILE, cache)
    if progress:
        progress(total, total)

    exigences = fusionner(brutes)
    duree = time.time() - debut

    par_categorie: dict[str, int] = {}
    par_statut: dict[str, int] = {}
    a_relire = sum(1 for x in exigences if x.get("a_relire"))
    corrigees = sum(1 for x in exigences
                    if "statut_original" in x or "categorie_originale" in x)
    for x in exigences:
        par_categorie[x["categorie"]] = par_categorie.get(x["categorie"], 0) + 1
        par_statut[x["statut"]] = par_statut.get(x["statut"], 0) + 1

    rapport = {
        "ok": True,
        "n_documents": len(files),
        "n_enfants": total,
        "child_tokens": child_tokens,
        "parent_tokens": PARENT_TOKENS,
        "decoupage": decoupage,
        "n_parents": len({(e["fichier"], e["parent_index"]) for e in enfants}),
        "n_tableaux": sum(1 for e in enfants if e.get("tableau")),
        "morceaux_vides": vides,
        "part_morceaux_vides": round(vides / total, 3) if total else 0.0,
        "extractions_brutes": len(brutes),
        "exigences_apres_fusion": len(exigences),
        "taux_de_doublons": round(1 - len(exigences) / len(brutes), 3) if brutes else 0.0,
        "morceaux_en_echec": echecs,
        "champs_corriges": corrigees,
        "lignes_a_relire": a_relire,
        "morceaux_repris_du_cache": repris,
        "duree_minutes": round(duree / 60, 1),
        "modele": model,
        "par_categorie": dict(sorted(par_categorie.items())),
        "par_statut": dict(sorted(par_statut.items())),
        "exigences": exigences,
    }
    if tronque:
        rapport["avertissement"] = (
            f"Seuls les {total} premiers morceaux ont été analysés "
            f"(--max-chunks). Le compte est INCOMPLET.")
    return rapport


# ===========================================================================
# 5. LIGNE DE COMMANDE
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Extraction exhaustive des exigences d'un appel d'offres.")
    ap.add_argument("--input", required=True, help="dossier du corpus")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="n'analyse que les N premiers morceaux (0 = tous)")
    ap.add_argument("--resume", action="store_true",
                    help=f"repart du cache ({CACHE_FILE}). Le cache est indexe "
                         f"par empreinte du contenu : changer --child-tokens ou "
                         f"--decoupage change les empreintes, donc rien n'est "
                         f"repris a tort. Supprime le fichier pour repartir a zero.")
    ap.add_argument("--sortie", default="exigences_extraites.json")
    ap.add_argument("--child-tokens", type=int, default=CHILD_TOKENS,
                    help=f"taille d'un morceau enfant en tokens (defaut "
                         f"{CHILD_TOKENS}). Mettre 80 pour reproduire la v2.")
    ap.add_argument("--decoupage", default=DECOUPAGE_DEFAUT,
                    choices=list(_STRATEGIES),
                    help="comment construire les parents. « fixe » reproduit le "
                         "comportement v1 : c'est le témoin à battre.")
    args = ap.parse_args()

    chemins = sorted(Path(args.input).iterdir())
    files = [(p.name, p.read_bytes()) for p in chemins if p.is_file()]
    if not files:
        raise SystemExit(f"Aucun fichier dans {args.input}.")

    apercu = decouper(files, args.decoupage, child_tokens=args.child_tokens)
    n_parents = len({(e["fichier"], e["parent_index"]) for e in apercu})
    avec_section = len({e["section"] for e in apercu if e["section"]})
    n_tab = sum(1 for e in apercu if e.get("tableau"))
    print(f"{len(files)} fichier(s) · découpage « {args.decoupage} » · "
          f"{len(apercu)} morceaux enfants ({args.child_tokens} tokens) "
          f"dans {n_parents} parents.")
    if n_tab:
        print(f"   {n_tab} tableau(x) envoyé(s) entiers, sans recoupe, "
              f"avec un prompt dédié.")
    if avec_section:
        print(f"   {avec_section} section(s) de titre reconnue(s) — "
              f"chaque parent porte la sienne en en-tête.")
    elif args.decoupage == "structurel":
        print("   [!] Aucun titre détecté : le découpage structurel est retombé "
              "sur les paragraphes. Compare avec « --decoupage fixe ».")
    if not args.max_chunks:
        print(f"Estimation : ~{len(apercu) * 12 / 60:.0f} à "
              f"{len(apercu) * 27 / 60:.0f} minutes selon la machine.\n")

    def montrer(fait, total):
        print(f"   morceau {fait}/{total}…      ", end="\r")

    r = extraire(files, args.model, args.max_chunks, args.resume, montrer,
                 decoupage=args.decoupage, child_tokens=args.child_tokens)
    print()
    if not r.get("ok"):
        raise SystemExit(f"[ÉCHEC] {r['error']}")

    Path(args.sortie).write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  EXIGENCES TROUVÉES : {r['exigences_apres_fusion']}")
    print(f"{'=' * 60}")
    print(f"  découpage               : {r['decoupage']} "
          f"({r['n_parents']} parents, {r['n_enfants']} enfants)")
    print(f"  extractions brutes      : {r['extractions_brutes']}")
    print(f"  morceaux sans exigence  : {r['morceaux_vides']}/{r['n_enfants']} "
          f"({r['part_morceaux_vides']:.0%})  <- plus c'est bas, mieux la "
          f"taille d'enfant est choisie")
    print(f"  doublons fusionnés      : {r['taux_de_doublons']:.0%}")
    print(f"  morceaux en échec       : {r['morceaux_en_echec']}/{r['n_enfants']}")
    print(f"  durée                   : {r['duree_minutes']} min")

    if r["champs_corriges"]:
        print(f"  champs normalisés       : {r['champs_corriges']} "
              f"(valeur d'origine conservée)")
    if r["lignes_a_relire"]:
        print(f"  lignes « a_relire »     : {r['lignes_a_relire']} "
              f"(catégorie ou statut non reconnus — à lire à la main)")

    print("\n  par catégorie :")
    for cat, n in r["par_categorie"].items():
        print(f"     {cat:<12} {n:>4}")
    print("\n  par statut :")
    for st, n in r["par_statut"].items():
        print(f"     {st:<14} {n:>4}")

    if r.get("avertissement"):
        print(f"\n[!] {r['avertissement']}")

    print(f"\nDétail : {args.sortie}")
    print("Matrice Excel : python export_matrice.py")


if __name__ == "__main__":
    main()