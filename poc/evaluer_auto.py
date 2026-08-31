# -*- coding: utf-8 -*-
"""
evaluer_auto.py - evaluation automatique des sorties RAG et KAG.

Produit deux chiffres par architecture, SANS liste de reference ecrite a la
main :

  PRECISION   part des exigences produites qui sont reellement enoncees dans
              le document. Un juge LLM relit chaque exigence avec le document
              sous les yeux. Aucune reference necessaire : le document suffit.

  RAPPEL RELATIF  part d'une PSEUDO-REFERENCE retrouvee par chaque
              architecture. La pseudo-reference est construite ici meme, en
              balayant le document du debut a la fin, sans recherche
              vectorielle et sans graphe.

CE QUE CES CHIFFRES VALENT, ET CE QU'ILS NE VALENT PAS
======================================================

LA PRECISION EST SOLIDE. Le juge voit le document entier (5428 tokens, il
tient dans la fenetre) et la phrase a verifier. C'est exactement le travail
fait a la main le 30/08, qui avait trouve « The bidder must use Oracle DB
Release 11.1.0.7 » - une description du parc existant d'Ooredoo transformee en
exigence.

LE RAPPEL EST RELATIF, PAS ABSOLU. C'est la limite a ecrire noir sur blanc
dans le rapport. La pseudo-reference est produite par une machine : elle
contient des erreurs et elle rate des choses. Elle ne dit donc pas combien
d'exigences manquent VRAIMENT.

Ce qu'elle a en revanche, et qui suffit pour comparer : elle est INDEPENDANTE
des deux systemes evalues. Elle ne passe ni par Milvus ni par Neo4j, elle lit
le texte dans l'ordre. Aucune des deux architectures n'est donc avantagee.

L'APPARIEMENT EST APPROXIMATIF, ET C'EST LE POINT FAIBLE. Comparer « The bill
generation solution must produce customer bills in PDF format » et « The bidder
must produce bills in PDF format » revient a comparer des formulations, pas des
sens. Mesure du 30/08 : un seuil a 0,60 comptait 101 exigences KAG comme
absentes du RAG ; l'inspection a la main a montre que la moitie etait la, ecrite
autrement. La vraie valeur etait plutot 20 a 30.
C'est pourquoi ce script donne le rappel a TROIS SEUILS au lieu d'un. Si les
trois chiffres sont proches, la mesure tient. S'ils s'ecartent beaucoup, c'est
que l'appariement gouverne le resultat et il faut le dire.

COUT
Environ 15 appels pour la pseudo-reference, puis un appel par lot de 10
exigences a juger. Sur 167 + 131 exigences, cela fait une trentaine d'appels.
Comptez 30 a 45 minutes en tout. Le cache evite de tout refaire.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

from index_rag import OLLAMA_BASE
from profiler import extract_text
from prompts import PROMPTS

REQUEST_TIMEOUT = 900
NUM_CTX = 16_384

# Taille des morceaux pour construire la pseudo-reference. Volontairement plus
# petits que les 3679 caracteres du KAG : ici on cherche l'exhaustivite, pas la
# preservation des relations. Mesure du 29/08 : plus le morceau est court, plus
# le modele extrait finement.
TAILLE_MORCEAU = 1400
CHEVAUCHEMENT = 200

# Exigences jugees par appel. Le document occupe deja ~5400 tokens du prompt,
# donc on ne peut pas en mettre beaucoup ; 10 tient largement.
LOT_JUGE = 10

# Trois seuils au lieu d'un, pour montrer la sensibilite de l'appariement.
SEUILS = (0.35, 0.45, 0.60)

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
# LE PROMPT DU JUGE
# ---------------------------------------------------------------------------
# Il vit ici et non dans prompts.py : ce n'est pas un prompt d'extraction, il
# ne sert qu'a l'evaluation. Le melanger aux autres inviterait a le modifier
# en meme temps qu'eux, et une mesure dont l'instrument change n'est plus une
# mesure.
PROMPT_JUGE = """You are auditing requirements that were automatically \
extracted from the Request For Proposal below.

For each numbered statement, decide whether the RFP really states it as an \
obligation on the BIDDER.

Answer for each statement with one verdict:
  "supported"     the RFP states this, and it binds the bidder;
  "distorted"     the RFP mentions the topic but the statement changes it \
(wrong figure, wrong actor, an obligation invented from a description);
  "absent"        the RFP does not state this at all.

Be strict about WHO is bound. The RFP also describes Ooredoo's own systems, \
its history, its existing database and its glossary. A statement that turns \
one of those into an obligation on the bidder is "distorted", not "supported".

Answer with JSON only, no commentary:
{"verdicts": [{"n": 1, "verdict": "..."}, {"n": 2, "verdict": "..."}]}

Give one verdict per statement, using the same numbers.

RFP:
\"\"\"
%s
\"\"\"

STATEMENTS:
%s
"""


# ---------------------------------------------------------------------------
# OUTILS
# ---------------------------------------------------------------------------
def _generer(prompt: str, model: str, num_ctx: int = NUM_CTX) -> str:
    payload = {
        "model": model, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 2000},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def _empreinte(texte: str) -> frozenset:
    t = unicodedata.normalize("NFKD", texte.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    mots = re.findall(r"[a-z0-9]+", t)
    return frozenset(m for m in mots if m not in MOTS_IGNORES and len(m) > 1)


def _recouvrement(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lire_document(chemin: str) -> str:
    """Lit le document, qu'on donne un fichier ou un dossier."""
    p = Path(chemin)
    fichiers = [p] if p.is_file() else [f for f in sorted(p.iterdir()) if f.is_file()]
    morceaux = []
    for f in fichiers:
        texte, _, _ = extract_text(f.name, f.read_bytes())
        if texte and texte.strip():
            morceaux.append(texte.strip())
    return "\n\n".join(morceaux)


def decouper(texte: str, taille: int, chevauchement: int) -> list[str]:
    """Decoupage lineaire. Volontairement bete : on veut parcourir tout le
    document dans l'ordre, sans aucune selection."""
    morceaux, i = [], 0
    while i < len(texte):
        morceaux.append(texte[i:i + taille])
        i += max(1, taille - chevauchement)
    return morceaux


# ---------------------------------------------------------------------------
# PSEUDO-REFERENCE
# ---------------------------------------------------------------------------
def construire_reference(texte: str, model: str, cache: Path | None) -> list[str]:
    """
    Balaye le document du debut a la fin et extrait les exigences de chaque
    morceau, avec le meme prompt que le RAG.

    INDEPENDANCE : aucune recherche vectorielle, aucun graphe. C'est ce qui
    rend cette liste utilisable pour comparer les deux architectures, meme si
    elle n'est pas une verite.
    """
    if cache and cache.exists():
        print(f"pseudo-reference reprise de {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))

    morceaux = decouper(texte, TAILLE_MORCEAU, CHEVAUCHEMENT)
    print(f"pseudo-reference : {len(morceaux)} morceaux a balayer")
    modele_prompt = PROMPTS["rag"]
    trouvees = []

    for i, m in enumerate(morceaux, start=1):
        etq = f"R{i}"
        prompt = modele_prompt % (etq, f"[{etq}]\n{m}")
        print(f"  {etq} {i}/{len(morceaux)}…", end=" ", flush=True)
        debut = time.time()
        try:
            brut = _generer(prompt, model)
            donnees = json.loads(brut)
            lignes = next((v for v in donnees.values()
                           if isinstance(v, list)), []) \
                if isinstance(donnees, dict) else donnees
            n = 0
            for l in lignes:
                if isinstance(l, dict):
                    t = str(l.get("requirement") or "").strip()
                    if t:
                        trouvees.append(t)
                        n += 1
            print(f"{n} exigences · {time.time() - debut:.0f}s")
        except Exception as e:
            print(f"ERREUR : {e}")

    # deduplication simple, seuil conservateur comme dans query_rag.py
    uniques, emps = [], []
    for t in trouvees:
        e = _empreinte(t)
        if any(_recouvrement(e, o) >= 0.72 for o in emps):
            continue
        uniques.append(t)
        emps.append(e)

    print(f"pseudo-reference : {len(trouvees)} brutes -> {len(uniques)} uniques")
    if cache:
        cache.write_text(json.dumps(uniques, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return uniques


# ---------------------------------------------------------------------------
# PRECISION
# ---------------------------------------------------------------------------
def juger(exigences: list[str], texte: str, model: str,
          lot: int = LOT_JUGE) -> dict:
    """Fait relire chaque exigence par un juge LLM, document sous les yeux."""
    verdicts = {}
    lots = [exigences[i:i + lot] for i in range(0, len(exigences), lot)]
    print(f"jugement : {len(exigences)} exigences en {len(lots)} lots")

    for k, groupe in enumerate(lots):
        depart = k * lot
        liste = "\n".join(f"{depart + j + 1}. {t}"
                          for j, t in enumerate(groupe))
        prompt = PROMPT_JUGE % (texte, liste)
        print(f"  lot {k + 1}/{len(lots)}…", end=" ", flush=True)
        debut = time.time()
        try:
            brut = _generer(prompt, model)
            donnees = json.loads(brut)
            lignes = next((v for v in donnees.values()
                           if isinstance(v, list)), []) \
                if isinstance(donnees, dict) else donnees
            for l in lignes:
                if isinstance(l, dict) and "n" in l:
                    v = str(l.get("verdict", "")).strip().lower()
                    if v in ("supported", "distorted", "absent"):
                        verdicts[int(l["n"])] = v
            print(f"{time.time() - debut:.0f}s")
        except Exception as e:
            print(f"ERREUR : {e}")

    n = len(exigences)
    compte = {"supported": 0, "distorted": 0, "absent": 0, "non_juge": 0}
    for i in range(1, n + 1):
        compte[verdicts.get(i, "non_juge")] = \
            compte.get(verdicts.get(i, "non_juge"), 0) + 1

    juges = n - compte["non_juge"]
    return {
        "n_exigences": n,
        "n_jugees": juges,
        "verdicts": compte,
        "precision": round(compte["supported"] / juges, 3) if juges else None,
        "detail": {i: verdicts.get(i, "non_juge") for i in range(1, n + 1)},
    }


# ---------------------------------------------------------------------------
# RAPPEL RELATIF
# ---------------------------------------------------------------------------
def rappel(exigences: list[str], reference: list[str]) -> dict:
    """
    Part de la pseudo-reference retrouvee, mesuree a trois seuils.

    Trois seuils et non un : l'appariement compare des formulations, pas des
    sens. Si les trois chiffres sont proches, la mesure tient. S'ils s'ecartent,
    c'est l'appariement qui gouverne le resultat, et il faut le dire plutot que
    de choisir le seuil le plus flatteur.
    """
    ec = [_empreinte(t) for t in exigences]
    res = {}
    for s in SEUILS:
        trouves = sum(1 for r in reference
                      if any(_recouvrement(_empreinte(r), c) >= s for c in ec))
        res[f"seuil_{s}"] = round(trouves / len(reference), 3) if reference else None
    return res


# ---------------------------------------------------------------------------
# PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evalue automatiquement les sorties RAG et KAG.")
    ap.add_argument("--doc", default="../corpus/rfp",
                    help="fichier ou dossier du document de reference")
    ap.add_argument("--fichiers", nargs="+", required=True,
                    help="sorties JSON a evaluer (exigences_rag_v6.json ...)")
    ap.add_argument("--gen-model", default="qwen3:8b")
    ap.add_argument("--cache-reference", default="pseudo_reference.json")
    ap.add_argument("--sans-precision", action="store_true",
                    help="saute le jugement LLM, ne calcule que le rappel")
    ap.add_argument("--out", default="evaluation_auto.json")
    args = ap.parse_args()

    texte = lire_document(args.doc)
    if not texte:
        raise SystemExit(f"[ECHEC] Aucun texte lisible dans {args.doc}")
    print(f"document : {len(texte.split())} mots\n")

    reference = construire_reference(texte, args.gen_model,
                                     Path(args.cache_reference))
    print()

    resultats = {}
    for f in args.fichiers:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        ex = [e["requirement"] for e in d["exigences"]]
        nom = d.get("architecture", f)
        print(f"=== {nom} ({f}) : {len(ex)} exigences ===")

        bloc = {
            "fichier": f,
            "architecture": nom,
            "n_exigences": len(ex),
            "rappel_relatif": rappel(ex, reference),
        }
        if not args.sans_precision:
            bloc["precision"] = juger(ex, texte, args.gen_model)
        resultats[f] = bloc
        print()

    sortie = {
        "document": args.doc,
        "modele_juge": args.gen_model,
        "n_pseudo_reference": len(reference),
        "seuils_appariement": list(SEUILS),
        "avertissement": (
            "Le rappel est RELATIF a une pseudo-reference produite par machine. "
            "Il ne dit pas combien d'exigences manquent vraiment, seulement "
            "laquelle des deux architectures en retrouve le plus. La precision, "
            "elle, est mesuree contre le document lui-meme."),
        "resultats": resultats,
    }
    Path(args.out).write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"pseudo-reference : {len(reference)} exigences\n")
    entete = f"{'architecture':<14}{'n':>5}{'precision':>11}"
    for s in SEUILS:
        entete += f"{'rappel@' + str(s):>13}"
    print(entete)
    for b in resultats.values():
        ligne = f"{b['architecture']:<14}{b['n_exigences']:>5}"
        p = b.get("precision", {}).get("precision")
        ligne += f"{p:>11.0%}" if p is not None else f"{'—':>11}"
        for s in SEUILS:
            v = b["rappel_relatif"][f"seuil_{s}"]
            ligne += f"{v:>13.0%}" if v is not None else f"{'—':>13}"
        print(ligne)

    for b in resultats.values():
        if "precision" in b:
            v = b["precision"]["verdicts"]
            print(f"\n{b['architecture']} — verdicts du juge : "
                  f"{v['supported']} conformes, {v['distorted']} deformees, "
                  f"{v['absent']} absentes, {v['non_juge']} non jugees")

    print(f"\nsortie : {args.out}")
    print("[!] Le rappel est RELATIF : la pseudo-reference vient d'une machine. "
          "A ecrire tel quel dans le rapport.")


if __name__ == "__main__":
    main()
