# -*- coding: utf-8 -*-
"""
collecter.py — UN FICHIER JSON PAR SYSTEME, depuis les resultats DEJA obtenus.

AUCUN appel LLM. AUCUN recalcul. Quelques secondes.

Il lit les reponses brutes deja ecrites par run_benchmark.py
(<systeme>_<prompt>_run<N>.txt), les evalue contre la liste de reference, et
ecrit :

    resultats_rag.json
    resultats_kag.json
    resultats_baseline.json

POURQUOI REPARTIR DES .txt ET PAS DE resultats.json
---------------------------------------------------
run_benchmark.py reecrit resultats.json a chaque lancement : relancer le KAG
seul efface les metadonnees du RAG et de la ligne de base. Les reponses brutes,
elles, portent le nom du systeme et survivent a tout.

Quand resultats.json existe encore, on y recupere la duree et la taille du
contexte. Sinon ces champs valent null — jamais une valeur inventee.

Utilisation :
    python collecter.py
    python collecter.py --dossiers resultats_benchmark resultats_kag_complet
    python collecter.py --prefixe mesures_
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import evaluate

_NOM = re.compile(r"^(?P<sys>[a-z]+)_(?P<prompt>[a-z0-9_]+?)_run(?P<run>\d+)\.txt$",
                  re.IGNORECASE)


def _meta(dossier: Path) -> dict[str, dict]:
    """Duree et contexte par systeme, si resultats.json n'a pas ete ecrase."""
    chemin = dossier / "resultats.json"
    if not chemin.exists():
        return {}
    try:
        data = json.loads(chemin.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for c in data.get("cases", []):
        runs = [r for r in c.get("runs", []) if r.get("ok")]
        out[str(c.get("case", "")).upper()] = {
            "contexte_mots": c.get("contexte_mots"),
            "n_morceaux": c.get("n_morceaux"),
            "n_triplets": c.get("n_triplets"),
            "parametres": data.get("parametres", {}),
            "durees": {r.get("run"): r.get("duree_s") for r in runs},
        }
    return out


def collecter(dossiers: list[Path], reference: list[dict]) -> dict[str, dict]:
    par_systeme: dict[str, list[dict]] = {}

    for dossier in dossiers:
        if not dossier.is_dir():
            continue
        meta = _meta(dossier)
        for fichier in sorted(dossier.glob("*.txt")):
            m = _NOM.match(fichier.name)
            if not m:
                continue
            texte = fichier.read_text(encoding="utf-8", errors="ignore")
            if not texte.strip():
                continue

            systeme = m.group("sys").upper()
            run = int(m.group("run"))
            info = meta.get(systeme, {})
            note = evaluate.evaluer(texte, reference)

            par_systeme.setdefault(systeme, []).append({
                "run": run,
                "source": dossier.name,
                "fichier": fichier.name,
                "prompt": m.group("prompt"),
                "duree_s": (info.get("durees") or {}).get(run),
                "contexte_mots": info.get("contexte_mots"),
                "n_morceaux": info.get("n_morceaux"),
                "n_triplets": info.get("n_triplets"),
                "parametres": info.get("parametres"),
                "rappel": note["rappel"],
                "n_trouves": note["n_trouves"],
                "n_reference": note["n_reference"],
                "precision": note["precision"],
                "n_annonces": note["n_annonces"],
                "n_rapproches": note["n_rapproches"],
                "n_a_verifier": len(note["a_verifier"]),
                "ids_trouves": note["ids_trouves"],
                "ids_manques": note["ids_manques"],
                "a_verifier": note["a_verifier"],
                "rappel_par_categorie": evaluate.resume_par_categorie(texte, reference),
                "reponse_est_json": note["reponse_json"],
            })

    return {s: _bloc(s, runs, reference) for s, runs in sorted(par_systeme.items())}


def _bloc(systeme: str, runs: list[dict], reference: list[dict]) -> dict:
    runs.sort(key=lambda r: (r["source"], r["run"]))

    rappels = [r["rappel"] for r in runs]
    precisions = [r["precision"] for r in runs if r["precision"] is not None]
    annonces = [r["n_annonces"] for r in runs if r["n_annonces"] is not None]
    durees = [r["duree_s"] for r in runs if r["duree_s"]]

    # Une duree tres superieure aux autres n'est pas une mesure mais une mise en
    # veille pendant le calcul. On compare au MINIMUM : avec deux runs dont un
    # aberrant, la mediane est deja contaminee.
    ecartees = []
    if len(durees) >= 2:
        plancher = min(durees)
        ecartees = [d for d in durees if d > plancher * 5]
        durees = [d for d in durees if d <= plancher * 5]

    cats: dict[str, list[float]] = {}
    for r in runs:
        for cat, v in r["rappel_par_categorie"].items():
            cats.setdefault(cat, []).append(v["rappel"])

    # Manques SYSTEMATIQUES : rates par tous les runs. Un critere rate une fois
    # sur trois releve du bruit de generation ; rate a chaque fois, c'est un
    # echec reel — le seul qui apprenne quelque chose.
    manques = set(runs[0]["ids_manques"])
    trouves_parfois: set[str] = set()
    for r in runs:
        manques &= set(r["ids_manques"])
        trouves_parfois |= set(r["ids_trouves"])

    etendue = round(max(rappels) - min(rappels), 3)

    return {
        "systeme": systeme,
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "n_criteres_reference": len(reference),
        "sources": sorted({r["source"] for r in runs}),

        "synthese": {
            "n_runs": len(runs),
            "rappel_moyen": round(statistics.mean(rappels), 3),
            "rappel_min": min(rappels),
            "rappel_max": max(rappels),
            "rappel_etendue": etendue,
            "conclut": etendue <= 0.10,
            "precision_moyenne": round(statistics.mean(precisions), 3) if precisions else None,
            "annonces_moyen": round(statistics.mean(annonces)) if annonces else None,
            "duree_moyenne_s": round(statistics.mean(durees)) if durees else None,
            "durees_ecartees_s": ecartees,
            "contexte_mots": runs[0]["contexte_mots"],
        },

        "rappel_par_categorie": {c: round(statistics.mean(v), 3)
                                 for c, v in sorted(cats.items())},

        "criteres": {
            "manques_par_tous_les_runs": sorted(manques),
            "trouves_au_moins_une_fois": sorted(trouves_parfois),
            "instables": sorted(trouves_parfois & set(runs[0]["ids_manques"])),
        },

        "avertissements": _avertissements(etendue, ecartees, runs),
        "runs": runs,
    }


def _avertissements(etendue: float, ecartees: list, runs: list[dict]) -> list[str]:
    out = []
    if etendue > 0.10:
        out.append(f"Étendue de {etendue:.0%} entre le pire et le meilleur run : "
                   f"cette case ne conclut rien. Citer l'étendue, pas la moyenne.")
    if ecartees:
        out.append(f"Durée(s) écartée(s) : "
                   f"{', '.join(f'{d:.0f} s' for d in ecartees)} — plus de cinq "
                   f"fois la plus courte, machine en veille pendant la mesure.")
    if any(r["duree_s"] is None for r in runs):
        out.append("Durée absente pour au moins un run : resultats.json a été "
                   "écrasé par un lancement ultérieur. Non reconstituable, non "
                   "inventée.")
    if any(not r["reponse_est_json"] for r in runs):
        out.append("Au moins une réponse n'est pas du JSON exploitable : sa "
                   "précision n'est pas mesurable, seul son rappel l'est.")
    contextes = {r["contexte_mots"] for r in runs if r["contexte_mots"]}
    if len(contextes) > 1:
        out.append(f"Tailles de contexte différentes entre runs "
                   f"({sorted(contextes)}) : ces runs ne sont pas comparables "
                   f"entre eux sans le préciser.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rassemble les résultats déjà obtenus en un JSON par système.")
    ap.add_argument("--dossiers", nargs="*", default=None,
                    help="dossiers de résultats (défaut : tous les resultats*)")
    ap.add_argument("--reference", default="criteres_reference.json")
    ap.add_argument("--prefixe", default="resultats_")
    args = ap.parse_args()

    reference = evaluate.charger_reference(args.reference)

    if args.dossiers:
        dossiers = [Path(d) for d in args.dossiers]
    else:
        dossiers = sorted(p for p in Path(".").iterdir()
                          if p.is_dir() and p.name.startswith("resultats"))
    if not dossiers:
        raise SystemExit("Aucun dossier de résultats trouvé.")

    blocs = collecter(dossiers, reference)
    if not blocs:
        raise SystemExit("Aucune réponse brute (<systeme>_<prompt>_run<N>.txt) "
                         "dans " + ", ".join(d.name for d in dossiers))

    print(f"Référence : {len(reference)} critères · dossiers lus : "
          f"{', '.join(d.name for d in dossiers)}\n")

    for systeme, bloc in blocs.items():
        chemin = Path(f"{args.prefixe}{systeme.lower()}.json")
        chemin.write_text(json.dumps(bloc, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        s = bloc["synthese"]
        prec = (f"{s['precision_moyenne']:.0%}"
                if s["precision_moyenne"] is not None else "n/a")
        print(f"  {chemin.name:<26} {s['n_runs']} run(s) · "
              f"rappel {s['rappel_moyen']:.0%} "
              f"({s['rappel_min']:.0%}–{s['rappel_max']:.0%}) · précision {prec}")
        for a in bloc["avertissements"]:
            print(f"      ! {a}")


if __name__ == "__main__":
    main()
