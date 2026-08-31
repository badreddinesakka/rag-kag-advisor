# -*- coding: utf-8 -*-
"""
run_benchmark.py — le banc d'essai complet.

Trois systèmes, un seul prompt (v2) :

    RAG        recherche vectorielle dans Milvus
    KAG        traversée du graphe Neo4j
    BASELINE   tout le document dans le contexte, sans recherche

Le prompt est STRICTEMENT identique pour les trois : seule change la façon
dont le contexte a été récupéré. C'est exactement ce qu'on cherche à mesurer.

Pour comparer deux stratégies d'INDEXATION plutôt que deux architectures,
construire deux collections et lancer le banc d'essai sur chacune :

    python index_rag.py --input dossier --recreate --flat --collection rag_v1
    python index_rag.py --input dossier --recreate        --collection rag_v2
    python run_benchmark.py --input dossier --collection rag_v1 --top-k 4 \
                            --sans-kag --sans-baseline
    python run_benchmark.py --input dossier --collection rag_v2 --top-k 4 \
                            --sans-kag --sans-baseline

Le --top-k 4 est essentiel : avec 13 passages dans l'index et top_k 12, le RAG
récupère presque tout le document et ne sélectionne rien. La stratégie de
découpage n'a alors aucun effet mesurable — mesuré, et c'est un piège.

Chaque case est lancée plusieurs fois (3 par défaut). Même à température 0,
Ollama n'est pas parfaitement déterministe. Si les runs d'une même case
divergent beaucoup, cette case ne conclut rien — et le savoir vaut mieux que
d'afficher un chiffre unique.

Utilisation :
    python run_benchmark.py --input dossier_rfp
    python run_benchmark.py --input dossier_rfp --runs 5 --sans-kag
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import evaluate
import query_baseline
import query_rag

# Le prompt court est retiré (v2) : il ne produit pas de liste identifiable,
# donc sa précision n'est pas mesurable. Une case sur deux du tableau restait
# vide, pour le double du temps de calcul.
PROMPTS_TESTES = ["detaille"]


def _ecart(valeurs: list[float]) -> float:
    return round(max(valeurs) - min(valeurs), 3) if len(valeurs) > 1 else 0.0


def _moyenne(valeurs: list[float]) -> float:
    return round(statistics.mean(valeurs), 3) if valeurs else 0.0


def lancer_case(nom: str, fonction, nom_prompt: str, reference: list[dict],
                runs: int, dossier_sortie: Path) -> dict:
    """Lance une case du tableau `runs` fois et agrège les mesures."""
    print(f"\n=== {nom} · prompt {nom_prompt} ===")
    resultats = []

    for i in range(runs):
        print(f"  run {i + 1}/{runs}…", end=" ", flush=True)
        debut = time.time()
        try:
            sortie = fonction(nom_prompt)
        except Exception as e:
            print(f"ERREUR : {e}")
            resultats.append({"ok": False, "error": str(e)})
            continue
        duree = time.time() - debut

        if not sortie.get("ok"):
            print(f"ÉCHEC : {sortie.get('error')}")
            resultats.append(sortie)
            continue

        note = evaluate.evaluer(sortie["reponse"], reference)
        note["par_categorie"] = evaluate.resume_par_categorie(
            sortie["reponse"], reference)

        # La réponse brute est conservée : c'est la pièce justificative du
        # rapport. Sans elle, les chiffres ne sont pas vérifiables.
        fichier = dossier_sortie / f"{nom.lower()}_{nom_prompt}_run{i + 1}.txt"
        fichier.write_text(sortie["reponse"], encoding="utf-8")

        resultats.append({
            "ok": True,
            "run": i + 1,
            "duree_s": round(duree, 1),
            "fichier_reponse": fichier.name,
            **{k: v for k, v in sortie.items() if k != "reponse"},
            "evaluation": note,
        })
        precision = note["precision"]
        print(f"rappel {note['rappel']:.0%}"
              + (f" · précision {precision:.0%}" if precision is not None else "")
              + f" · {duree:.0f}s")

    reussis = [r for r in resultats if r.get("ok")]
    if not reussis:
        return {"case": nom, "prompt": nom_prompt, "ok": False,
                "runs": resultats}

    rappels = [r["evaluation"]["rappel"] for r in reussis]
    precisions = [r["evaluation"]["precision"] for r in reussis
                  if r["evaluation"]["precision"] is not None]
    # Nombre de lignes produites : dit si un rappel faible vient d'un système
    # bavard mais imprécis, ou d'un système avare mais juste.
    annonces = [r["evaluation"]["n_annonces"] for r in reussis
                if r["evaluation"]["n_annonces"] is not None]

    return {
        "case": nom,
        "prompt": nom_prompt,
        "ok": True,
        "n_runs_reussis": len(reussis),
        "rappel_moyen": _moyenne(rappels),
        "rappel_ecart": _ecart(rappels),
        "precision_moyenne": _moyenne(precisions) if precisions else None,
        "precision_ecart": _ecart(precisions) if precisions else None,
        "annonces_moyen": round(_moyenne(annonces)) if annonces else None,
        "duree_moyenne_s": _moyenne([r["duree_s"] for r in reussis]),
        "contexte_mots": reussis[0].get("contexte_mots"),
        "n_morceaux": reussis[0].get("n_morceaux"),
        "n_triplets": reussis[0].get("n_triplets"),
        "runs": resultats,
    }


def tableau_markdown(cases: list[dict], reference: list[dict]) -> str:
    """Le tableau à coller dans le rapport."""
    lignes = [
        "# Banc d'essai — extraction de critères d'un appel d'offres",
        "",
        f"Généré le {datetime.now():%d/%m/%Y à %H:%M}. "
        f"Liste de référence : {len(reference)} critères.",
        "",
        "## Résultats",
        "",
        "| Système | Rappel moyen | Écart entre runs | Précision | "
        "Annoncés | Contexte (mots) | Durée moy. |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        if not c.get("ok"):
            lignes.append(f"| {c['case']} | échec | — | — | — | — | — |")
            continue
        prec = (f"{c['precision_moyenne']:.0%}"
                if c["precision_moyenne"] is not None else "n/a")
        annonces = c.get("annonces_moyen")
        lignes.append(
            f"| {c['case']} | {c['rappel_moyen']:.0%} | "
            f"±{c['rappel_ecart']:.0%} | {prec} | "
            f"{annonces if annonces is not None else '—'} | "
            f"{c['contexte_mots']} | {c['duree_moyenne_s']:.0f}s |")

    lignes += [
        "",
        "## Comment lire ce tableau",
        "",
        "- **Rappel** : part des critères de la liste de référence retrouvés "
        "dans la réponse. C'est la mesure principale.",
        "- **Écart entre runs** : différence entre le meilleur et le pire run. "
        "Un écart large veut dire que la case ne conclut rien.",
        "- **Précision** : part des critères annoncés qui correspondent à un "
        "critère réel de la liste de référence.",
        "- **Annoncés** : nombre de lignes produites. Un rappel faible avec "
        "beaucoup de lignes annoncées signale un système bavard et imprécis ; "
        "avec peu de lignes, un système avare mais juste.",
        "- **Contexte (mots)** : quantité d'information vue par le système. "
        "Si deux systèmes voient des quantités très différentes, la "
        "comparaison n'est pas à armes égales et il faut le dire.",
        "",
        "## Coût, à ne pas oublier",
        "",
        "| Système | Appels LLM à la construction | Appels LLM par question |",
        "|---|---|---|",
        "| RAG | 0 | 1 |",
        "| KAG | 1 par morceau du corpus | 1 |",
        "| Ligne de base | 0 | 1 |",
        "",
        "Si le KAG répond mieux, ce n'est pas forcément le graphe qui est "
        "meilleur : c'est peut-être simplement que le LLM a déjà lu tout le "
        "corpus une fois, à la construction. Le RAG, lui, n'en voit que "
        "quelques morceaux au moment de la question.",
        "",
        "## Ce qu'il reste à faire à la main",
        "",
        "Les lignes « à vérifier » du fichier JSON ne sont PAS des inventions "
        "prouvées. Chacune est soit une invention, soit un critère absent de "
        "la liste de référence, soit une reformulation que les mots-clés "
        "n'ont pas reconnue. Il faut les lire une par une avant de conclure "
        "sur la précision.",
    ]
    return "\n".join(lignes)


def main():
    ap = argparse.ArgumentParser(description="Banc d'essai RAG / KAG / baseline.")
    ap.add_argument("--input", required=True, help="dossier du corpus")
    ap.add_argument("--reference", default="criteres_reference.json")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--collection", default="advisor_rag")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--gen-model", default="qwen2.5:7b")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--retrieve-k", type=int, default=5)
    ap.add_argument("--limite-triplets", type=int, default=400)
    ap.add_argument("--num-ctx", type=int, default=16_384)
    ap.add_argument("--sortie", default="resultats_benchmark")
    ap.add_argument("--sans-rag", action="store_true")
    ap.add_argument("--sans-kag", action="store_true")
    ap.add_argument("--sans-baseline", action="store_true")
    args = ap.parse_args()

    reference = evaluate.charger_reference(args.reference)
    dossier = Path(args.sortie)
    dossier.mkdir(exist_ok=True)

    print(f"Liste de référence : {len(reference)} critères.")
    print(f"{args.runs} run(s) par case. Résultats dans {dossier}/")

    cases = []

    if not args.sans_rag:
        def rag(p):
            return query_rag.repondre(p, args.collection, args.embed_model,
                                      args.gen_model, args.top_k,
                                      args.retrieve_k, args.num_ctx)
        for p in PROMPTS_TESTES:
            cases.append(lancer_case("RAG", rag, p, reference, args.runs, dossier))

    if not args.sans_kag:
        # Import tardif : sans le paquet neo4j installé, le reste du banc
        # d'essai doit quand même tourner.
        try:
            import query_kag

            def kag(p):
                return query_kag.repondre(p, args.gen_model,
                                          args.limite_triplets,
                                          num_ctx=args.num_ctx)
            for p in PROMPTS_TESTES:
                cases.append(lancer_case("KAG", kag, p, reference, args.runs, dossier))
        except Exception as e:
            print(f"\n[!] KAG indisponible : {e}")

    if not args.sans_baseline:
        def base(p):
            return query_baseline.repondre(args.input, p, args.gen_model,
                                           args.num_ctx)
        for p in PROMPTS_TESTES:
            cases.append(lancer_case("BASELINE", base, p, reference,
                                     args.runs, dossier))

    # --- écriture ----------------------------------------------------------
    (dossier / "resultats.json").write_text(
        json.dumps({"reference": args.reference,
                    "n_criteres_reference": len(reference),
                    "parametres": vars(args),
                    "cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    rapport = tableau_markdown(cases, reference)
    (dossier / "rapport.md").write_text(rapport, encoding="utf-8")

    print("\n" + "=" * 70)
    print(rapport)
    print("=" * 70)
    print(f"\nDétail complet : {dossier}/resultats.json")
    print(f"Tableau à coller : {dossier}/rapport.md")
    print(f"Réponses brutes : {dossier}/*.txt")


if __name__ == "__main__":
    main()
