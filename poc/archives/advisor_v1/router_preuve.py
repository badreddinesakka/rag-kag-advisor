# -*- coding: utf-8 -*-
"""
router_preuve.py — met une étiquette de preuve sur chaque sortie de router.py,
et ajoute une TROISIÈME architecture possible : ne rien indexer du tout.

Pourquoi une troisième sortie
-----------------------------
router.py ne sait répondre que RAG ou KAG. Sur un corpus qui tient entièrement
dans la fenêtre de contexte du LLM, les deux sont de mauvaises réponses : on
découpe, on plonge, on cherche, on recolle — pour reconstituer un texte qu'on
aurait pu donner en entier.

Ce n'est pas une intuition. Sur le corpus RFP (~5 400 tokens), le contexte
complet a obtenu 55 % de rappel contre 40 % pour le RAG. Le système qui
n'indexe rien a battu celui qui indexe.

La vérification est arithmétique et gratuite : elle passe AVANT le routeur.

Usage :
    python router_preuve.py --profile profil.json [--probe sonde.json]
"""

from __future__ import annotations

import json
from pathlib import Path

import router
from decision import contraint, consequence, regle, mesure, resume, config

# --- fenêtre de contexte -----------------------------------------------------
# À changer si le modèle de génération change. Ollama tronque silencieusement
# au-delà de num_ctx : si cette valeur ne correspond pas à ta config Ollama,
# le corpus sera coupé sans avertissement.
MODELE_GENERATION = "qwen2.5:7b"
FENETRE_TOKENS    = 32_768

# On ne remplit jamais la fenêtre à ras bord : il faut de la place pour la
# consigne, la question, et surtout la réponse. Un tiers réservé est prudent.
PART_RESERVEE = 0.35

# Au-delà de cette part du budget, ça passe mais c'est serré : le premier
# document ajouté fera déborder. On le signale au lieu de laisser une surprise.
PART_SERREE = 0.70


def budget_tokens() -> int:
    return int(FENETRE_TOKENS * (1 - PART_RESERVEE))


def tient_dans_le_contexte(c: dict) -> tuple[bool, bool, int, int]:
    """(ça tient, c'est serré, tokens du corpus, budget)."""
    tokens = int(c.get("total_tokens_est", 0) or 0)
    budget = budget_tokens()
    return tokens <= budget, tokens > budget * PART_SERREE, tokens, budget


def _choix_contexte(c: dict, tokens: int, budget: int, serre: bool) -> list:
    """Config quand on n'indexe rien : elle tient en trois lignes."""
    ch = [
        regle("architecture", "CONTEXTE",
              f"{tokens} tokens contre un budget de {budget} : le corpus entier "
              f"tient dans la fenêtre. Découper puis rechercher ne peut que "
              f"perdre de l'information, jamais en ajouter."),
        contraint("generation_model", MODELE_GENERATION,
                  "seul modèle de génération servi par Ollama"),
        contraint("num_ctx", FENETRE_TOKENS,
                  "fenêtre du modèle — Ollama tronque en silence au-delà"),
        regle("decoupage", "aucun",
              "rien à découper : tout le texte part dans le prompt"),
        regle("embedding_model", "aucun",
              "aucun index, donc aucun vecteur à calculer"),
    ]
    if serre:
        ch.append(regle("marge", "SERRÉE",
                        f"{tokens}/{budget} tokens utilisés : un document de plus "
                        f"fait basculer vers le RAG. À surveiller si le corpus grossit."))
    return ch


def _choix_rag(c: dict) -> list:
    return [
        contraint("embedding_model", c["embedding_model"],
                  "seul modèle d'embedding servi par Ollama sur cette machine"),
        contraint("embedding_dim", c["_embedding_dim"],
                  "imposée par le modèle, et exigée par Milvus à la création"),
        contraint("distance", c["_metric_type"],
                  "cosinus, standard pour des vecteurs normalisés"),
        contraint("index_type", c["_index_type"],
                  f"conséquence mécanique du nombre de morceaux (~{c['_n_chunks_est']})"),
        regle("n_chunks_est", c["_n_chunks_est"],
              "estimation du routeur, calculée en tokens — le découpage mesuré "
              "donnera un autre compte"),

        regle("chunk_size", c["chunk_size"],
              "fourchette de la littérature, jamais vérifiée sur ce corpus"),
        regle("chunk_overlap", c["chunk_overlap"],
              "15 % de la taille, usage courant"),
        regle("top_k", c["top_k"],
              "valeur de la littérature — aucune alternative essayée ici"),
        regle("retrieve_k", c["retrieve_k"],
              "recette standard avec reranker — non vérifiée"),
        regle("reranker", c["reranker"] or "aucun",
              "activé sur une règle de langue, jamais comparé aux deux cas"),
        regle("gen_verification", c["gen_verification"],
              "second appel LLM — coût et gain non mesurés"),
    ]


def _choix_kag(c: dict) -> list:
    return [
        contraint("extraction_model", c["_extraction_model"],
                  "seul modèle d'extraction téléchargé dans Ollama"),
        contraint("graph_store", c["graph_store"],
                  "seule base de graphe installée"),

        regle("chunk_size", c["chunk_size"],
              "morceaux courts pour extraire plus d'entités — jamais vérifié"),
        regle("ontology_mode", c["_ontology_mode"],
              "décidé par l'homogénéité du corpus, qui mesure surtout la "
              "régularité des LONGUEURS de documents"),
        regle("entity_resolution", c["_entity_resolution"],
              "seuil sur la part d'entités partagées, non calibré"),
        regle("extraction_passes", c["_extraction_passes"],
              "1 ou 2 selon le volume — le gain d'une 2e passe n'est pas chiffré"),
        regle("community_detection", c["_community_detection"],
              "trois conditions cumulées, aucune vérifiée"),
        regle("limite_triplets", "non fixée",
              "PARAMÈTRE LE PLUS SENSIBLE MESURÉ À CE JOUR (60 triplets → 19 %, "
              "contexte complet → 50 %) et pourtant absent de la config"),
    ]


def dossier_temporaire(files: list[tuple[str, bytes]]) -> str:
    """
    Écrit des (nom, octets) dans un dossier, et rend son chemin.

    chunk_quality et chunker lisent un DOSSIER. Streamlit, lui, ne manipule que
    des octets en mémoire. Sans ce pont, la mesure du découpage serait possible
    en ligne de commande et impossible dans l'interface — la boucle resterait
    ouverte là où l'utilisateur regarde.

    L'appelant est responsable d'effacer le dossier.
    """
    import tempfile
    d = tempfile.mkdtemp(prefix="advisor_corpus_")
    for name, data in files:
        (Path(d) / Path(name).name).write_bytes(data)
    return d


def _choix_decoupage(ch: dict, dossier: str | None, embeddings: bool) -> list:
    """
    Sans --input : on ne peut que PROPOSER. Le découpage reste « réglé ».
    Avec --input : on découpe pour de vrai, chunk_quality note, et le découpage
    devient le premier paramètre « mesuré » de l'Advisor.
    """
    cands = ch["candidates"]
    if dossier is None:
        liste = ", ".join(x["name"] for x in cands)
        return [regle("decoupage", cands[0]["name"],
                      f"candidats proposés mais PAS ENCORE mesurés ({liste}) — "
                      f"relancer avec --input pour trancher")]

    import chunk_quality
    comp = chunk_quality.compare_candidates(
        dossier, cands, use_embeddings=embeddings, verbose=False
    )
    valides = comp.get("resultats") or []
    if not valides:
        return [regle("decoupage", cands[0]["name"],
                      f"mesure impossible : {comp.get('verdict', 'aucun découpage construit')}")]

    # Une seule règle d'égalité pour tout l'Advisor : celle de decision.py.
    # On range donc les candidats du plus simple au plus complexe — c'est cet
    # ordre qui départage quand les notes se tiennent.
    rang = {"fixed": 0, "structural": 1, "semantic": 2}
    valides.sort(key=lambda r: (rang.get(r["strategy"], 9), r["duree_s"]))
    gagne = mesure("decoupage", [(r["name"], r["note"]) for r in valides])
    vainqueur = next(r for r in valides if r["name"] == gagne.valeur)

    # chunk_chars et chunk_strategy ne sont PAS des mesures séparées : ce sont
    # les propriétés du découpage gagnant. En faire des mesures indépendantes
    # affichait des notes fausses quand plusieurs candidats partageaient la
    # même taille (elles s'écrasaient entre elles).
    sortie = [
        gagne,
        consequence("chunk_chars", vainqueur["target_chars"],
                    f"taille du découpage « {vainqueur['name']} », qui a gagné la mesure"),
        consequence("chunk_strategy", vainqueur["strategy"],
                    f"stratégie du découpage gagnant — c'est elle que l'indexation applique"),
        # chunker travaille en CARACTÈRES, router.py annonce chunk_size en TOKENS.
        # Confondre les deux divise la taille réelle par ~4,6.
        consequence("chunk_unit", "caractères",
                  f"le découpage mesuré vaut {vainqueur['target_chars']} CARACTÈRES — "
                  f"c'est cette valeur qui fait foi, pas le chunk_size en tokens"),
    ]
    for r in comp.get("non_mesures", []):
        sortie.append(regle(f"decoupage:{r['name']}", "non mesuré", r["raison"]))
    return sortie



# --------------------------------------------------------------------------
# Reprise des mesures faites par les outils séparés
# --------------------------------------------------------------------------
# embed_compare.py, mesure_topk.py et mesure_reranker.py écrivent chacun leur
# résultat dans un JSON. Sans ce bloc, l'Advisor continuait d'afficher
# « top_k = 5 — NON MESURÉ » alors que top_k venait d'être mesuré : l'outil
# affirmait ne pas savoir ce qu'il savait.
#
# Une mesure vaut pour LE CORPUS sur lequel elle a été faite. On compare donc le
# nombre de morceaux enregistré dans le fichier à celui du corpus courant, et on
# refuse d'appliquer une mesure qui vient visiblement d'ailleurs.

FICHIERS_MESURE = {
    "topk":      "topk_mesure.json",
    "embedding": "embedding_mesure.json",
    "reranker":  "reranker_mesure.json",
}


def _lire(nom: str, dossier: str) -> dict | None:
    f = Path(dossier) / nom
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def charger_mesures(dossier: str = ".", n_morceaux: int | None = None) -> list:
    """Les Choix issus des mesures externes, ou une liste vide."""
    sortie, ignores = [], []

    def bon_corpus(d: dict) -> bool:
        # Tolérance de 10 % : le découpage peut varier légèrement d'un run à
        # l'autre sans que le corpus ait changé.
        if n_morceaux is None or not d.get("n_morceaux"):
            return True
        return abs(d["n_morceaux"] - n_morceaux) <= max(5, 0.1 * n_morceaux)

    d = _lire(FICHIERS_MESURE["topk"], dossier)
    if d and d.get("par_k"):
        if bon_corpus(d):
            import mesure_topk
            sortie += mesure_topk.choix_topk(d)
        else:
            ignores.append(f"top_k ({d['n_morceaux']} morceaux au lieu de {n_morceaux})")

    d = _lire(FICHIERS_MESURE["embedding"], dossier)
    if d and d.get("resultats"):
        import embed_compare
        sortie += embed_compare.choix_embedding(d)

    d = _lire(FICHIERS_MESURE["reranker"], dossier)
    if d and d.get("lignes"):
        if bon_corpus(d):
            import mesure_reranker
            sortie += mesure_reranker.choix_reranker(d)
        else:
            ignores.append(f"reranker ({d['n_morceaux']} morceaux au lieu de {n_morceaux})")

    for i in ignores:
        sortie.append(regle(f"mesure_ignoree:{i.split()[0]}", "écartée",
                            f"une mesure de {i} existe mais porte sur un autre "
                            f"corpus — elle n'est pas appliquée"))
    return sortie


def fusionner(base: list, mesures: list) -> list:
    """
    Une mesure remplace le réglage du même nom.

    Le sens de la fusion n'est jamais l'inverse : une valeur mesurée sur ce
    corpus l'emporte toujours sur une valeur tirée de la littérature.
    """
    par_nom = {c.nom: c for c in mesures}
    fusion, vus = [], set()
    for c in base:
        if c.nom in par_nom:
            fusion.append(par_nom[c.nom])
            vus.add(c.nom)
        else:
            fusion.append(c)
    fusion += [c for c in mesures if c.nom not in vus]
    return fusion


def choix_advisor(corpus: dict, mutability: str = "figé", probe: dict | None = None,
                  dossier: str | None = None, embeddings: bool = True,
                  files: list | None = None) -> list:
    """Décide l'architecture, puis rend la config étiquetée."""
    # Depuis Streamlit on reçoit des octets, pas un dossier : on en fabrique un.
    tmp = None
    if dossier is None and files:
        tmp = dossier = dossier_temporaire(files)
    try:
        return _choix_advisor(corpus, mutability, probe, dossier, embeddings)
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


def _choix_advisor(corpus: dict, mutability: str, probe: dict | None,
                   dossier: str | None, embeddings: bool) -> list:
    tient, serre, tokens, budget = tient_dans_le_contexte(corpus)
    if tient:
        # On s'arrête ici : pas de sondage, pas de graphe, pas d'index.
        return _choix_contexte(corpus, tokens, budget, serre)

    res = router.decide(corpus, mutability=mutability, probe=probe)
    archi = res["architecture"]

    tete = [regle("architecture", archi,
                  f"{tokens} tokens dépassent le budget de {budget} : il faut "
                  f"indexer. Score KAG {res['kag_suitability']}/100 vs seuil "
                  f"{res['kag_threshold']} — pondérations posées à la main, "
                  f"jamais confrontées à un graphe réel")]

    corps = _choix_decoupage(res["chunking"], dossier, embeddings)
    corps += _choix_rag(res["rag"]["versions"]["équilibré"]["config"]) if archi == "RAG" \
        else _choix_kag(res["kag"]["config"])

    # Les mesures externes écrasent les réglages correspondants.
    return fusionner(tete + corps, charger_mesures())



def config_indexation(choix: list) -> dict:
    """
    La config au format attendu par index_rag.index_corpus.

    Traduit les noms lisibles en clés « _ » que l'indexation lit. C'est cette
    fonction qui FERME LA BOUCLE : sans elle, le découpage gagnant reste un
    résultat affiché et l'indexation continue de découper à sa façon.
    """
    v = config(choix)
    sortie = dict(v)
    sortie["_embedding_model"] = v.get("embedding_model")
    sortie["_embedding_dim"]   = v.get("embedding_dim")
    sortie["_index_type"]      = v.get("index_type")
    sortie["_metric_type"]     = v.get("distance")
    sortie["_n_chunks_est"]    = v.get("n_chunks_est")
    if v.get("chunk_strategy") and v.get("chunk_chars"):
        sortie["_chunk_strategy"] = v["chunk_strategy"]
        sortie["_chunk_chars"]    = int(v["chunk_chars"])
    return sortie


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Routeur, avec niveau de preuve.")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--probe")
    ap.add_argument("--mutability", default="figé", choices=["figé", "vivant"])
    ap.add_argument("--input", help="dossier des documents — déclenche la MESURE "
                                    "des découpages (sans lui, ils sont seulement proposés)")
    ap.add_argument("--no-embed", action="store_true",
                    help="mesure sans cohésion ni cohérence (instantané, sans Ollama)")
    ap.add_argument("--out", default="advisor_mesure.json",
                    help="fichier de config écrit pour index_rag.py --config")
    args = ap.parse_args()

    data = router._load_json(args.profile)
    corpus = data.get("corpus", data)
    sonde = router._load_json(args.probe) if args.probe else None

    ch = choix_advisor(corpus, mutability=args.mutability, probe=sonde,
                       dossier=args.input, embeddings=not args.no_embed)
    print(resume(ch))

    n_mes = sum(1 for c in ch if c.statut == "mesuré")
    print(f"\n→ {n_mes} paramètre(s) mesuré(s) sur {len(ch)}.")
    cfg = config_indexation(ch)
    if "_chunk_strategy" in cfg:
        print(f"\nBOUCLE FERMÉE : l'indexation utilisera « {cfg['_chunk_strategy']} » "
              f"à {cfg['_chunk_chars']} caractères.")
    else:
        print("\nBoucle OUVERTE : aucun découpage mesuré, l'indexation découpera "
              "à sa façon. Relancer avec --input.")
    print(cfg)

    # Sans ce fichier, index_rag.py refait tourner le routeur tout seul et
    # n'entend jamais parler de la mesure : la boucle resterait ouverte.
    import json as _json
    Path(args.out).write_text(_json.dumps(cfg, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nConfig écrite dans {args.out} — à passer à "
          f"« python index_rag.py --input <dossier> --config {args.out} --recreate ».")