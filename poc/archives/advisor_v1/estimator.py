# -*- coding: utf-8 -*-
"""
estimator.py — Étage 3 : l'ESTIMATEUR (v2).

À partir du profil du corpus et d'une configuration, projette :
  - un temps de TRANSFORMATION (une seule fois : indexation RAG / construction KAG)
  - un temps de REQUÊTE (par question)

Changements par rapport à la v1
-------------------------------
1. LE COÛT D'EXTRACTION EST PILOTÉ PAR LES TOKENS GÉNÉRÉS, PAS LUS. La v1
   divisait le corpus par un « débit d'entrée » de 400 tokens/s et annonçait
   2,8 min pour 37 k tokens — irréaliste d'un ordre de grandeur. En extraction de
   triplets, le LLM LIT vite (prefill) et ÉCRIT lentement : c'est la génération
   des triplets qui domine. On modélise donc explicitement le volume de sortie.

2. AJOUT DU PARSING. Sur des PDF, pdfplumber est souvent plus lent que
   l'embedding. La v1 l'ignorait complètement.

3. Le KAG paie aussi ses embeddings (les nœuds sont vectorisés pour l'entrée
   dans le graphe).

IMPORTANT — honnêteté : ce sont des ESTIMATIONS INDICATIVES. Le temps réel
dépend du matériel (GPU/CPU), du modèle et de la charge. Les constantes ci-dessous
sont regroupées et modifiables : à recalibrer avec quelques mesures réelles sur la
machine cible. Tu disposes déjà de mesures issues de ton POC KAG et des durées
chronométrées par run_sweep.py — ce sont les meilleures valeurs à mettre ici.
"""

from __future__ import annotations

# --- constantes de débit approximatives (À RECALIBRER) ----------------------
# Ordres de grandeur indicatifs pour du local sur GPU grand public.
EMBED_TOKENS_PER_SEC   = 8_000   # débit d'embedding (e5-base ; e5-large ~2x plus lent)
PARSE_TOKENS_PER_SEC   = 20_000  # extraction de texte PDF (pdfplumber)

# Extraction de triplets : modèle génératif, coût dominé par la SORTIE.
LLM_OUTPUT_TOKENS_PER_SEC = 30    # qwen2.5:14b en local, ordre de grandeur
EXTRACT_OUTPUT_RATIO      = 0.50  # tokens de triplets générés par token de corpus lu
LLM_PREFILL_TOKENS_PER_SEC = 1_500  # lecture du prompt (rapide)

GEN_SECONDS_PER_QUERY  = 2.0    # génération de la réponse par le LLM
SEARCH_SECONDS         = 0.05   # recherche vectorielle (négligeable)
RERANK_SEC_PER_CAND    = 0.03   # reranking par candidat (cross-encoder)
GRAPH_QUERY_SECONDS    = 0.4    # traversée du graphe par requête

COMMUNITY_OVERHEAD_MULT = 1.6   # surcoût de construction si communautés
MULTIPASS_MULT          = 1.8   # surcoût si extraction en plusieurs passes
EMBED_LARGE_MULT        = 2.0   # e5-large vs e5-base


def _fmt(seconds: float) -> str:
    """Format lisible d'une durée."""
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"


# Modèles d'embedding « lourds ». La v2 cherchait seulement le mot « large »
# dans le nom, ce qui rendait bge-m3 invisible alors qu'il est de taille
# comparable à e5-large : le temps annoncé était deux fois trop optimiste.
SLOW_EMBED_TAGS = ("large", "bge-m3")


def _embed_seconds(tokens: float, config: dict) -> float:
    rate = EMBED_TOKENS_PER_SEC
    name = (config.get("_embedding_model") or config.get("embedding_model") or "").lower()
    if any(tag in name for tag in SLOW_EMBED_TAGS):
        rate /= EMBED_LARGE_MULT
    return tokens / rate


# --- RAG ---------------------------------------------------------------------
def estimate_rag(corpus: dict, config: dict) -> dict:
    tokens = corpus.get("total_tokens_est", 0)

    # une fois : lecture des fichiers + embedding de tout le corpus
    parsing_s = tokens / PARSE_TOKENS_PER_SEC
    embed_s = _embed_seconds(tokens, config)
    indexing_s = parsing_s + embed_s

    # par requête : recherche (+ reranking) + génération
    query_s = SEARCH_SECONDS + GEN_SECONDS_PER_QUERY
    if config.get("reranker"):
        n_cand = config.get("retrieve_k", config.get("top_k", 5))
        query_s += RERANK_SEC_PER_CAND * n_cand
    if config.get("gen_verification"):
        query_s += GEN_SECONDS_PER_QUERY   # un second appel LLM de vérification

    return {
        "transformation_label": "Indexation (une fois)",
        "transformation_seconds": indexing_s,
        "transformation_human": _fmt(indexing_s),
        "transformation_breakdown": {
            "lecture des fichiers": _fmt(parsing_s),
            "embeddings": _fmt(embed_s),
        },
        "query_seconds": query_s,
        "query_human": _fmt(query_s),
    }


# --- KAG ---------------------------------------------------------------------
def estimate_kag(corpus: dict, config: dict) -> dict:
    tokens = corpus.get("total_tokens_est", 0)

    parsing_s = tokens / PARSE_TOKENS_PER_SEC

    # extraction de triplets : le LLM lit le corpus (prefill) puis ÉCRIT les
    # triplets (génération, lente) — c'est la génération qui domine.
    prefill_s = tokens / LLM_PREFILL_TOKENS_PER_SEC
    generation_s = (tokens * EXTRACT_OUTPUT_RATIO) / LLM_OUTPUT_TOKENS_PER_SEC
    extraction_s = prefill_s + generation_s

    # Le nombre de passes vient de la couche machine du routeur. Le repli sur la
    # phrase française ne sert qu'à rester compatible avec un routeur v3.
    passes = config.get("_extraction_passes")
    if passes is None:
        passes = 2 if "plusieurs passes" in (config.get("extraction_strategy") or "") else 1
    if passes > 1:
        extraction_s *= MULTIPASS_MULT ** (passes - 1)

    embed_s = _embed_seconds(tokens, config)

    construction_s = parsing_s + extraction_s + embed_s
    if config.get("community_detection"):
        construction_s *= COMMUNITY_OVERHEAD_MULT

    # par requête : traversée du graphe (déjà construit -> rapide) + génération
    query_s = GRAPH_QUERY_SECONDS + GEN_SECONDS_PER_QUERY

    return {
        "transformation_label": "Construction du graphe (une fois)",
        "transformation_seconds": construction_s,
        "transformation_human": _fmt(construction_s),
        "transformation_breakdown": {
            "lecture des fichiers": _fmt(parsing_s),
            "extraction de triplets (LLM)": _fmt(extraction_s),
            "embeddings des nœuds": _fmt(embed_s),
        },
        "query_seconds": query_s,
        "query_human": _fmt(query_s),
    }


def estimate(architecture: str, corpus: dict, config: dict) -> dict:
    return estimate_kag(corpus, config) if architecture == "KAG" else estimate_rag(corpus, config)