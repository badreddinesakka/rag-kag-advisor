# -*- coding: utf-8 -*-
"""
estimator.py — projection de durée. LE SEUL FICHIER QUI DÉPEND DE LA MACHINE.

À LIRE AVANT DE CROIRE UN CHIFFRE D'ICI
=======================================
Tout le reste de l'Advisor ne décide qu'à partir du CORPUS : il ne nomme aucun
modèle, ne suppose aucune fenêtre de contexte, ne connaît aucun matériel. Ce
fichier est l'exception, et il l'assume : un temps de calcul ne se déduit pas
d'un texte, il dépend de la carte graphique, du modèle et de la charge.

Les constantes ci-dessous ont donc le statut le plus faible de tout le projet :
elles ne sont NI mesurées NI universelles. Elles servent à donner un ordre de
grandeur, pas un engagement.

CE QUI A ÉTÉ RECALIBRÉ (30/08), et ce que ça dit
------------------------------------------------
La v2 annonçait des MINUTES pour une construction KAG qui en a pris des HEURES.
Mesures réelles sur le RFP (11 morceaux de 3 679 caractères, RTX A2000 4 Go) :

    qwen2.5:7b    79 s par morceau
    mistral:7b   110 s par morceau
    qwen3:8b     157 s par morceau

Soit un facteur DEUX entre deux modèles de taille comparable, sur le même
matériel et le même texte. Aucune formule ne prédit cela.

LA SEULE ESTIMATION FIABLE reste celle que l'on mesure : lancer trois morceaux
avec --max-chunks 3, chronométrer, multiplier. C'est exactement ce que fait
index_kag.py en affichant les secondes par morceau à la fin de chaque run.
Ce fichier ne remplace pas cette mesure, il la précède.
"""

from __future__ import annotations

# --- constantes de débit approximatives (À RECALIBRER) ----------------------
# Ordres de grandeur indicatifs pour du local sur GPU grand public.
EMBED_TOKENS_PER_SEC   = 8_000   # débit d'embedding (e5-base ; e5-large ~2x plus lent)
PARSE_TOKENS_PER_SEC   = 20_000  # extraction de texte PDF (pdfplumber)

# Extraction de triplets : modèle génératif, coût dominé par la SORTIE.
# Recalibré le 30/08 : 11 morceaux de 3 679 caractères en 14,5 min avec
# qwen2.5:7b, soit ~79 s par morceau. La v2 posait 30 tokens/s et
# annonçait des minutes là où il fallait des heures.
LLM_OUTPUT_TOKENS_PER_SEC = 12    # mesuré, pas supposé
EXTRACT_OUTPUT_RATIO      = 0.50  # tokens de triplets générés par token lu
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
# Repérage par taille, pas par nom de produit : un modèle dont le nom
# contient « large », « m3 » ou « xl » est lourd, quel que soit l'éditeur.
SLOW_EMBED_TAGS = ("large", "m3", "xl", "7b", "8b")


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