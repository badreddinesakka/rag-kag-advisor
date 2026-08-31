# -*- coding: utf-8 -*-
"""
router.py — Étage 2 : le ROUTEUR (v4).

Décide RAG ou KAG à partir de la STRUCTURE du corpus, jamais du type de question.

Changement de la v4 : DEUX COUCHES DE SORTIE
--------------------------------------------
Jusqu'ici les configs étaient des phrases françaises ("LLM fort, résolution
soignée, plusieurs passes"). Lisibles par un humain, inutilisables par du code :
l'Advisor DÉCRIVAIT une config sans la PRODUIRE.

Chaque config expose désormais :
  - des clés lisibles, inchangées, pour l'interface et le rapport ;
  - des clés préfixées « _ », valeurs machine (booléens, entiers, listes),
    consommées par rag_milvus.py et kag_ingest.py.

`render_config_table` filtre déjà les clés « _ » : l'interface ne change pas.
Règle : on décide en valeurs machine, puis on rédige la phrase à partir d'elles.
Jamais l'inverse, sinon le code doit parser du français.

Changements par rapport à la v2
-------------------------------
1. LA DÉCISION REPOSE SUR LE SONDAGE LLM (probe.py), pas sur le comptage spaCy.
   Le comptage d'entités s'est révélé trop bruité : « Board », « EBITDA » et
   « QAR » étaient comptés comme des entités, et des pays alignés dans un
   tableau comme des entités liées. Le sondage mesure les vraies relations.

2. PLUS DE SATURATION. En v2, les quatre signaux étaient plafonnés à 1,00 : sur
   le corpus Ooredoo ils l'étaient tous, donc le score ne dépendait plus que de
   la taille. La nouvelle formule s = x / (x + seuil) vaut 0,5 au seuil, monte
   sans jamais atteindre 1, et distingue donc « au seuil » de « très au-dessus ».

3. ORDRE DES ÉTAPES. Les mesures gratuites tranchent d'abord (un corpus de
   5 000 mots ne justifie jamais un graphe : inutile de réveiller le LLM). Le
   sondage n'est lancé que si le doute subsiste.

4. REPLI. Sans Ollama, on retombe sur les signaux spaCy — mais en mode prudent :
   ils sont peu fiables, donc le seuil KAG est relevé.

Les SEUILS restent des défauts d'ingénierie à calibrer.
"""

from __future__ import annotations

# --- seuils du sondage LLM (défauts calibrables) ----------------------------
RELATIONS_PER_1000_KAG = 8.0    # relations réelles pour 1000 tokens
CROSS_DOC_SHARE_KAG    = 0.25   # part d'entités présentes dans >= 2 documents
REUSE_KAG              = 0.30   # part des relations dont le type est réutilisé

W_RELATIONS = 0.45
W_CROSS_DOC = 0.35
W_REUSE     = 0.20

# --- seuils de volume --------------------------------------------------------
HARD_MIN_WORDS_KAG = 15_000    # en dessous : KAG exclu, on ne sonde même pas
FULL_AMORT_WORDS   = 100_000   # au-dessus : le coût de construction est amorti
MIN_AMORT_FACTOR   = 0.40

# --- seuils de décision ------------------------------------------------------
KAG_THRESHOLD_STATIC   = 50
KAG_THRESHOLD_LIVE     = 80
KAG_THRESHOLD_FALLBACK = 75    # sans sondage : on exige beaucoup plus

# --- repli sur les signaux spaCy (peu fiables) ------------------------------
DEGREE_KAG       = 3.0
CONNECTIVITY_KAG = 0.30

# --- modèles réellement servis par Ollama -----------------------------------
# L'indexation passe par Ollama, qui tourne sur la machine hôte et utilise sa
# carte graphique. Le routeur ne doit donc jamais recommander un modèle
# qu'Ollama ne sait pas servir : la configuration serait jolie mais
# inapplicable. Ces constantes sont le seul endroit à changer si la liste des
# modèles téléchargés évolue.
EMBEDDING_MODEL = "bge-m3"   # multilingue, 1024 dimensions, sert FR et EN
EMBEDDING_DIM   = 1024       # Milvus exige la taille du vecteur à la création

# Deux constantes distinctes pour deux rôles, même si elles valent aujourd'hui
# la même chose : seul qwen2.5:7b est téléchargé. Le jour où qwen2.5:14b est
# récupéré (`ollama pull qwen2.5:14b`), remplacer la ligne HEAVY suffit à
# rendre la distinction figé/vivant réelle.
EXTRACTION_MODEL_HEAVY = "qwen2.5:7b"   # corpus figé : on pourrait se permettre plus lourd
EXTRACTION_MODEL_LIGHT = "qwen2.5:7b"   # corpus vivant : reconstruit souvent, donc léger

# --- seuils de configuration RAG --------------------------------------------
SHORT_DOC_WORDS      = 400
LARGE_CORPUS_CHUNKS  = 50_000
HOMOGENEITY_ONTOLOGY = 0.60

def _n(x) -> str:
    return f"{x:,}".replace(",", " ")
def _saturating(value: float, threshold: float) -> float:
    """
    Convertit une mesure en note 0-1 sans plafond brutal.
    Vaut 0,5 exactement au seuil ; 0,75 à trois fois le seuil ; ne vaut jamais 1.
    """
    value = max(0.0, value or 0.0)
    total = value + threshold
    return value / total if total else 0.0


def _amortization(words: int) -> float:
    ratio = min(words / FULL_AMORT_WORDS, 1.0)
    return MIN_AMORT_FACTOR + (1 - MIN_AMORT_FACTOR) * ratio


def needs_probe(corpus: dict) -> bool:
    """Un corpus trop petit ne justifie jamais un graphe : inutile d'appeler le LLM."""
    return (corpus.get("total_words", 0) or 0) >= HARD_MIN_WORDS_KAG


# --- score fondé sur le sondage LLM -----------------------------------------
def _kag_from_probe(corpus: dict, probe: dict) -> tuple[int, list[str]]:
    words = corpus.get("total_words", 0) or 0

    rel_density = probe.get("relations_per_1000_tokens", 0.0)
    cross_doc   = probe.get("cross_doc_entity_share", 0.0)
    reuse       = probe.get("relation_reuse", 0.0)

    s_rel   = _saturating(rel_density, RELATIONS_PER_1000_KAG)
    s_cross = _saturating(cross_doc, CROSS_DOC_SHARE_KAG)
    s_reuse = _saturating(reuse, REUSE_KAG)

    structural = W_RELATIONS * s_rel + W_CROSS_DOC * s_cross + W_REUSE * s_reuse
    amort = _amortization(words)
    score = round(100 * structural * amort)

    couverts = probe.get("docs_covered", 0)
    total_docs = corpus.get("n_docs", 0) or 0
    taux_ecartes = probe.get("unverified_rate", 0) or 0

    # Le commentaire sur les relations écartées était affiché à l'identique quelle
    # que soit la valeur : il annonçait « un taux faible confirme… » même à 41 %.
    # Une explication qui ne dépend pas de la mesure n'explique rien.
    if taux_ecartes <= 0.20:
        note_ecartes = ("taux faible : le LLM extrait bien des relations réellement "
                        "présentes dans le texte.")
    else:
        note_ecartes = ("taux ÉLEVÉ. Attention à l'interprétation : ce compteur "
                        "additionne deux causes distinctes — les relations dont le "
                        "sujet ou l'objet ne se retrouve pas dans le texte "
                        "(invention), et celles écartées par les filtres de format "
                        "(nombre, montant, sujet de plus de 6 mots). Il ne mesure "
                        "donc pas l'hallucination à lui seul.")

    reasons = [
        f"Le LLM a lu {probe.get('chunks_sampled', 0)} morceaux de texte répartis sur "
        f"{couverts} documents "
        f"({probe.get('chunks_per_doc_avg', 0)} morceaux par document en moyenne) "
        f"et y a trouvé {probe.get('relations_kept', 0)} relations réelles.",
        f"Densité de relations : {rel_density}/1000 tokens (seuil {RELATIONS_PER_1000_KAG}) — "
        f"mesure directe de ce que contiendrait le graphe.",
        f"Entités présentes dans plusieurs documents : {cross_doc:.0%} "
        f"(seuil {CROSS_DOC_SHARE_KAG:.0%}) — c'est ce qui fait qu'un graphe relie le corpus "
        f"au lieu de le découper.",
        f"Types de relations réutilisés : {reuse:.0%} (seuil {REUSE_KAG:.0%}) — "
        f"{probe.get('distinct_predicates', 0)} types différents. Un vrai graphe réutilise "
        f"ses types de relations ; des formulations toutes différentes ne se relient à rien.",
        f"Relations proposées puis écartées : "
        f"{probe.get('relations_unverified', 0)} ({taux_ecartes:.0%}) — {note_ecartes}",
        f"Volume {_n(words)} mots → facteur d'amortissement {amort:.2f} "
        f"(plein tarif à partir de {_n(FULL_AMORT_WORDS)} mots).",
    ]

    # Une décision d'architecture prise sur une fraction du corpus doit le dire.
    if total_docs and couverts < total_docs:
        reasons.insert(1,
            f"ATTENTION : {couverts} des {total_docs} documents seulement ont été "
            f"sondés ({couverts / total_docs:.0%} du corpus). Les documents non lus "
            f"ne contribuent ni à la densité de relations ni aux types réutilisés. "
            f"Augmente le budget de morceaux pour couvrir tout le corpus.")

    # Un signal à moins de 20 % de son seuil peut basculer d'un sondage à l'autre :
    # la décision ne repose alors pas sur un écart solide.
    marginaux = [nom for nom, val, seuil in (
        ("densité de relations", rel_density, RELATIONS_PER_1000_KAG),
        ("entités multi-documents", cross_doc, CROSS_DOC_SHARE_KAG),
        ("réutilisation des types", reuse, REUSE_KAG),
    ) if seuil and abs(val - seuil) / seuil <= 0.20]
    if marginaux:
        reasons.append(
            f"Signal(aux) à moins de 20 % du seuil : {', '.join(marginaux)}. "
            f"À cette distance, la mesure peut basculer d'un sondage à l'autre : "
            f"ne présente pas ce signal comme tranché.")

    return score, reasons


# --- score de repli (spaCy, peu fiable) -------------------------------------
def _kag_from_counts(corpus: dict) -> tuple[int, list[str]]:
    degree = corpus.get("avg_entity_degree")
    conn   = corpus.get("cross_doc_connectivity")
    words  = corpus.get("total_words", 0) or 0

    if degree is None:
        return 0, ["Ni sondage LLM ni signaux d'entités disponibles → RAG par défaut."]

    s_deg  = _saturating(degree, DEGREE_KAG)
    s_conn = _saturating(conn or 0, CONNECTIVITY_KAG)
    structural = 0.5 * s_deg + 0.5 * s_conn
    score = round(100 * structural * _amortization(words))

    reasons = [
        "Sondage LLM indisponible : décision fondée sur le comptage d'entités, "
        "beaucoup moins fiable (les noms communs capitalisés et les tableaux "
        "faussent la mesure). Le seuil KAG est donc relevé.",
        f"Degré moyen d'entité {degree} (seuil {DEGREE_KAG}).",
        f"Connectivité inter-documents {(conn or 0):.0%} (seuil {CONNECTIVITY_KAG:.0%}).",
    ]
    return score, reasons


# ===========================================================================
# CANDIDATS DE DÉCOUPAGE (v5)
# ===========================================================================
# Jusqu'ici le routeur IMPOSAIT une taille de morceau tirée de la littérature.
# Défendable, jamais vérifié sur le corpus. Il PROPOSE désormais 3 ou 4
# manières de découper, que chunk_quality.py note ensuite sur les documents
# réels. Les règles ci-dessous ne désignent plus la réponse : elles éliminent
# les candidats absurdes pour qu'il en reste peu à mesurer.
#
# Un candidat = (stratégie, taille visée en caractères).

TITLES_ENOUGH      = 0.50   # part de documents à titres pour tenter le structurel
SECTION_TOO_SHORT  = 400    # sections plus courtes : le structurel émiette
TABLE_SHARE_HIGH   = 0.15   # au-delà, protéger les tableaux devient décisif
CHARS_PER_TOKEN    = 4.6    # même conversion que index_rag.py


def chunking_candidates(c: dict) -> dict:
    """Propose les découpages à mesurer, avec la raison de chaque choix."""
    titles_frac = c.get("docs_with_titles_frac", 0.0) or 0.0
    section_med = c.get("section_chars_median")
    table_share = c.get("table_chars_share", 0.0) or 0.0
    short_docs  = (c.get("avg_doc_words", 0) or 0) < SHORT_DOC_WORDS

    # Taille de référence : celle que le corpus contient réellement.
    # À défaut de sections mesurées, on retombe sur la règle de littérature.
    if section_med and section_med >= SECTION_TOO_SHORT:
        base = int(min(2400, max(600, section_med)))
        base_reason = (
            f"Taille de référence {base} caractères : c'est la longueur médiane "
            f"des sections réellement trouvées dans le corpus, pas une valeur "
            f"tirée d'un article."
        )
    else:
        base = 1400 if not short_docs else 700
        base_reason = (
            f"Taille de référence {base} caractères : aucune section exploitable "
            f"n'a pu être mesurée, on retombe sur la fourchette de la littérature "
            f"(~500-800 tokens)."
        )

    candidates: list[dict] = [
        {
            "name": "fixe",
            "strategy": "fixed",
            "target_chars": base,
            "why": "Témoin : le découpage actuel, en taille fixe. Sans lui, aucun "
                   "gain ne peut être chiffré.",
        }
    ]

    if titles_frac >= TITLES_ENOUGH:
        candidates.append({
            "name": "structurel",
            "strategy": "structural",
            "target_chars": base,
            "why": f"{titles_frac:.0%} des documents ont des titres détectables : "
                   f"on peut couper aux sections au lieu de couper au compteur.",
        })
    else:
        candidates.append({
            "name": "fixe court",
            "strategy": "fixed",
            "target_chars": max(400, base // 2),
            "why": f"Seuls {titles_frac:.0%} des documents ont des titres : le "
                   f"découpage par sections n'est pas praticable. On teste à la "
                   f"place une granularité plus fine.",
        })

    candidates.append({
        "name": "sémantique",
        "strategy": "semantic",
        "target_chars": base,
        "why": "Coupe là où le sujet change, mesuré par les embeddings. Plus lent "
               "à construire : c'est justement ce qu'il faut vérifier.",
    })

    if table_share >= TABLE_SHARE_HIGH:
        candidates.append({
            "name": "structurel large",
            "strategy": "structural",
            "target_chars": int(base * 1.6),
            "why": f"{table_share:.0%} du texte est dans des tableaux : des "
                   f"morceaux plus larges évitent de séparer un tableau de son "
                   f"titre.",
        })

    notes = [base_reason]
    empty = c.get("empty_pages_frac", 0.0) or 0.0
    if empty >= 0.20:
        notes.append(
            f"ATTENTION : {empty:.0%} des pages du corpus n'ont aucune couche de "
            f"texte (documents scannés ou captures d'écran). Aucun découpage ne "
            f"récupérera ce contenu : il faut un OCR, ou exclure ces documents et "
            f"le dire dans le rapport."
        )
    for doc in (c.get("image_only_docs") or [])[:5]:
        notes.append(f"Document quasi illisible (beaucoup de pages, peu de mots) : {doc}")

    return {
        "base_chars": base,
        "base_tokens_equiv": round(base / CHARS_PER_TOKEN),
        "candidates": candidates,
        "notes": notes,
        "_measured": False,   # passe à True quand chunk_quality les a notés
    }


# --- configuration RAG (3 versions) -----------------------------------------
def _rag_common(c: dict) -> tuple[dict, list[str]]:
    reasons = []
    short = c.get("avg_doc_words", 0) < SHORT_DOC_WORDS
    many_tables = (c.get("docs_with_tables_frac") or 0) >= 0.5
    multi = c.get("is_multilingual", False)

    chunk = 300 if short else 650
    if short:
        reasons.append(f"chunk_size={chunk} : documents courts → petits morceaux.")
    else:
        reasons.append(
            f"chunk_size={chunk} : longueur moyenne → morceaux moyens "
            f"(~500-800, point d'équilibre de la littérature)."
        )
    if many_tables:
        reasons.append("Beaucoup de tableaux → découpe qui ne coupe pas un tableau en deux.")

    # Un seul modèle d'embedding, quelle que soit la langue : c'est le seul que
    # l'on peut réellement faire tourner (voir EMBEDDING_MODEL ci-dessus).
    # Afficher un choix qui n'existe pas serait trompeur.
    model = EMBEDDING_MODEL
    if multi:
        reasons.append(
            f"Corpus multilingue → {EMBEDDING_MODEL}, modèle d'embedding multilingue "
            f"({EMBEDDING_DIM} dimensions)."
        )
    else:
        reasons.append(
            f"Corpus monolingue → {EMBEDDING_MODEL} également : c'est le seul modèle "
            f"d'embedding disponible, et il reste bon en monolingue."
        )

    overlap = round(chunk * 0.15)
    # Le pas d'avancement est `chunk - overlap`, pas `chunk` : ignorer le
    # recouvrement sous-estimait le nombre de morceaux d'environ 15 %.
    stride = max(1, chunk - overlap)
    n_chunks_est = max(1, int(c.get("total_tokens_est", 0) / stride))

    approx_index = n_chunks_est > LARGE_CORPUS_CHUNKS
    if approx_index:
        index = "HNSW (approché)"
        reasons.append(f"~{n_chunks_est} morceaux → index approché HNSW pour tenir la vitesse.")
    else:
        index = "FLAT (exact)"
        reasons.append(f"~{n_chunks_est} morceaux → index exact FLAT (précision parfaite).")

    common = {
        # --- couche LISIBLE (interface, rapport) ---------------------------
        "chunk_size": chunk,
        "chunk_overlap": overlap,
        "embedding_model": model,
        "vector_index": index,
        "distance": "cosine",
        # --- couche MACHINE (consommée par le code, masquée dans l'UI) -----
        # Les clés préfixées « _ » sont filtrées par render_config_table :
        # elles n'apparaissent pas dans l'interface mais pilotent le pipeline.
        "_n_chunks_est": n_chunks_est,
        "_chunk_unit": "tokens",
        "_embedding_model": model,
        "_embedding_dim": EMBEDDING_DIM,
        "_index_type": "HNSW" if approx_index else "FLAT",
        "_metric_type": "COSINE",
        "_index_params": ({"M": 16, "efConstruction": 200} if approx_index else {}),
    }
    return common, reasons


def _rag_versions(c: dict) -> dict:
    common, common_reasons = _rag_common(c)
    multi = c.get("is_multilingual", False)
    reranker_model = "BAAI/bge-reranker-v2-m3"
    versions = {}

    v = dict(common)
    v.update({"reranker": None, "top_k": 3, "retrieve_k": 3, "gen_verification": False})
    versions["vitesse"] = {
        "config": v,
        "reasons": [
            "Reranker coupé : c'est lui qui ralentit chaque question.",
            "top_k=3 : moins de morceaux à traiter.",
        ],
    }

    v = dict(common)
    use_rr = multi
    v.update({
        "reranker": reranker_model if use_rr else None,
        "top_k": 5,
        "retrieve_k": 12 if use_rr else 5,
        "gen_verification": False,
    })
    versions["équilibré"] = {
        "config": v,
        "reasons": [
            ("Reranker activé car le corpus est multilingue (c'est là qu'il rapporte le plus)."
             if use_rr else
             "Reranker coupé par défaut (corpus monolingue : le gain est surtout cross-lingue)."),
            ("On récupère 12 morceaux puis on garde les 5 meilleurs : un reranker n'a d'intérêt "
             "que s'il a plus de candidats que la sortie finale."
             if use_rr else "top_k=5 : réglage standard sans reranker."),
        ],
    }

    v = dict(common)
    v.update({"reranker": reranker_model, "top_k": 5, "retrieve_k": 18, "gen_verification": True})
    versions["qualité"] = {
        "config": v,
        "reasons": [
            "Reranker activé : gain de précision documenté.",
            "On récupère 18 morceaux puis on resserre à 5 : recette standard avec reranker.",
            "Vérification de la réponse par un second appel LLM — coûteux mais plus sûr.",
        ],
    }

    return {"common_reasons": common_reasons, "versions": versions}


# --- configuration KAG -------------------------------------------------------
def _kag_config(c: dict, mutability: str, probe: dict | None = None) -> dict:
    reasons = []
    homog = c.get("homogeneity", 0) or 0
    words = c.get("total_words", 0) or 0
    short = c.get("avg_doc_words", 0) < SHORT_DOC_WORDS

    cross_doc = (probe or {}).get("cross_doc_entity_share")
    if cross_doc is None:
        cross_doc = c.get("cross_doc_connectivity", 0) or 0

    chunk = 300 if short else 500
    reasons.append(
        f"Chunking ~{chunk} tokens : morceaux courts → plus d'entités extraites "
        f"(l'inverse du RAG)."
    )

    if homog >= HOMOGENEITY_ONTOLOGY:
        ontology = "contrainte (types d'entités/relations prédéfinis)"
        reasons.append(f"Corpus homogène ({homog:.0%}) → schéma d'entités contraint, plus propre.")
    else:
        ontology = "ouverte (extraction libre)"
        reasons.append(f"Corpus hétérogène ({homog:.0%}) → extraction ouverte, plus souple.")

    # --- décisions prises sous forme de BOOLÉENS et d'ENTIERS ---------------
    # La v3 ne produisait que des phrases françaises ("LLM fort, plusieurs
    # passes"), illisibles pour du code. On décide d'abord en valeurs machine,
    # puis on rédige la phrase à partir d'elles — jamais l'inverse.
    frozen = (mutability == "figé")
    passes = 2 if (frozen and words > FULL_AMORT_WORDS) else 1
    strict_resolution = cross_doc >= CROSS_DOC_SHARE_KAG
    constrained = homog >= HOMOGENEITY_ONTOLOGY
    communities = (words >= FULL_AMORT_WORDS and strict_resolution and frozen)
    extraction_model = EXTRACTION_MODEL_HEAVY if frozen else EXTRACTION_MODEL_LIGHT

    if frozen:
        extraction = "LLM fort, résolution d'entités soignée" + (
            ", plusieurs passes" if passes > 1 else ""
        )
        weight = "lourde (investissement unique)"
        reasons.append("Corpus figé → construction lourde permise : elle n'est faite qu'une fois.")
    else:
        extraction = "LLM léger, une seule passe, résolution basique"
        weight = "légère (corpus vivant)"
        reasons.append("Corpus vivant → construction légère : le graphe serait à refaire à chaque mise à jour.")

    resolution = "soignée (désambiguïsation)" if strict_resolution else "basique (surface)"

    reasons.append(
        "Détection de communautés : "
        + ("activée (graphe assez gros et bien relié)." if communities
           else "désactivée (graphe trop petit, peu relié, ou corpus vivant).")
    )

    return {
        "config": {
            # --- couche LISIBLE (interface, rapport) -----------------------
            "chunk_size": chunk,
            "extraction_model": extraction_model,
            "extraction_strategy": extraction,
            "ontology": ontology,
            "entity_resolution": resolution,
            "community_detection": communities,
            "graph_store": "Neo4j",
            "query_modes": "local + global" if communities else "local",
            "construction_weight": weight,
            # --- couche MACHINE (consommée par kag_ingest) -----------------
            "_chunk_unit": "tokens",
            "_extraction_model": extraction_model,
            "_extraction_passes": passes,
            "_ontology_mode": "constrained" if constrained else "open",
            "_entity_resolution": "strict" if strict_resolution else "basic",
            "_community_detection": bool(communities),
            "_query_modes": ["local", "global"] if communities else ["local"],
            "_mutability": mutability,
            "_homogeneity": round(homog, 3),
            "_cross_doc_share": round(cross_doc, 3),
        },
        "reasons": reasons,
    }


# --- point d'entrée ----------------------------------------------------------
def decide(corpus: dict, mutability: str = "figé", probe: dict | None = None) -> dict:
    """
    corpus     : la fiche d'identité (clé 'corpus' du profiler).
    mutability : 'figé' ou 'vivant'.
    probe      : le résultat de probe.probe_corpus(), ou None si pas de sondage.
    """
    words = corpus.get("total_words", 0) or 0
    used_probe = bool(probe and probe.get("available"))

    if words < HARD_MIN_WORDS_KAG:
        kag_score = 0
        kag_reasons = [
            f"Corpus de {_n(words)} mots : il tient entièrement dans une fenêtre de contexte. "
            f"En dessous de {_n(HARD_MIN_WORDS_KAG)} mots, construire un graphe coûte plus cher "
            f"que ça ne rapporte → RAG, sans même sonder le corpus."
        ]
        threshold = KAG_THRESHOLD_STATIC
    elif used_probe:
        kag_score, kag_reasons = _kag_from_probe(corpus, probe)
        threshold = KAG_THRESHOLD_LIVE if mutability == "vivant" else KAG_THRESHOLD_STATIC
    else:
        kag_score, kag_reasons = _kag_from_counts(corpus)
        threshold = max(KAG_THRESHOLD_FALLBACK,
                        KAG_THRESHOLD_LIVE if mutability == "vivant" else 0)

    choose_kag = kag_score >= threshold

    decision_reasons = list(kag_reasons)
    if mutability == "vivant" and words >= HARD_MIN_WORDS_KAG:
        decision_reasons.append(
            f"Corpus déclaré VIVANT → la barre pour le KAG monte à {threshold}/100 "
            f"(reconstruire un graphe à chaque mise à jour coûte cher)."
        )
    decision_reasons.append(
        f"Score KAG = {kag_score}/100 vs seuil {threshold} → "
        f"architecture retenue : {'KAG' if choose_kag else 'RAG'}."
    )

    rag_score = max(0, min(100, 100 - kag_score))

    return {
        "architecture": "KAG" if choose_kag else "RAG",
        "mutability": mutability,
        "decision_source": ("sondage LLM" if used_probe else
                            ("volume seul" if words < HARD_MIN_WORDS_KAG
                             else "comptage d'entités (repli)")),
        "kag_suitability": kag_score,
        "rag_suitability": rag_score,
        "kag_threshold": threshold,
        "fit_score": kag_score if choose_kag else rag_score,
        "decision_reasons": decision_reasons,
        "rag": _rag_versions(corpus),
        "kag": _kag_config(corpus, mutability, probe),
        # Le découpage est proposé quelle que soit l'architecture retenue :
        # index_kag.py découpe lui aussi avant d'extraire les triplets.
        "chunking": chunking_candidates(corpus),
    }


# --- ligne de commande -------------------------------------------------------
import json


def _load_json(path: str):
    """
    Lit un JSON quel que soit son encodage.

    PowerShell écrit « > fichier.json » en UTF-16 ; cmd.exe en ANSI ; Python en
    UTF-8. On renifle donc la marque d'ordre des octets au lieu d'imposer un
    encodage, sinon la commande marche chez l'un et pas chez l'autre.
    """
    raw = open(path, "rb").read()
    for bom, enc in ((b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"),
                     (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return json.loads(raw.decode(enc))
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252", errors="replace"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Routeur RAG/KAG (étage 2).")
    ap.add_argument("--profile", required=True, help="JSON produit par profiler.py")
    ap.add_argument("--probe", help="JSON produit par probe.py (facultatif)")
    ap.add_argument("--mutability", default="figé", choices=["figé", "vivant"])
    args = ap.parse_args()

    data = _load_json(args.profile)
    corpus = data.get("corpus", data)

    probe_data = None
    if args.probe:
        probe_data = _load_json(args.probe)

    result = decide(corpus, mutability=args.mutability, probe=probe_data)
    print(f"\nArchitecture retenue : {result['architecture']}  "
          f"(KAG {result['kag_suitability']}/100 · seuil {result['kag_threshold']})")
    print(f"Source de la décision : {result['decision_source']}\n")
    for r in result["decision_reasons"]:
        print(f"  - {r}")

    ch = result["chunking"]
    print(f"\nCandidats de découpage à mesurer "
          f"(référence {ch['base_chars']} caractères ≈ {ch['base_tokens_equiv']} tokens) :")
    for cand in ch["candidates"]:
        print(f"  · {cand['name']:<18} {cand['strategy']:<11} "
              f"{cand['target_chars']:>5} car.  — {cand['why']}")
    for note in ch["notes"]:
        print(f"  ! {note}")
    print()