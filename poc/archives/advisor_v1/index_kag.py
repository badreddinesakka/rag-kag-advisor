# -*- coding: utf-8 -*-
"""
index_kag.py — Étage 5b : l'INDEXATION KAG.

Prend la configuration produite par l'Advisor et construit réellement le graphe
de connaissances dans Neo4j.

Chaîne complète :
    documents -> découpage -> extraction de triplets (Ollama)
              -> alignement des entités -> graphe Neo4j

Remplace l'ancien kag_ingest.py. Ce qui change, et pourquoi :

  1. LECTURE. L'ancien ne lisait que des .txt : aucun des 16 PDF du corpus
     n'entrait dans le graphe. On passe par profiler.extract_text.

  2. ÉCRITURE PAR LOTS. L'ancien ouvrait une session Neo4j PAR TRIPLET. Sur
     quelques milliers de triplets, cela fait autant d'allers-retours réseau.
     On envoie désormais par paquets avec UNWIND.

  3. CONTRAINTE D'UNICITÉ. Sans index sur Entity.name, chaque MERGE parcourt
     tous les nœuds existants. Le coût grandit avec le graphe. La contrainte
     crée l'index et rend le MERGE instantané.

  4. SOURCES CONSERVÉES. L'ancien écrasait source_chunk à chaque réécriture de
     la même relation. On accumule désormais une liste : la traçabilité
     graphe <-> texte survit.

  5. RÉGLAGES VENUS DE L'ADVISOR. Taille de morceau, modèle d'extraction,
     nombre de passes et seuil d'alignement ne sont plus écrits en dur.

  6. MOT DE PASSE. Plus dans le code : variable d'environnement NEO4J_PASSWORD.

Utilisation en ligne de commande :
    python index_kag.py --input ooredoo
    python index_kag.py --input ooredoo --reset
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import urllib.request

from profiler import extract_text
from index_rag import CHARS_PER_TOKEN, split_text, check_ollama

# --- adresses et identifiants (surchargeables par l'environnement) -----------
NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "bader1234")

_OLLAMA_GENERATE = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

REQUEST_TIMEOUT = 300
WRITE_BATCH     = 500     # triplets écrits dans Neo4j en une transaction
MAX_SOURCE_CHARS = 500    # extrait de texte conservé sur la relation

# Seuils d'alignement des entités, pilotés par la décision de l'Advisor.
# "strict"  : l'Advisor a vu des entités traverser les documents -> il faut
#             vraiment les fusionner, donc on tolère plus de variations.
# "basic"   : peu de recouvrement -> fusionner à tort abîmerait le graphe.
ALIGNMENT_STRICT = 0.80
ALIGNMENT_BASIC  = 0.88

# Suffixes juridiques retirés en mode strict, pour que « Ooredoo » et
# « Ooredoo Group » convergent vers le même nœud.
_LEGAL_SUFFIXES = {"sa", "s.a", "sarl", "sas", "inc", "ltd", "llc", "plc",
                   "corp", "corporation", "company", "co", "group", "groupe",
                   "holding", "holdings"}


CACHE_FILE      = "kag_triplets_cache.json"
CACHE_EVERY     = 5       # sauvegarde de reprise tous les N morceaux

# En dessous de ce nombre de types de relations, une ontologie « contrainte »
# n'est pas un schéma : c'est un bâillon. Mesuré sur le corpus RFP : le routeur
# a imposé le mode contraint (homogénéité 1,0, valeur mécanique sur un document
# unique) alors que le sondage n'avait trouvé que 3 prédicats — « filiale »,
# « est », « dirige ». Le LLM ne pouvait donc extraire AUCUNE exigence, puisque
# aucune ne s'exprime avec ces trois verbes. Résultat : 62 % de précision contre
# 88 % pour le RAG, et des triplets faits de fragments de phrases.
# On retombe sur l'extraction ouverte plutôt que d'obéir à un schéma vide.
MIN_ONTOLOGY_PREDICATES = 10


def _load_cache(path: str) -> dict:
    """Relit les relations déjà extraites lors d'un run précédent."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(path: str, cache: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


# ===========================================================================
# 1. PROMPT D'EXTRACTION
# ===========================================================================
BASE_PROMPT = """Tu es un extracteur d'information (Open Information Extraction).
À partir du texte ci-dessous, extrais TOUS les faits explicites sous forme de \
triplets (sujet, relation, objet).

Règles strictes :
- Sujet et objet : entités courtes (1 à 4 mots), SANS article.
  Exemple : "forfait flexi" et non "des forfaits flexi".
- Relation : un verbe ou une expression TRÈS courte (1 à 3 mots). JAMAIS une phrase.
- CONSERVE les codes, numéros et noms exacts tels quels : si le texte dit "*124#",
  le triplet doit contenir "*124#" mot pour mot.
- Un fait par triplet.
- N'invente RIEN qui ne soit explicitement dans le texte.
%(ontology)s
Réponds UNIQUEMENT avec du JSON, sans commentaire, au format :
{"relations": [{"sujet": "...", "relation": "...", "objet": "..."}]}

TEXTE :
\"\"\"
%(chunk)s
\"\"\"
"""

GLEANING_SUFFIX = """
Tu as déjà extrait ces relations de ce texte :
%s

Relis le texte et donne UNIQUEMENT les relations que tu as MANQUÉES. \
Si tu n'en trouves aucune, renvoie une liste vide.
"""


def ontology_from_probe(probe: dict | None, n: int = 15) -> list[str]:
    """
    Construit la liste des types de relations autorisés à partir du sondage.

    En mode « contraint », l'Advisor veut un schéma fermé. Le sondage a déjà lu
    un échantillon du corpus et compté les types de relations réellement
    présents : c'est la meilleure source disponible, et elle ne coûte rien.
    """
    if not probe:
        return []
    top = probe.get("top_predicates") or []
    return [str(p) for p, _ in top[:n] if str(p).strip()]


def _build_prompt(chunk: str, allowed: list[str]) -> str:
    if allowed:
        ontology = ("- Utilise UNIQUEMENT ces types de relation : "
                    + ", ".join(f"« {a} »" for a in allowed)
                    + ". Si aucun ne convient, n'extrais pas le fait.\n")
    else:
        ontology = ""
    return BASE_PROMPT % {"ontology": ontology, "chunk": chunk}


# ===========================================================================
# 2. APPEL AU LLM
# ===========================================================================
def _call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        _OLLAMA_GENERATE, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def _parse(raw: str) -> list[dict]:
    """Lit la réponse du LLM. Tolère les deux formats qu'il produit."""
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    rels = obj.get("relations") if isinstance(obj, dict) else obj
    if not isinstance(rels, list):
        return []
    out = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        s = str(r.get("sujet", "")).strip()
        p = str(r.get("relation", "")).strip()
        o = str(r.get("objet", "")).strip()
        if s and p and o:
            out.append({"sujet": s, "relation": p, "objet": o})
    return out


def extract_from_chunk(chunk: str, model: str, allowed: list[str],
                       passes: int = 1) -> list[dict]:
    """
    Extrait les triplets d'un morceau.

    Si l'Advisor demande plusieurs passes, la deuxième est une passe de
    RATTRAPAGE : on montre au LLM ce qu'il a déjà trouvé et on lui demande ce
    qu'il a manqué. Relancer le même prompt à l'identique ne servirait à rien —
    avec temperature 0, il redonnerait exactement la même réponse.
    """
    found = _parse(_call_ollama(_build_prompt(chunk, allowed), model))

    for _ in range(max(0, passes - 1)):
        if not found:
            break
        already = "; ".join(f"{r['sujet']} -> {r['relation']} -> {r['objet']}"
                            for r in found[:30])
        prompt = _build_prompt(chunk, allowed) + GLEANING_SUFFIX % already
        extra = _parse(_call_ollama(prompt, model))
        seen = {(r["sujet"].lower(), r["relation"].lower(), r["objet"].lower())
                for r in found}
        for r in extra:
            key = (r["sujet"].lower(), r["relation"].lower(), r["objet"].lower())
            if key not in seen:
                seen.add(key)
                found.append(r)

    return found


# ===========================================================================
# 3. ALIGNEMENT DES ENTITÉS
# ===========================================================================
def normalize_entity(name: str) -> str:
    """Minuscules et espaces normalisés. Première étape, purement textuelle."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _strip_legal(name: str) -> str:
    """Retire un suffixe juridique final (« ooredoo group » -> « ooredoo »)."""
    parts = name.split()
    while len(parts) > 1 and parts[-1].strip(".") in _LEGAL_SUFFIXES:
        parts.pop()
    return " ".join(parts) if parts else name


def _same_shape(a: str, b: str) -> bool:
    """
    Deux noms qui n'ont pas le même nombre de mots désignent rarement la même
    chose. « my ooredoo » (l'application) et « ooredoo » (l'entreprise) se
    ressemblent assez pour tromper la comparaison floue, alors que ce sont deux
    nœuds distincts. Les suffixes juridiques ayant déjà été retirés en amont, un
    mot en plus est un mot qui compte.
    """
    return len(a.split()) == len(b.split())


def build_alignment(names: list[str], threshold: float,
                    strict: bool) -> dict[str, str]:
    """
    Regroupe les variantes d'écriture d'une même entité.

    Sans cette étape, « appli my ooredoo » et « application my ooredoo »
    deviennent deux nœuds distincts, et le graphe se fragmente en îlots isolés
    que le raisonnement multi-saut ne peut pas traverser.

    Les noms courts sont traités en premier : ils deviennent les représentants
    du groupe, ce qui donne des nœuds aux noms simples.
    """
    unique = sorted(set(names), key=len)
    canonical: list[str] = []
    mapping: dict[str, str] = {}

    for name in unique:
        probe_name = _strip_legal(name) if strict else name
        # On demande plusieurs candidats et on garde le premier qui a le même
        # nombre de mots : le plus ressemblant n'est pas toujours le bon.
        matches = difflib.get_close_matches(probe_name, canonical, n=5,
                                            cutoff=threshold)
        chosen = next((m for m in matches if _same_shape(m, probe_name)), None)
        if chosen:
            mapping[name] = chosen
        else:
            canonical.append(probe_name)
            mapping[name] = probe_name

    return mapping


# ===========================================================================
# 4. NEO4J
# ===========================================================================
def _driver():
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError("neo4j n'est pas installé. Lance : pip install neo4j") from e
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "NEO4J_PASSWORD n'est pas défini. Sous PowerShell :\n"
            '   $env:NEO4J_PASSWORD = "ton_mot_de_passe"'
        )
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# La contrainte crée aussi un index. Sans elle, chaque MERGE sur Entity.name
# parcourt tous les nœuds : le temps de construction explose avec la taille.
_CONSTRAINT = ("CREATE CONSTRAINT entity_name IF NOT EXISTS "
               "FOR (e:Entity) REQUIRE e.name IS UNIQUE")

# UNWIND déroule une liste envoyée en un seul aller-retour réseau.
# Le CASE évite d'empiler le même extrait de texte à chaque réindexation.
_WRITE = """
UNWIND $rows AS row
MERGE (a:Entity {name: row.sujet})
MERGE (b:Entity {name: row.objet})
MERGE (a)-[r:RELATION {type: row.relation}]->(b)
SET r.sources = CASE
    WHEN row.source IN coalesce(r.sources, []) THEN r.sources
    ELSE coalesce(r.sources, []) + row.source
END
"""

# Suppression par paquets : un DETACH DELETE global sur un gros graphe
# saturerait la mémoire de Neo4j.
_DELETE_BATCH = ("MATCH (n:Entity) WITH n LIMIT 10000 "
                 "DETACH DELETE n RETURN count(n) AS deleted")


def reset_graph(driver) -> int:
    total = 0
    with driver.session() as session:
        while True:
            deleted = session.run(_DELETE_BATCH).single()["deleted"]
            total += deleted
            if deleted == 0:
                break
    return total


def write_triplets(driver, rows: list[dict]) -> None:
    with driver.session() as session:
        session.run(_CONSTRAINT)
        for start in range(0, len(rows), WRITE_BATCH):
            session.run(_WRITE, rows=rows[start:start + WRITE_BATCH])


# ===========================================================================
# 5. ORCHESTRATION
# ===========================================================================
def index_corpus(files: list[tuple[str, bytes]], config: dict,
                 probe: dict | None = None, reset: bool = False,
                 progress=None, max_chunks: int = 0,
                 resume: bool = False, chunk_size: int = 0,
                 force_open: bool = False) -> dict:
    """
    Construit le graphe complet.

    files      : liste de (nom, octets).
    config     : la configuration KAG de l'Advisor (clés « _ » comprises).
    probe      : le sondage, utilisé si l'Advisor demande une ontologie contrainte.
    reset      : si True, vide le graphe avant de commencer.
    progress   : fonction optionnelle appelée avec (fait, total, message).
    max_chunks : n'analyse que les N premiers morceaux (0 = tous). Sert à
                 mesurer la vitesse réelle avant de lancer le corpus entier.
    resume     : repart des relations déjà extraites lors d'un run interrompu.

    L'extraction est la partie longue : un appel LLM par morceau. Elle est donc
    sauvegardée au fur et à mesure dans un fichier de reprise. Sans cela, une
    interruption à la fin d'un run de 25 minutes perdrait tout le travail.
    """
    def say(done, total, msg):
        if progress:
            progress(done, total, msg)

    model  = config.get("_extraction_model") or config.get("extraction_model") or "qwen2.5:7b"
    passes = int(config.get("_extraction_passes", 1) or 1)
    strict = config.get("_entity_resolution", "basic") == "strict"
    threshold = ALIGNMENT_STRICT if strict else ALIGNMENT_BASIC

    allowed = []
    ontologie_refusee = None
    if config.get("_ontology_mode") == "constrained" and not force_open:
        allowed = ontology_from_probe(probe)
        if len(allowed) < MIN_ONTOLOGY_PREDICATES:
            ontologie_refusee = (
                f"Ontologie contrainte DEMANDÉE par le routeur mais REFUSÉE : le "
                f"sondage n'a trouvé que {len(allowed)} type(s) de relation "
                f"({', '.join(allowed) if allowed else 'aucun'}), il en faut au "
                f"moins {MIN_ONTOLOGY_PREDICATES}. Un schéma aussi étroit empêche "
                f"toute extraction. Extraction OUVERTE à la place.")
            allowed = []

    ok, err = check_ollama(model)
    if not ok:
        return {"ok": False, "error": err}

    # --- découpage ---------------------------------------------------------
    say(0, 1, "Lecture et découpage des documents…")
    # Le routeur propose 500 tokens (~2 300 caractères). Sur un document dense
    # en exigences, un morceau aussi large noie les relations : le LLM en sort
    # quelques-unes et ignore le reste. Un morceau plus court force un passage
    # plus attentif — même logique que le découpage enfant du RAG.
    chunk_tokens = chunk_size or config.get("chunk_size", 500)
    chunk_chars  = max(200, int(chunk_tokens * CHARS_PER_TOKEN))
    overlap_chars = max(0, chunk_chars // 10)

    # Découpage MESURÉ (voir index_rag.parents_mesures) : quand la config porte
    # « _chunk_strategy », le découpage vient de chunker.py avec la stratégie
    # qui a gagné chunk_quality, au lieu du split_text interne.
    #
    # ATTENTION, une réserve à assumer : chunk_quality note la PROPRETÉ des
    # morceaux (titres coupés, références orphelines, cohésion). Ce n'est pas
    # le même critère que « ce morceau permet-il d'extraire des relations ».
    # Le découpage gagnant côté RAG n'est donc pas forcément le meilleur ici.
    # Tant que rien ne mesure l'extraction elle-même, --chunk-size reste
    # prioritaire pour permettre de contredire la mesure à la main.
    strategie = config.get("_chunk_strategy")
    mes_chars = config.get("_chunk_chars")
    decoupage_source = "split_text interne (tokens x 4.6)"

    chunks = []
    if strategie and mes_chars and not chunk_size:
        from index_rag import parents_mesures
        modele_emb = config.get("_embedding_model") or "bge-m3"
        for p in parents_mesures(files, strategie, int(mes_chars), modele_emb):
            chunks.append((p["source"], p["text"]))
        chunk_chars = int(mes_chars)
        decoupage_source = f"chunker.py — stratégie mesurée « {strategie} »"
    else:
        for name, data in files:
            text, _, _ = extract_text(name, data)
            for piece in split_text(text or "", chunk_chars, overlap_chars):
                chunks.append((name, piece))

    if not chunks:
        return {"ok": False, "error": "Aucun texte exploitable dans les documents."}

    truncated = False
    if max_chunks and max_chunks < len(chunks):
        chunks = chunks[:max_chunks]
        truncated = True

    # --- extraction --------------------------------------------------------
    cache = _load_cache(CACHE_FILE) if resume else {}
    reused = 0
    raw_triplets: list[tuple[dict, str]] = []
    failed_chunks = 0
    total = len(chunks)

    for i, (name, piece) in enumerate(chunks):
        # Clé = empreinte du CONTENU, plus le numéro du morceau. L'ancienne
        # version indexait par numéro seul : changer de corpus ou de découpage
        # faisait réutiliser les triplets du corpus précédent, en silence.
        # Le README demandait d'effacer le cache à la main ; ce n'est plus
        # nécessaire, une empreinte différente ne peut pas se confondre.
        key = hashlib.sha1(piece.encode("utf-8")).hexdigest()
        if key in cache:
            found = cache[key]
            reused += 1
        else:
            say(i, total, f"Extraction des relations : morceau {i + 1}/{total}")
            try:
                found = extract_from_chunk(piece, model, allowed, passes)
            except Exception:
                failed_chunks += 1
                continue
            cache[key] = found
            if (i + 1) % CACHE_EVERY == 0:
                _save_cache(CACHE_FILE, cache)

        if not found:
            failed_chunks += 1
        for t in found:
            raw_triplets.append((t, piece[:MAX_SOURCE_CHARS]))

    _save_cache(CACHE_FILE, cache)

    if not raw_triplets:
        return {"ok": False,
                "error": "Aucune relation extraite. Le corpus n'a peut-être pas "
                         "de structure relationnelle, ou le modèle a échoué."}

    # --- alignement --------------------------------------------------------
    say(total, total, "Regroupement des entités…")
    all_names = []
    for t, _ in raw_triplets:
        all_names.append(normalize_entity(t["sujet"]))
        all_names.append(normalize_entity(t["objet"]))

    before = len(set(all_names))
    mapping = build_alignment(all_names, threshold, strict)
    after = len(set(mapping.values()))

    rows = []
    for t, source in raw_triplets:
        s = mapping.get(normalize_entity(t["sujet"]), normalize_entity(t["sujet"]))
        o = mapping.get(normalize_entity(t["objet"]), normalize_entity(t["objet"]))
        if not s or not o or s == o:
            continue
        rows.append({"sujet": s, "objet": o,
                     "relation": t["relation"].strip().lower(), "source": source})

    # --- écriture ----------------------------------------------------------
    say(total, total, "Écriture dans Neo4j…")
    try:
        driver = _driver()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    try:
        deleted = reset_graph(driver) if reset else 0
        write_triplets(driver, rows)
        with driver.session() as session:
            n_nodes = session.run(
                "MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            n_rels = session.run(
                "MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
    except Exception as e:
        return {"ok": False, "error": f"Erreur Neo4j : {e}"}
    finally:
        driver.close()

    say(total, total, "Graphe construit.")

    report = {
        "ok": True,
        "neo4j_uri": NEO4J_URI,
        "n_documents": len(files),
        "n_chunks": total,
        "chunks_sans_relation": failed_chunks,
        "triplets_extraits": len(raw_triplets),
        "triplets_ecrits": len(rows),
        "entites_avant_alignement": before,
        "entites_apres_alignement": after,
        "noeuds_dans_le_graphe": n_nodes,
        "relations_dans_le_graphe": n_rels,
        "noeuds_supprimes_au_reset": deleted,
        "extraction_model": model,
        "extraction_passes": passes,
        "entity_resolution": "strict" if strict else "basic",
        "alignment_threshold": threshold,
        "ontology_mode": ("open" if not allowed
                          else config.get("_ontology_mode", "open")),
        "ontology_predicates": allowed,
        "chunk_size_tokens": chunk_tokens,
        "chunk_size_chars": chunk_chars,
        "decoupage_source": decoupage_source,
        "decoupage_mesure": strategie or "non — découpage par défaut",
        "morceaux_repris_du_cache": reused,
    }

    if ontologie_refusee:
        report["ontologie_refusee"] = ontologie_refusee

    if truncated:
        report["avertissement_partiel"] = (
            f"Seuls les {total} premiers morceaux ont été analysés (--max-chunks). "
            f"Le graphe est INCOMPLET : ne l'utilise pas pour un benchmark.")

    if config.get("_community_detection"):
        report["avertissement"] = (
            "L'Advisor demande la détection de communautés. Elle exige le plugin "
            "GDS de Neo4j, qui n'est pas installé. Le graphe est construit sans.")

    try:
        with open("index_kag_config.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["config_file"] = "index_kag_config.json"
    except Exception:
        pass

    return report


# ===========================================================================
# 6. LIGNE DE COMMANDE
# ===========================================================================
def config_from_advisor(files, mutability="figé", use_probe=True):
    """Fait tourner le profiler, le sondage et le routeur. Renvoie (config, probe)."""
    from profiler import profile_corpus
    from router import decide

    profiled = profile_corpus(files)
    corpus = profiled["corpus"]
    if corpus is None:
        raise SystemExit("Aucun document exploitable.")

    probe_result = None
    if use_probe:
        from probe import probe_corpus
        probe_result = probe_corpus(files)
        if not probe_result.get("available"):
            print(f"[!] Sondage indisponible : {probe_result.get('error')}")
            probe_result = None

    decision = decide(corpus, mutability=mutability, probe=probe_result)
    if decision["architecture"] != "KAG":
        print(f"[!] L'Advisor recommande {decision['architecture']}, pas KAG. "
              f"On construit quand même le graphe demandé.")
    return decision["kag"]["config"], probe_result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Indexation KAG dans Neo4j.")
    ap.add_argument("--input", required=True, help="dossier contenant le corpus")
    ap.add_argument("--reset", action="store_true", help="vide le graphe avant")
    ap.add_argument("--mutability", default="figé", choices=["figé", "vivant"])
    ap.add_argument("--no-probe", action="store_true",
                    help="ne lance pas le sondage LLM (plus rapide)")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="n'analyse que les N premiers morceaux (0 = tous). "
                         "Sert à mesurer la vitesse avant un run complet.")
    ap.add_argument("--resume", action="store_true",
                    help="repart des relations déjà extraites "
                         f"({CACHE_FILE}) au lieu de tout refaire")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="taille de morceau en tokens (0 = celle de l'Advisor). "
                         "Plus court = plus de relations extraites, plus lent.")
    ap.add_argument("--force-open", action="store_true",
                    help="ignore l'ontologie contrainte du routeur")
    ap.add_argument("--config", metavar="JSON",
                    help="config produite par router_preuve.py --out. SANS elle, "
                         "le routeur est refait ici et le découpage mesuré est "
                         "ignoré : la boucle reste ouverte.")
    args = ap.parse_args()

    paths = sorted(Path(args.input).iterdir())
    corpus_files = [(p.name, p.read_bytes()) for p in paths if p.is_file()]
    if not corpus_files:
        raise SystemExit(f"Aucun fichier dans {args.input}.")

    print(f"{len(corpus_files)} fichiers lus dans {args.input}.")
    if args.config:
        import json as _json
        cfg = _json.loads(Path(args.config).read_text(encoding="utf-8"))
        probe_data = None
        if cfg.get("_chunk_strategy"):
            print(f"Découpage MESURÉ repris de {args.config} : "
                  f"{cfg['_chunk_strategy']} à {cfg['_chunk_chars']} caractères.")
        else:
            print(f"[!] {args.config} ne contient aucun découpage mesuré.")
        # Le sondage sert à l'ontologie ; sans lui, le routeur retombe sur sa
        # règle. On le relance seulement s'il n'est pas désactivé.
        if not args.no_probe:
            _, probe_data = config_from_advisor(corpus_files, args.mutability, True)
    else:
        print("[!] Aucun --config : le routeur est refait ici, sans la mesure. "
              "La boucle reste OUVERTE.")
        cfg, probe_data = config_from_advisor(corpus_files, args.mutability,
                                              not args.no_probe)
    print("Configuration retenue :")
    for k, v in cfg.items():
        print(f"   {k} = {v}")

    def show(done, total, msg):
        print(f"   {msg}      ", end="\r")

    import time
    started = time.time()
    result = index_corpus(corpus_files, cfg, probe_data, args.reset, show,
                          args.max_chunks, args.resume, args.chunk_size,
                          args.force_open)
    elapsed = time.time() - started
    print()
    if not result.get("ok"):
        raise SystemExit(f"[ÉCHEC] {result['error']}")

    print("\n[OK] Graphe construit.")
    if result.get("ontologie_refusee"):
        print(f"\n[!] {result['ontologie_refusee']}\n")
    for k, v in result.items():
        print(f"   {k} = {v}")

    # Le seul chiffre qui permette de prévoir la durée d'un run complet.
    done = result["n_chunks"] - result.get("morceaux_repris_du_cache", 0)
    if done > 0:
        print(f"\n   Durée : {elapsed / 60:.1f} min pour {done} morceaux extraits "
              f"({elapsed / done:.1f} s par morceau).")

    print("\nDans Neo4j Browser (http://localhost:7474) :")
    print("   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100")