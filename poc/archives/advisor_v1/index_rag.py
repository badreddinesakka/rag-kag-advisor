# -*- coding: utf-8 -*-
"""
index_rag.py — Étage 5a : l'INDEXATION RAG (v2, parent-enfant).

Prend la configuration produite par l'Advisor et construit réellement l'index
vectoriel dans Milvus.

Chaîne complète :
    documents -> découpage parent/enfant -> vecteurs (Ollama) -> Milvus

CE QUI CHANGE PAR RAPPORT À LA v1
=================================
La v1 découpait en morceaux de 650 tokens, soit ~3 000 caractères : une page
entière par morceau. Sur le corpus RFP, cela donnait 14 morceaux pour tout le
document. Le vecteur d'un tel morceau représente « une page qui parle un peu de
tout » — la recherche ne peut pas viser plus précis que la taille du morceau.
Mesure : 54 % de rappel sur l'extraction de critères.

DÉCOUPAGE PARENT / ENFANT
-------------------------
  - ENFANT (~120 tokens) : ce qu'on VECTORISE et ce qu'on CHERCHE.
    Une ligne de tableau, une puce. Le vecteur porte une seule idée.
  - PARENT (~650 tokens) : ce qu'on RENVOIE au LLM.
    Sans lui, la ligne « CCNP | Cisco | Professional | Yes » ne veut rien dire :
    on ignore que la colonne « Yes » signifie « obligatoire » et que le tableau
    s'intitule « Required Certifications ».

On cherche fin, on répond large. C'est la seule façon d'avoir à la fois la
précision de la recherche et le contexte nécessaire à la réponse.

Appui bibliographique : Bhat et al. 2025 (réf. [5]) montre que les petits
morceaux conviennent aux corpus à réponses factuelles courtes. Un appel d'offres
en est un. La v1 appliquait la règle « documents longs -> morceaux longs », qui
raisonne sur la taille des documents et non sur la granularité de la réponse
attendue.

Le mode plat de la v1 reste disponible (--flat) : c'est lui qui sert de témoin
dans la comparaison avant/après.

Utilisation :
    python index_rag.py --input dossier_rfp --recreate
    python index_rag.py --input dossier_rfp --recreate --flat        # ancien mode
    python index_rag.py --input dossier_rfp --recreate --child-tokens 80
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from profiler import extract_text

# --- adresses (surchargeables par variables d'environnement) -----------------
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")

_OLLAMA_GENERATE = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BASE = _OLLAMA_GENERATE.rsplit("/api/", 1)[0]

DEFAULT_COLLECTION = "advisor_rag"
EMBED_BATCH        = 16      # morceaux envoyés à Ollama en une fois
INSERT_BATCH       = 200     # lignes écrites dans Milvus en une fois
REQUEST_TIMEOUT    = 300

# L'Advisor raisonne en TOKENS, Ollama ne fournit pas de compteur de tokens.
# On convertit donc en caractères. Le profiler compte 1,3 token par mot ; un mot
# français ou anglais fait environ 6 caractères espace comprise, d'où 6 / 1,3.
CHARS_PER_TOKEN = 4.6

# --- découpage parent-enfant -------------------------------------------------
# 120 tokens ≈ 550 caractères : une ligne de tableau, une puce, une phrase.
# Assez court pour que le vecteur porte une seule idée, assez long pour que la
# ligne ne soit pas coupée en deux.
CHILD_TOKENS  = 120
PARENT_TOKENS = 650
CHILD_OVERLAP_RATIO  = 0.20   # entre enfants : évite de couper une ligne
PARENT_OVERLAP_RATIO = 0.10   # entre parents : évite de perdre une transition

# Endroits où couper de préférence, du plus propre au moins propre.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


# ===========================================================================
# 1. DÉCOUPAGE
# ===========================================================================
# --------------------------------------------------------------------------
# Pont vers le découpage MESURÉ (chunker.py + chunk_quality.py)
# --------------------------------------------------------------------------
# Jusqu'ici index_rag découpait avec son propre split_text, en tokens convertis
# en caractères. L'Advisor mesurait trois découpages, en désignait un, et ce
# gagnant n'était jamais utilisé : la boucle restait ouverte.
#
# Quand la config porte « _chunk_strategy » et « _chunk_chars » (posées par
# router_preuve.py après la mesure), les PARENTS sont désormais produits par
# chunker.py avec la stratégie gagnante. Les enfants restent découpés ici.
#
# chunker.chunk_corpus lit un DOSSIER, alors qu'on reçoit des (nom, octets) :
# on passe donc par un dossier temporaire, effacé aussitôt après.

def parents_mesures(files: list[tuple[str, bytes]], strategy: str,
                    target_chars: int, embed_model: str = "bge-m3") -> list[dict]:
    """Découpe avec la stratégie qui a gagné la mesure. Renvoie les parents."""
    import shutil
    import tempfile
    import chunker

    tmp = tempfile.mkdtemp(prefix="advisor_chunk_")
    try:
        for name, data in files:
            (Path(tmp) / Path(name).name).write_bytes(data)
        chunks = chunker.chunk_corpus(
            tmp, strategy=strategy, target_size=target_chars,
            embed_model=embed_model, verbose=False,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return [{"text": c.text, "source": c.doc_id, "section": c.section,
             "index": c.index} for c in chunks]


def split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """
    Découpe un texte en morceaux d'environ `chunk_chars` caractères.

    On ne coupe pas au caractère près : on cherche une coupure propre (fin de
    paragraphe, fin de ligne, fin de phrase, espace) dans le dernier quart du
    morceau. Couper au milieu d'un mot ou d'une phrase abîme le vecteur.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    search_zone = max(1, chunk_chars // 4)

    while start < len(text):
        end = min(start + chunk_chars, len(text))

        if end < len(text):
            cut = -1
            for sep in _SEPARATORS:
                found = text.rfind(sep, end - search_zone, end)
                if found > start:
                    cut = found + len(sep)
                    break
            if cut > 0:
                end = cut

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)

    return chunks


def build_chunks(files: list[tuple[str, bytes]], config: dict,
                 parent_child: bool = True,
                 child_tokens: int = CHILD_TOKENS,
                 parent_tokens: int | None = None) -> list[dict]:
    """
    Lit tous les fichiers et les découpe.

    En mode parent-enfant, chaque ligne produite contient :
      - `text`        : l'enfant, ce qui sera VECTORISÉ et cherché ;
      - `parent_text` : le parent, ce qui sera RENVOYÉ au LLM.

    En mode plat (--flat, comportement de la v1), `parent_text` est vide et
    `text` porte le morceau de 650 tokens. Les deux modes écrivent le même
    schéma, ce qui permet de comparer sans changer le code d'interrogation.

    Le découpage se fait document par document : un morceau ne chevauche jamais
    deux fichiers différents.
    """
    if parent_tokens is None:
        parent_tokens = config.get("chunk_size", PARENT_TOKENS)

    parent_chars = max(400, int(parent_tokens * CHARS_PER_TOKEN))
    parent_overlap = int(parent_chars * PARENT_OVERLAP_RATIO)

    rows = []

    if not parent_child:
        # --- mode v1 : découpage plat, un seul niveau ----------------------
        chunk_tokens   = config.get("chunk_size", 650)
        overlap_tokens = config.get("chunk_overlap", round(chunk_tokens * 0.15))
        chunk_chars    = max(200, int(chunk_tokens * CHARS_PER_TOKEN))
        overlap_chars  = max(0, int(overlap_tokens * CHARS_PER_TOKEN))
        if overlap_chars >= chunk_chars:
            overlap_chars = chunk_chars // 5

        for name, data in files:
            text, _, _ = extract_text(name, data)
            for i, piece in enumerate(split_text(text or "", chunk_chars,
                                                 overlap_chars)):
                rows.append({"text": piece, "parent_text": "", "source": name,
                             "chunk_index": i, "parent_index": i})
        return rows

    # --- mode parent-enfant ------------------------------------------------
    child_chars = max(150, int(child_tokens * CHARS_PER_TOKEN))
    child_overlap_m = int(child_chars * CHILD_OVERLAP_RATIO)

    # Découpage MESURÉ : les parents viennent de chunker.py, avec la stratégie
    # qui a gagné chunk_quality. C'est ici que la boucle se ferme.
    strategy = config.get("_chunk_strategy")
    mes_chars = config.get("_chunk_chars")
    if strategy and mes_chars:
        model = config.get("_embedding_model") or config.get("embedding_model") or "bge-m3"
        for pi, parent in enumerate(parents_mesures(files, strategy,
                                                    int(mes_chars), model)):
            enfants = split_text(parent["text"], child_chars, child_overlap_m) \
                      or [parent["text"]]
            for ci, enfant in enumerate(enfants):
                rows.append({"text": enfant, "parent_text": parent["text"],
                             "source": parent["source"],
                             "chunk_index": ci, "parent_index": pi})
        return rows

    child_overlap = int(child_chars * CHILD_OVERLAP_RATIO)

    for name, data in files:
        text, _, _ = extract_text(name, data)
        if not text or not text.strip():
            continue
        compteur = 0
        for pi, parent in enumerate(split_text(text, parent_chars,
                                               parent_overlap)):
            for enfant in split_text(parent, child_chars, child_overlap):
                rows.append({
                    "text": enfant,          # vectorisé
                    "parent_text": parent,   # renvoyé au LLM
                    "source": name,
                    "chunk_index": compteur,
                    "parent_index": pi,
                })
                compteur += 1
    return rows


# ===========================================================================
# 2. VECTEURS (Ollama)
# ===========================================================================
def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """
    Transforme une liste de textes en vecteurs, via Ollama.

    Deux routes possibles selon la version d'Ollama :
      - /api/embed      : accepte plusieurs textes d'un coup (récent, rapide) ;
      - /api/embeddings : un seul texte par appel (ancien, repli).

    bge-m3 n'a PAS besoin de préfixe. Les modèles e5 exigeaient d'écrire
    "passage: " devant chaque morceau et "query: " devant chaque question ;
    oublier ce détail dégradait silencieusement les résultats.
    """
    if not texts:
        return []
    try:
        body = _post(f"{OLLAMA_BASE}/api/embed", {"model": model, "input": texts})
        vectors = body.get("embeddings")
        if vectors and len(vectors) == len(texts):
            return vectors
    except urllib.error.HTTPError:
        pass  # Ollama trop ancien : on bascule sur l'ancienne route.

    vectors = []
    for t in texts:
        body = _post(f"{OLLAMA_BASE}/api/embeddings", {"model": model, "prompt": t})
        vec = body.get("embedding")
        if not vec:
            raise RuntimeError("Ollama n'a renvoyé aucun vecteur.")
        vectors.append(vec)
    return vectors


def check_ollama(model: str) -> tuple[bool, str]:
    """Vérifie qu'Ollama répond ET que le modèle demandé est bien téléchargé."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return False, f"Ollama injoignable sur {OLLAMA_BASE} ({e})."

    names = [m.get("name", "") for m in body.get("models", [])]
    short = {n.split(":")[0] for n in names}
    if model.split(":")[0] not in short:
        return False, (f"Le modèle « {model} » n'est pas téléchargé. "
                       f"Lance : ollama pull {model}")
    return True, ""


# ===========================================================================
# 3. MILVUS
# ===========================================================================
def _milvus_client():
    try:
        from pymilvus import MilvusClient
    except ImportError as e:
        raise RuntimeError(
            "pymilvus n'est pas installé. Lance : pip install pymilvus"
        ) from e
    return MilvusClient(uri=MILVUS_URI)


def create_collection(client, name: str, config: dict, chunk_chars: int,
                      parent_chars: int):
    """
    Crée la collection avec un schéma EXPLICITE.

    Le schéma explicite est ce qui permet d'appliquer réellement le choix
    FLAT / HNSW de l'Advisor. Avec la création automatique, Milvus impose ses
    propres réglages et la décision de l'Advisor ne sert à rien.

    Le champ `parent_text` stocke le contexte à renvoyer. On le garde dans la
    même ligne plutôt que dans une seconde collection : sur ces volumes, la
    duplication coûte moins cher qu'une jointure à faire à la main à chaque
    requête.
    """
    from pymilvus import DataType, MilvusClient

    dim = config.get("_embedding_dim", 1024)
    # Milvus compte la longueur des VARCHAR en OCTETS, pas en caractères. Un
    # accent français en occupe deux. On prévoit donc large.
    text_max   = min(65_535, max(2_000, chunk_chars * 4))
    parent_max = min(65_535, max(4_000, parent_chars * 4))

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=text_max)
    schema.add_field("parent_text", DataType.VARCHAR, max_length=parent_max)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("parent_index", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=config.get("_index_type", "FLAT"),
        metric_type=config.get("_metric_type", "COSINE"),
        params=config.get("_index_params") or {},
    )

    client.create_collection(collection_name=name, schema=schema,
                             index_params=index_params)


# ===========================================================================
# 4. ORCHESTRATION
# ===========================================================================
def index_corpus(files: list[tuple[str, bytes]], config: dict,
                 collection: str = DEFAULT_COLLECTION,
                 recreate: bool = False, progress=None,
                 parent_child: bool = True,
                 child_tokens: int = CHILD_TOKENS,
                 parent_tokens: int | None = None) -> dict:
    """
    Construit l'index complet.

    files        : liste de (nom, octets), comme pour le profiler.
    config       : la configuration RAG de l'Advisor (clés « _ » comprises).
    collection   : nom de la collection Milvus.
    recreate     : si True, supprime la collection existante avant de recommencer.
    progress     : fonction optionnelle appelée avec (fait, total, message).
    parent_child : False = découpage plat de la v1 (mode témoin).
    child_tokens : taille de l'enfant, ce qui est vectorisé.
    parent_tokens: taille du parent, ce qui est renvoyé au LLM.

    Retourne un compte rendu, ou {"ok": False, "error": ...}.
    """
    def say(done, total, msg):
        if progress:
            progress(done, total, msg)

    model = config.get("_embedding_model") or config.get("embedding_model") or "bge-m3"

    ok, err = check_ollama(model)
    if not ok:
        return {"ok": False, "error": err}

    if parent_tokens is None:
        parent_tokens = config.get("chunk_size", PARENT_TOKENS)

    # --- découpage ---------------------------------------------------------
    say(0, 1, "Lecture et découpage des documents…")
    rows = build_chunks(files, config, parent_child, child_tokens, parent_tokens)
    if not rows:
        return {"ok": False, "error": "Aucun texte exploitable dans les documents."}

    mesure_strategy = config.get("_chunk_strategy")
    mesure_chars    = config.get("_chunk_chars")

    if parent_child:
        chunk_chars  = max(150, int(child_tokens * CHARS_PER_TOKEN))
        # Quand le découpage a été mesuré, les parents ne viennent PAS du
        # calcul en tokens : les annoncer ici donnerait un compte rendu faux.
        parent_chars = (int(mesure_chars) if mesure_chars
                        else max(400, int(parent_tokens * CHARS_PER_TOKEN)))
    else:
        chunk_chars  = max(200, int(config.get("chunk_size", 650) * CHARS_PER_TOKEN))
        parent_chars = chunk_chars

    # --- collection --------------------------------------------------------
    try:
        client = _milvus_client()
        exists = client.has_collection(collection)
    except Exception as e:
        return {"ok": False,
                "error": f"Milvus injoignable sur {MILVUS_URI} ({e}). "
                         f"Vérifie que le conteneur milvus-standalone tourne."}

    if exists and not recreate:
        return {"ok": False,
                "error": f"La collection « {collection} » existe déjà. "
                         f"Coche « recréer » (ou --recreate) pour l'écraser."}
    if exists:
        client.drop_collection(collection)

    # Le schéma Milvus est dimensionné sur les morceaux RÉELS, pas sur la taille
    # visée. chunker.py ne coupe pas au milieu d'un tableau ni d'un paragraphe :
    # une cible de 1382 caractères peut produire un morceau de 6000. Dimensionner
    # sur la cible fait échouer l'insertion au premier morceau qui déborde.
    if rows:
        chunk_chars  = max(chunk_chars,  max(len(r["text"]) for r in rows))
        parent_chars = max(parent_chars, max(len(r["parent_text"]) for r in rows))

    try:
        create_collection(client, collection, config, chunk_chars, parent_chars)
    except Exception as e:
        return {"ok": False, "error": f"Création de la collection impossible : {e}"}

    # --- vecteurs + écriture ----------------------------------------------
    total = len(rows)
    inserted = 0
    buffer = []
    dim_reelle = None

    for start in range(0, total, EMBED_BATCH):
        batch = rows[start:start + EMBED_BATCH]
        try:
            vectors = embed_texts([r["text"] for r in batch], model)
        except Exception as e:
            return {"ok": False,
                    "error": f"Échec du calcul des vecteurs au morceau {start} : {e}"}

        # La dimension annoncée par le routeur n'est pas vérifiée ailleurs :
        # si le modèle d'embedding change, l'erreur Milvus est incompréhensible.
        # On la contrôle au premier lot, où le message peut encore être clair.
        if dim_reelle is None and vectors:
            dim_reelle = len(vectors[0])
            attendue = config.get("_embedding_dim", 1024)
            if dim_reelle != attendue:
                return {"ok": False,
                        "error": f"Le modèle « {model} » renvoie des vecteurs de "
                                 f"{dim_reelle} dimensions, la collection en attend "
                                 f"{attendue}. Corrige EMBEDDING_DIM dans router.py."}

        for row, vec in zip(batch, vectors):
            buffer.append({
                "vector": vec,
                "text": row["text"],
                "parent_text": row["parent_text"],
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "parent_index": row["parent_index"],
            })

        if len(buffer) >= INSERT_BATCH:
            client.insert(collection_name=collection, data=buffer)
            inserted += len(buffer)
            buffer = []

        say(min(start + EMBED_BATCH, total), total,
            f"Vecteurs calculés : {min(start + EMBED_BATCH, total)}/{total}")

    if buffer:
        client.insert(collection_name=collection, data=buffer)
        inserted += len(buffer)

    client.flush(collection_name=collection)
    say(total, total, "Index construit.")

    n_parents = len({(r["source"], r["parent_index"]) for r in rows})

    report = {
        "ok": True,
        "collection": collection,
        "milvus_uri": MILVUS_URI,
        "n_documents": len(files),
        "mode_decoupage": "parent-enfant" if parent_child else "plat (v1)",
        "n_chunks": inserted,
        "n_parents": n_parents,
        "enfants_par_parent": round(inserted / n_parents, 1) if n_parents else 0,
        "child_tokens": child_tokens if parent_child else None,
        "parent_tokens": parent_tokens if parent_child else None,
        "n_chunks_estimated_by_advisor": config.get("_n_chunks_est"),
        "embedding_model": model,
        "embedding_dim": dim_reelle or config.get("_embedding_dim", 1024),
        "index_type": config.get("_index_type", "FLAT"),
        "metric_type": config.get("_metric_type", "COSINE"),
        "chunk_size_tokens": config.get("chunk_size"),
        "chunk_overlap_tokens": config.get("chunk_overlap"),
        "chunk_size_chars": chunk_chars,
        "parent_size_chars_vise": (int(mesure_chars) if mesure_chars
                                   else max(400, int(parent_tokens * CHARS_PER_TOKEN))),
        "decoupage_mesure": mesure_strategy or "non — découpage par défaut",
        "parent_size_max_observe": (max(len(r["parent_text"]) for r in rows)
                                    if rows else 0),
        "parent_size_median_observe": (sorted(len(r["parent_text"]) for r in rows)
                                       [len(rows) // 2] if rows else 0),
        "parent_source": ("chunker.py (stratégie mesurée)" if mesure_strategy
                          else "split_text interne (tokens x 4.6)"),
        "chars_per_token_used": CHARS_PER_TOKEN,
    }

    # Sans cette trace, dans un mois personne ne saura avec quels réglages
    # l'index a été construit — et une comparaison RAG/KAG deviendra
    # ininterprétable.
    try:
        with open(f"{collection}_config.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["config_file"] = f"{collection}_config.json"
    except Exception:
        pass

    return report


# ===========================================================================
# 5. LIGNE DE COMMANDE
# ===========================================================================
def config_from_advisor(files, mutability="figé", version="équilibré",
                        use_probe=False) -> dict:
    """
    Fait tourner le profiler puis le routeur pour obtenir la configuration.

    Sert au mode ligne de commande, pour pouvoir tester ce fichier sans passer
    par Streamlit. Dans l'application, la configuration est déjà calculée : on
    la passe directement à index_corpus().
    """
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
    if decision["architecture"] != "RAG":
        print(f"[!] L'Advisor recommande {decision['architecture']}, pas RAG. "
              f"On construit quand même l'index RAG demandé.")
    return decision["rag"]["versions"][version]["config"]


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Indexation RAG dans Milvus.")
    ap.add_argument("--input", required=True, help="dossier contenant le corpus")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--recreate", action="store_true",
                    help="écrase la collection si elle existe déjà")
    ap.add_argument("--version", default="équilibré",
                    choices=["vitesse", "équilibré", "qualité"],
                    help="version RAG dont on prend la configuration "
                         "(l'index est le même pour les trois)")
    ap.add_argument("--mutability", default="figé", choices=["figé", "vivant"])
    ap.add_argument("--config", metavar="JSON",
                    help="config produite par router_preuve.py --out. "
                         "SANS elle, le routeur est refait ici et le découpage "
                         "mesuré est ignoré : la boucle reste ouverte.")
    ap.add_argument("--probe", action="store_true",
                    help="lance aussi le sondage LLM (plus lent)")
    ap.add_argument("--flat", action="store_true",
                    help="découpage plat de la v1, sans parent-enfant "
                         "(sert de témoin pour la comparaison)")
    ap.add_argument("--child-tokens", type=int, default=CHILD_TOKENS,
                    help="taille de l'enfant : ce qui est vectorisé et cherché")
    ap.add_argument("--parent-tokens", type=int, default=None,
                    help="taille du parent : ce qui est renvoyé au LLM "
                         "(par défaut, le chunk_size de l'Advisor)")
    args = ap.parse_args()

    paths = sorted(Path(args.input).iterdir())
    corpus_files = [(p.name, p.read_bytes()) for p in paths if p.is_file()]
    if not corpus_files:
        raise SystemExit(f"Aucun fichier dans {args.input}.")

    print(f"{len(corpus_files)} fichiers lus dans {args.input}.")
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if cfg.get("_chunk_strategy"):
            print(f"Découpage MESURÉ repris de {args.config} : "
                  f"{cfg['_chunk_strategy']} à {cfg['_chunk_chars']} caractères.")
        else:
            print(f"[!] {args.config} ne contient aucun découpage mesuré "
                  f"(relancer router_preuve.py avec --input).")
    else:
        print("[!] Aucun --config : le routeur est refait ici, sans la mesure. "
              "La boucle reste OUVERTE.")
        cfg = config_from_advisor(corpus_files, args.mutability, args.version, args.probe)
    print("Configuration retenue :")
    for k, v in cfg.items():
        print(f"   {k} = {v}")

    mode = "plat (v1)" if args.flat else "parent-enfant"
    print(f"\nDécoupage : {mode}")

    def show(done, total, msg):
        print(f"   {msg}      ", end="\r")

    result = index_corpus(corpus_files, cfg, args.collection, args.recreate,
                          show, parent_child=not args.flat,
                          child_tokens=args.child_tokens,
                          parent_tokens=args.parent_tokens)
    print()
    if not result.get("ok"):
        raise SystemExit(f"[ÉCHEC] {result['error']}")

    print("\n[OK] Index construit.")
    for k, v in result.items():
        print(f"   {k} = {v}")