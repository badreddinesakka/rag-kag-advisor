# -*- coding: utf-8 -*-
"""
profiler.py — Étage 1 : le PROFILER (v2).

Ne comprend rien au sens des documents : il MESURE.

Changements par rapport à la v1
-------------------------------
1. FILTRAGE DU BRUIT. spaCy `xx_ent_wiki_sm` (entraîné sur WikiNER) sur-étiquette
   tout ce qui est capitalisé. Sur des PDF, les en-têtes, sommaires et fragments
   de tableaux produisent des centaines de fausses entités. On rejette désormais
   les chaînes trop courtes, trop longues, majoritairement numériques, et les
   mots de structure documentaire (page, figure, annexe, mois...).

2. FUSION D'ALIAS. « Ooredoo », « Ooredoo Tunisie », « OOREDOO S.A. » comptaient
   comme trois entités distinctes, ce qui écrasait la connectivité inter-documents.
   On normalise (accents, casse, suffixes juridiques) puis on fusionne les formes
   dont les tokens sont inclus l'un dans l'autre.

3. ENTITÉS VUES UNE SEULE FOIS = BRUIT PROBABLE. Une entité apparaissant une
   seule fois dans tout le corpus est exclue des métriques de graphe.

4. ÉCHANTILLONNAGE RÉPARTI. La v1 lisait les 20 000 premiers caractères, soit la
   page de garde et le sommaire. On échantillonne maintenant trois fenêtres
   (début / milieu / fin) pour le même budget de calcul.

5. NOUVEAU SIGNAL : DEGRÉ MOYEN D'ENTITÉ. Un graphe ne se justifie pas par le
   nombre de NŒUDS mais par le nombre d'ARÊTES. On mesure les co-occurrences
   d'entités dans un même passage : `avg_entity_degree = 2 × paires / entités`.
   C'est le signal le plus honnêtement pro-KAG du profil.

6. BUG CORRIGÉ : `table_homog` était inversé (il valait 1 pour un corpus
   moitié-moitié et 0 pour un corpus parfaitement uniforme).

7. Lecture PDF protégée (un PDF chiffré ou corrompu ne fait plus planter l'app).

Dégradation propre : si une dépendance manque, le signal vaut None et le routeur
retombe sur RAG par défaut (jamais de crash).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import io
import json
import re
import unicodedata

# --- dépendances optionnelles ------------------------------------------------
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

_NLP = None
_NLP_TRIED = False


def _get_nlp():
    """Charge spaCy multilingue (xx_ent_wiki_sm) une seule fois, en paresseux."""
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    _NLP_TRIED = True
    try:
        import spacy
        _NLP = spacy.load("xx_ent_wiki_sm")
    except Exception:
        _NLP = None
    return _NLP


# --- constantes de mesure (knobs assumés) -----------------------------------
RELATIONAL_LABELS = {"PER", "PERSON", "ORG", "LOC", "GPE"}

SAMPLE_CHARS    = 24_000   # budget NER total par document
SAMPLE_WINDOWS  = 3        # réparti en 3 fenêtres : début / milieu / fin
LANG_SAMPLE     = 5_000    # échantillon pour langdetect
COOC_WINDOW     = 1_500    # taille d'un « passage » pour les co-occurrences
TOKENS_PER_WORD = 1.3      # approximation mots -> tokens

MIN_ENTITY_CHARS  = 3      # en dessous : sigle ambigu / bruit
MAX_ENTITY_WORDS  = 5      # au-dessus : fragment de phrase mal segmenté
MAX_DIGIT_RATIO   = 0.30   # au-dessus : référence, date, montant
MIN_ENTITY_COUNT  = 2      # vue une seule fois dans tout le corpus -> écartée
MAX_ALIAS_POOL    = 3_000  # borne le coût quadratique de la fusion d'alias

_LEGAL_SUFFIXES = {
    "sa", "sarl", "sas", "spa", "inc", "ltd", "llc", "plc", "gmbh", "bv", "nv",
    "corp", "corporation", "company", "co", "group", "groupe", "holding",
}

_ENTITY_STOPLIST = {
    # structure documentaire
    "page", "pages", "figure", "figures", "table", "tables", "tableau", "tableaux",
    "annexe", "annexes", "sommaire", "chapitre", "section", "article", "articles",
    "source", "sources", "note", "notes", "total", "sous", "rapport", "document",
    "www", "http", "https", "pdf", "email", "mail", "tel", "fax", "ref", "vol",
    # mois / jours
    "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout",
    "septembre", "octobre", "novembre", "decembre",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

_WS = re.compile(r"\s+")
_PUNCT_EDGE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


def _canonical(raw: str) -> str | None:
    """
    Normalise une mention d'entité, ou renvoie None si c'est du bruit.

    Renvoyer None ici est le principal garde-fou du profiler : c'est ce filtre
    qui empêche une densité d'entités absurde (>90 pour 1000 mots) de saturer
    le score du routeur.
    """
    s = _WS.sub(" ", raw.strip())
    s = _PUNCT_EDGE.sub("", s)
    if not s:
        return None
    if len(s.split()) > MAX_ENTITY_WORDS:
        return None

    s = _strip_accents(s.lower())
    if len(s) < MIN_ENTITY_CHARS:
        return None

    n_letters = sum(ch.isalpha() for ch in s)
    if n_letters < 2:
        return None
    n_digits = sum(ch.isdigit() for ch in s)
    if n_digits and n_digits / len(s) > MAX_DIGIT_RATIO:
        return None

    tokens = [t for t in s.split() if t and t not in _LEGAL_SUFFIXES]
    if not tokens:
        return None
    if all(t in _ENTITY_STOPLIST for t in tokens):
        return None
    return " ".join(tokens)


def _build_alias_map(counts: dict[str, int]) -> dict[str, str]:
    """
    Fusionne les variantes d'une même entité vers un représentant.

    Règle : deux formes fusionnent si l'ensemble de leurs tokens est inclus dans
    l'autre ET qu'elles partagent au moins un token discriminant (>= 4 lettres).
    Le représentant est la forme la plus fréquente.
    """
    items = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0])))[:MAX_ALIAS_POOL]
    reps: list[tuple[str, set[str]]] = []
    alias: dict[str, str] = {}
    for name, _ in items:
        toks = set(name.split())
        target = None
        for rname, rtoks in reps:
            if toks <= rtoks or rtoks <= toks:
                if any(len(t) >= 4 for t in (toks & rtoks)):
                    target = rname
                    break
        if target:
            alias[name] = target
        else:
            reps.append((name, toks))
            alias[name] = name
    return alias


def _windows(text: str) -> list[str]:
    """Échantillonne le document en SAMPLE_WINDOWS fenêtres réparties."""
    if len(text) <= SAMPLE_CHARS:
        return [text]
    w = SAMPLE_CHARS // SAMPLE_WINDOWS
    n = len(text)
    starts = [0, max(0, (n - w) // 2), max(0, n - w)]
    return [text[s:s + w] for s in starts]


def _couverture_ner(textes: list[str]) -> float:
    """
    Part du corpus réellement lue par la reconnaissance d'entités.

    Les entités ne sont cherchées que dans SAMPLE_CHARS caractères par document.
    Tous les signaux qui en découlent — entities_per_1000_words,
    cross_doc_connectivity, avg_entity_degree — portent donc sur une FRACTION du
    corpus, alors qu'ils sont présentés comme des mesures du corpus entier.

    Plus les documents sont longs, plus la fenêtre est étroite en proportion, et
    plus ces signaux baissent MÉCANIQUEMENT. Sans ce chiffre, rien ne permet de
    savoir si un cross_doc_connectivity faible traduit un corpus peu connecté ou
    simplement des documents longs.
    """
    total = sum(len(t or "") for t in textes) or 1
    lu = sum(min(len(t or ""), SAMPLE_CHARS) for t in textes)
    return round(lu / total, 3)


# --- mesure de la STRUCTURE (v3) --------------------------------------------
# Le profiler mesurait le CONTENU (langues, entités, tableaux) mais rien de la
# FORME. Or c'est la forme qui décide comment découper : un document à titres
# nets se découpe par sections, un document sans titres ne le peut pas.
# Sans ces quatre mesures, le routeur ne peut proposer aucun candidat de
# découpage — il ne peut qu'imposer une taille en aveugle.

TITLE_SIZE_RATIO = 1.12    # un titre est écrit au moins 12 % plus gros
TITLE_MAX_CHARS  = 120     # ... et reste court
MIN_TITLES_DOC   = 2       # en dessous, on ne parle pas d'une hiérarchie


def _lines_with_size(page) -> list[tuple[str, float]]:
    """Regroupe les mots d'une page en lignes : (texte, taille de police max)."""
    try:
        words = page.extract_words(extra_attrs=["size"]) or []
    except Exception:
        return []
    buckets: dict[int, list[dict]] = {}
    for w in words:
        key = int(round(float(w.get("top", 0)) / 3.0))   # tolérance verticale
        buckets.setdefault(key, []).append(w)
    out = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda w: float(w.get("x0", 0)))
        text = " ".join(str(w.get("text", "")) for w in group).strip()
        if not text:
            continue
        sizes = [float(w.get("size", 0) or 0) for w in group]
        out.append((text, max(sizes) if sizes else 0.0))
    return out


def _dominant_size(lines: list[tuple[str, float]]) -> float:
    """Taille du corps de texte = celle qui porte le plus de caractères."""
    weight: dict[float, int] = {}
    for text, size in lines:
        rounded = round(size * 2) / 2
        weight[rounded] = weight.get(rounded, 0) + len(text)
    if not weight:
        return 0.0
    return max(weight.items(), key=lambda kv: kv[1])[0]


def _empty_structure() -> dict:
    return {
        "has_titles": False,
        "n_titles": 0,
        "n_empty_pages": 0,
        "empty_pages_frac": 0.0,
        "section_chars_median": None,
        "table_chars_share": 0.0,
        "body_font_size": None,
    }


def _pdf_read(data: bytes) -> tuple[str, int, int, dict]:
    """
    Une seule ouverture du PDF : texte, pages, tableaux ET structure.

    On ne relit pas le fichier une seconde fois pour la structure : sur un
    corpus de plusieurs dizaines de PDF, le parsing pdfplumber est de loin le
    poste le plus cher du profiler.
    """
    struct = _empty_structure()
    if pdfplumber is None:
        return "", 0, 0, struct
    try:
        text_parts: list[str] = []
        n_tables = 0
        pages_lines: list[list[tuple[str, float]]] = []
        table_chars = 0

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    text_parts.append("")
                try:
                    tables = page.find_tables()
                    n_tables += len(tables)
                except Exception:
                    tables = []
                try:
                    for tbl in (page.extract_tables() or []):
                        for row in tbl:
                            table_chars += sum(len(c or "") for c in row)
                except Exception:
                    pass
                pages_lines.append(_lines_with_size(page))

        text = "\n".join(text_parts)

        # --- pages sans couche de texte (guides en captures d'écran) --------
        n_empty = sum(1 for lines in pages_lines if not lines)
        struct["n_empty_pages"] = n_empty
        struct["empty_pages_frac"] = round(n_empty / n_pages, 2) if n_pages else 0.0

        # --- titres et longueur des sections --------------------------------
        flat = [ln for lines in pages_lines for ln in lines]
        body = _dominant_size(flat)
        struct["body_font_size"] = round(body, 1) if body else None
        threshold = body * TITLE_SIZE_RATIO if body else float("inf")

        section_lengths: list[int] = []
        current = 0
        n_titles = 0
        for line_text, size in flat:
            is_title = (
                size >= threshold
                and len(line_text) <= TITLE_MAX_CHARS
                and not line_text.endswith(".")
            )
            if is_title:
                n_titles += 1
                if current > 0:
                    section_lengths.append(current)
                current = 0
            else:
                current += len(line_text) + 1
        if current > 0:
            section_lengths.append(current)

        struct["n_titles"] = n_titles
        struct["has_titles"] = n_titles >= MIN_TITLES_DOC
        if section_lengths:
            section_lengths.sort()
            struct["section_chars_median"] = section_lengths[len(section_lengths) // 2]

        total_chars = max(len(text), 1)
        struct["table_chars_share"] = round(min(1.0, table_chars / total_chars), 2)

        return text, n_pages, n_tables, struct
    except Exception:
        # PDF chiffré, corrompu, ou non pris en charge : dégradation propre
        return "", 0, 0, struct


# --- lecture du texte --------------------------------------------------------
def extract_all(name: str, data: bytes) -> tuple[str, int, int, dict]:
    """(texte, n_pages, n_tables, structure). Structure vide hors PDF."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return _pdf_read(data)
    text, n_pages, n_tables = extract_text(name, data)
    struct = _empty_structure()
    if suffix in (".txt", ".md"):
        # en markdown, un titre est une ligne commençant par #
        titles = [l for l in text.splitlines() if l.strip().startswith("#")]
        struct["n_titles"] = len(titles)
        struct["has_titles"] = len(titles) >= MIN_TITLES_DOC
    return text, n_pages, n_tables, struct


def extract_text(name: str, data: bytes) -> tuple[str, int, int]:
    """(texte, n_pages, n_tables) depuis des octets, selon l'extension.

    Signature inchangée : index_rag.py et index_kag.py l'appellent telle quelle.
    """
    suffix = Path(name).suffix.lower()

    if suffix == ".pdf":
        text, n_pages, n_tables, _ = _pdf_read(data)
        return text, n_pages, n_tables

    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore"), 0, 0

    if suffix == ".json":
        try:
            obj = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return "", 0, 0
        if isinstance(obj, str):
            return obj, 0, 0
        if isinstance(obj, dict):
            for k in ("text", "content", "full_text", "body"):
                if isinstance(obj.get(k), str):
                    return obj[k], 0, 0
        if isinstance(obj, list):
            parts = []
            for item in obj:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for k in ("text", "content", "full_text", "body"):
                        if isinstance(item.get(k), str):
                            parts.append(item[k])
                            break
            return "\n".join(parts), 0, 0
        return "", 0, 0

    return "", 0, 0


# --- profil d'un document ----------------------------------------------------
def profile_one(name: str, text: str, n_pages: int, n_tables: int,
                doc_index: int, structure: dict | None = None) -> dict | None:
    if not text or not text.strip():
        return None

    structure = structure or _empty_structure()
    words = text.split()
    prof = {
        "file": name,
        "n_pages": n_pages,
        "n_words": len(words),
        "n_chars": len(text or ""),
        "n_tables": n_tables,
        # --- structure (v3) : ce qui décide COMMENT découper ----------------
        "has_titles": structure["has_titles"],
        "n_titles": structure["n_titles"],
        "section_chars_median": structure["section_chars_median"],
        "empty_pages_frac": structure["empty_pages_frac"],
        "table_chars_share": structure["table_chars_share"],
        "language": None,
        "sampled_words": 0,
        "entities_per_1000_words": None,
        "relational_entity_share": None,
        "_labels": Counter(),      # interne : distribution des étiquettes NER
        "_mentions": [],           # interne : (passage_id, entité canonique)
        "_raw_ents": 0,            # interne : mentions brutes vues par spaCy
        "_kept_ents": 0,           # interne : mentions retenues après filtrage
    }

    if detect is not None:
        try:
            prof["language"] = detect(text[:LANG_SAMPLE])
        except Exception:
            pass

    nlp = _get_nlp()
    if nlp is None:
        return prof

    for wi, win in enumerate(_windows(text)):
        prof["sampled_words"] += len(win.split())
        try:
            ents = nlp(win).ents
        except Exception:
            continue
        for e in ents:
            prof["_raw_ents"] += 1
            canon = _canonical(e.text)
            if canon is None:
                continue
            prof["_kept_ents"] += 1
            prof["_labels"][e.label_] += 1
            if e.label_ in RELATIONAL_LABELS:
                passage_id = f"{doc_index}:{wi}:{e.start_char // COOC_WINDOW}"
                prof["_mentions"].append((passage_id, canon))

    sw = max(prof["sampled_words"], 1)
    prof["entities_per_1000_words"] = round(1000 * prof["_kept_ents"] / sw, 1)
    total_labels = sum(prof["_labels"].values())
    if total_labels:
        rel = sum(v for k, v in prof["_labels"].items() if k in RELATIONAL_LABELS)
        prof["relational_entity_share"] = round(rel / total_labels, 2)
    return prof


# --- agrégation : la fiche d'identité ---------------------------------------
def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    if m == 0:
        return 0.0
    var = sum((v - m) ** 2 for v in values) / len(values)
    return (var ** 0.5) / m


def aggregate(doc_profiles: list[dict]) -> dict:
    n = len(doc_profiles)

    # --- langues -------------------------------------------------------------
    langs = Counter(d["language"] for d in doc_profiles if d["language"])
    tot_lang = sum(langs.values()) or 1
    lang_dist = {k: round(v / tot_lang, 2) for k, v in langs.most_common()}
    is_multilingual = sum(1 for s in lang_dist.values() if s >= 0.15) >= 2

    n_words = sum(d["n_words"] for d in doc_profiles)

    def mean(key):
        vals = [d[key] for d in doc_profiles if d.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    # --- densité d'entités : au niveau CORPUS, pas moyenne de moyennes -------
    sampled_words = sum(d.get("sampled_words", 0) for d in doc_profiles)
    kept_mentions = sum(d.get("_kept_ents", 0) for d in doc_profiles)
    raw_mentions = sum(d.get("_raw_ents", 0) for d in doc_profiles)
    density = round(1000 * kept_mentions / sampled_words, 1) if sampled_words else None

    labels = Counter()
    for d in doc_profiles:
        labels.update(d.get("_labels", {}))
    total_labels = sum(labels.values())
    rel_share = None
    if total_labels:
        rel = sum(v for k, v in labels.items() if k in RELATIONAL_LABELS)
        rel_share = round(rel / total_labels, 2)

    # --- entités de graphe : filtrage + fusion d'alias -----------------------
    mention_counts = Counter()
    for d in doc_profiles:
        for _, canon in d.get("_mentions", []):
            mention_counts[canon] += 1

    frequent = {k: v for k, v in mention_counts.items() if v >= MIN_ENTITY_COUNT}
    alias = _build_alias_map(frequent)

    ent_docs: dict[str, set[int]] = defaultdict(set)
    passages: dict[str, set[str]] = defaultdict(set)
    rep_counts = Counter()
    for i, d in enumerate(doc_profiles):
        for passage_id, canon in d.get("_mentions", []):
            rep = alias.get(canon)
            if rep is None:
                continue
            rep_counts[rep] += 1
            ent_docs[rep].add(i)
            passages[passage_id].add(rep)

    distinct = len(ent_docs)

    connectivity = None
    if distinct:
        shared = sum(1 for docs in ent_docs.values() if len(docs) >= 2)
        connectivity = round(shared / distinct, 2)

    # --- co-occurrences : les ARÊTES, pas les nœuds --------------------------
    pairs: set[tuple[str, str]] = set()
    for ents in passages.values():
        if len(ents) < 2:
            continue
        for a, b in combinations(sorted(ents), 2):
            pairs.add((a, b))
    avg_degree = round(2 * len(pairs) / distinct, 2) if distinct else None

    top_entities = [
        {"entity": e, "mentions": c, "docs": len(ent_docs[e])}
        for e, c in rep_counts.most_common(25)
    ]

    # --- homogénéité (bug de la v1 corrigé) ---------------------------------
    lang_homog = max(lang_dist.values()) if lang_dist else 1.0
    len_cv = _coefficient_of_variation([d["n_words"] for d in doc_profiles])
    len_homog = max(0.0, 1.0 - min(len_cv, 1.0))
    with_tables = sum(1 for d in doc_profiles if d["n_tables"] > 0)
    table_frac = with_tables / n if n else 0.0
    # 1 si tous les docs se ressemblent (tous avec ou tous sans tableaux),
    # 0 si le corpus est coupé en deux moitiés.
    table_homog = 2 * abs(table_frac - 0.5)
    # ATTENTION — ce que cette valeur mesure vraiment.
    #
    # Elle moyenne trois choses sans rapport entre elles : la part de la langue
    # dominante, la régularité des LONGUEURS de documents, et l'uniformité de la
    # présence de tableaux. Elle ne dit RIEN du sujet des documents.
    #
    # Or elle pilotait la décision « ontologie contrainte ou ouverte », qui porte
    # sur la prévisibilité des TYPES d'entités. Un corpus de documents de
    # longueurs inégales ressortait donc en ontologie ouverte, pour une raison
    # sans rapport avec la question posée.
    #
    # Le nom est conservé pour ne pas casser router.py, mais les trois termes
    # sont désormais exposés séparément : celui qui décide doit voir lequel tire
    # la valeur vers le bas.
    #
    # Avec UN SEUL document, les trois termes valent 1 par construction : la
    # valeur ne mesure alors plus rien du tout, et on renvoie None.
    if len(doc_profiles) < 2:
        homogeneity = None
        homogeneity_note = ("non calculable : un seul document, les trois termes "
                            "valent 1 par construction")
    else:
        homogeneity = round((lang_homog + len_homog + table_homog) / 3, 2)
        faible = min([("langue", lang_homog), ("longueurs", len_homog),
                      ("tableaux", table_homog)], key=lambda x: x[1])
        homogeneity_note = (f"régularité de FORME, pas de sujet — terme le plus "
                            f"bas : {faible[0]} ({faible[1]:.2f})")

    # --- structure du corpus (v3) -------------------------------------------
    # Quatre mesures, quatre décisions de découpage :
    #   docs_with_titles_frac  -> le découpage par sections est-il possible ?
    #   section_chars_median   -> quelle taille de morceau viser ?
    #   empty_pages_frac       -> combien de pages sont des images sans texte ?
    #   table_chars_share      -> faut-il protéger les tableaux ?
    with_titles = sum(1 for d in doc_profiles if d.get("has_titles"))
    titles_frac = round(with_titles / n, 2) if n else 0.0

    sec_lengths = sorted(
        d["section_chars_median"] for d in doc_profiles
        if d.get("section_chars_median")
    )
    section_median = sec_lengths[len(sec_lengths) // 2] if sec_lengths else None

    empty_pages_tot = sum((d.get("empty_pages_frac", 0.0) or 0.0) * (d.get("n_pages") or 1)
                          for d in doc_profiles)
    pages_tot = sum((d.get("n_pages") or 1) for d in doc_profiles) or 1
    empty_frac = round(empty_pages_tot / pages_tot, 2)

    # Longueurs réelles (l'ancienne version estimait à n_words x 6 et traînait
    # un « * 0 » resté d'un essai). Les moyennes de corpus sont pondérées par la
    # TAILLE des documents : sans cela un guide d'une page pèse autant qu'un
    # rapport de cinquante.
    doc_chars = [max(d.get("n_chars", 0) or 0, 1) for d in doc_profiles]
    total_chars = sum(doc_chars) or 1
    table_share = round(
        sum((d.get("table_chars_share", 0.0) or 0.0) * c
            for d, c in zip(doc_profiles, doc_chars)) / total_chars, 2)

    # Part du corpus réellement lue par la reconnaissance d'entités.
    ner_coverage = round(sum(min(c, SAMPLE_CHARS) for c in doc_chars) / total_chars, 3)

    # documents illisibles : beaucoup de pages, presque pas de mots
    image_only = [
        d["file"] for d in doc_profiles
        if d["n_pages"] >= 3 and d["n_words"] / max(d["n_pages"], 1) < 60
    ]

    return {
        "n_docs": n,
        "total_words": n_words,
        "total_tokens_est": int(n_words * TOKENS_PER_WORD),
        "avg_doc_words": round(n_words / n) if n else 0,
        "languages": lang_dist,
        "is_multilingual": is_multilingual,
        "tables_per_doc": mean("n_tables"),
        "docs_with_tables_frac": round(table_frac, 2),

        # --- signaux de STRUCTURE (v3) --------------------------------------
        "docs_with_titles_frac": titles_frac,
        "section_chars_median": section_median,
        "empty_pages_frac": empty_frac,
        "table_chars_share": table_share,
        "image_only_docs": image_only,

        # signaux d'entités (après filtrage du bruit)
        # ner_coverage : part du corpus réellement lue. Tous les signaux qui
        # suivent portent sur cette fraction, pas sur le corpus entier.
        "ner_coverage": ner_coverage,
        "ner_coverage_note": (
            "les entités ne sont cherchées que dans les premiers "
            f"{SAMPLE_CHARS} caractères de chaque document ; "
            "sous 1.0, les signaux d'entités ci-dessous sont sous-estimés"),
        "entities_per_1000_words": density,
        "relational_entity_share": rel_share,
        "cross_doc_connectivity": connectivity,
        "distinct_entities": distinct,
        "entity_pairs": len(pairs),
        "avg_entity_degree": avg_degree,
        "homogeneity": homogeneity,
        "homogeneity_note": homogeneity_note,
        "homogeneity_langue": round(lang_homog, 2),
        "homogeneity_longueurs": round(len_homog, 2),
        "homogeneity_tableaux": round(table_homog, 2),

        # transparence : de quoi vérifier que la mesure n'est pas du bruit
        "top_entities": top_entities,
        "ner_label_distribution": dict(labels.most_common()),
        "entity_filter_stats": {
            "mentions_brutes": raw_mentions,
            "mentions_retenues": kept_mentions,
            "taux_de_rejet": round(1 - kept_mentions / raw_mentions, 2) if raw_mentions else None,
            "entites_distinctes_avant_alias": len(mention_counts),
            "entites_vues_une_seule_fois": len(mention_counts) - len(frequent),
            "entites_distinctes_apres_alias": distinct,
        },

        "ner_available": _get_nlp() is not None,
        "lang_available": detect is not None,
    }


def profile_corpus(files: list[tuple[str, bytes]]) -> dict:
    """
    files : liste de (nom, octets).
    Retourne {"corpus": <fiche agrégée>, "documents": [<profil par doc>]}.
    """
    docs = []
    for i, (name, data) in enumerate(files):
        text, n_pages, n_tables, structure = extract_all(name, data)
        p = profile_one(name, text, n_pages, n_tables, doc_index=i,
                        structure=structure)
        if p:
            docs.append(p)
    if not docs:
        return {"corpus": None, "documents": []}
    fiche = aggregate(docs)
    clean_docs = [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs]
    return {"corpus": fiche, "documents": clean_docs}


# --- utilisation en ligne de commande ---------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Profiler de corpus (étage 1).")
    ap.add_argument("--input", required=True, help="Dossier de documents")
    ap.add_argument("--entities", action="store_true",
                    help="Affiche les entités les plus fréquentes (diagnostic du bruit)")
    ap.add_argument("--out", help="écrit le profil dans ce fichier JSON (UTF-8). "
                                  "À préférer à « > fichier.json » : la redirection "
                                  "PowerShell écrit en UTF-16 et casse la relecture.")
    args = ap.parse_args()

    paths = sorted(Path(args.input).iterdir())
    files = [(p.name, p.read_bytes()) for p in paths if p.is_file()]
    result = profile_corpus(files)
    corpus = result["corpus"]
    if corpus is None:
        raise SystemExit("Aucun document exploitable.")

    if args.entities:
        print("\n--- entités les plus fréquentes (vérifier qu'il ne s'agit pas de bruit) ---")
        for row in corpus["top_entities"]:
            print(f"  {row['mentions']:>4} mentions · {row['docs']:>2} docs · {row['entity']}")
        print("\n--- filtrage ---")
        print(json.dumps(corpus["entity_filter_stats"], ensure_ascii=False, indent=2))
        print()

    printable = {k: v for k, v in corpus.items() if k != "top_entities"}

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Profil écrit dans {args.out} ({corpus['n_docs']} documents).")

    print(json.dumps(printable, ensure_ascii=False, indent=2))