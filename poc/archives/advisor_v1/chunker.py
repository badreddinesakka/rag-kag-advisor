"""
chunker.py — Découpage de documents pour l'Advisor RAG / KAG.

Trois stratégies de découpage :
  - "fixed"      : taille fixe, sans couper une phrase (comportement actuel, amélioré)
  - "structural" : coupe aux titres détectés par la taille de police
  - "semantic"   : coupe là où le sujet change (embeddings bge-m3 via Ollama)

Règles communes aux trois :
  - on ne coupe jamais au milieu d'une phrase
  - un tableau est un morceau entier, jamais coupé
  - chaque morceau est préfixé par "document > section" pour se suffire à lui-même

Utilisation en bibliothèque :
    from chunker import chunk_document, chunk_corpus
    chunks = chunk_document("doc.pdf", strategy="structural", target_size=1200)

Utilisation en ligne de commande :
    python chunker.py --input ooredoo --strategy structural
    python chunker.py --input ooredoo --strategy all --out comparaison.json

Dépendances : pdfplumber (déjà installé). Ollama seulement pour "semantic".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path

# --------------------------------------------------------------------------
# Réglages par défaut
# --------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("CHUNKER_EMBED_MODEL", "bge-m3")

DEFAULT_TARGET = 1200      # taille visée d'un morceau, en caractères
DEFAULT_OVERLAP = 150      # recouvrement, en caractères (stratégie "fixed" seulement)
MIN_CHUNK = 200            # en dessous, on fusionne avec le morceau suivant
MAX_CHUNK_FACTOR = 1.6     # un morceau ne dépasse jamais target * ce facteur

# Un titre est une ligne écrite plus gros que le corps du texte.
TITLE_SIZE_RATIO = 1.12    # au moins 12 % plus gros que la taille dominante
TITLE_MAX_CHARS = 120      # un titre reste court
MIN_DOCS_WITH_TITLES = 0.3 # sous ce taux, la stratégie structurelle n'est pas fiable

SUPPORTED = {".pdf", ".txt", ".md"}


# --------------------------------------------------------------------------
# Structure d'un morceau
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str               # texte final, en-tête compris
    body: str               # texte sans l'en-tête
    doc_id: str             # nom du fichier source
    section: str            # titre de la section d'origine ("" si inconnue)
    strategy: str           # fixed | structural | semantic
    index: int              # position du morceau dans le document
    is_table: bool = False
    n_chars: int = 0
    n_sentences: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Block:
    """Bloc de texte brut extrait d'un document, avant découpage."""
    text: str
    section: str = ""
    is_table: bool = False
    page: int = 0


@dataclass
class ParsedDoc:
    doc_id: str
    blocks: list[Block] = field(default_factory=list)
    has_titles: bool = False
    n_pages: int = 0
    n_empty_pages: int = 0     # pages sans couche de texte (captures d'écran)
    body_font_size: float = 0.0


# --------------------------------------------------------------------------
# 1. Lecture des documents
# --------------------------------------------------------------------------

def _group_words_into_lines(words: list[dict]) -> list[tuple[str, float]]:
    """Regroupe les mots d'une page en lignes. Renvoie (texte, taille max)."""
    lines: dict[int, list[dict]] = {}
    for w in words:
        key = int(round(w.get("top", 0) / 3.0))  # tolérance verticale
        lines.setdefault(key, []).append(w)

    out = []
    for key in sorted(lines):
        group = sorted(lines[key], key=lambda w: w.get("x0", 0))
        text = " ".join(w.get("text", "") for w in group).strip()
        if not text:
            continue
        sizes = [float(w.get("size", 0) or 0) for w in group]
        out.append((text, max(sizes) if sizes else 0.0))
    return out


def _dominant_size(all_lines: list[tuple[str, float]]) -> float:
    """Taille de police du corps de texte = celle qui porte le plus de caractères."""
    weight: dict[float, int] = {}
    for text, size in all_lines:
        rounded = round(size * 2) / 2  # arrondi au demi-point
        weight[rounded] = weight.get(rounded, 0) + len(text)
    if not weight:
        return 0.0
    return max(weight.items(), key=lambda kv: kv[1])[0]


def _table_to_text(table: list[list]) -> str:
    rows = []
    for row in table:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_pdf(path: Path) -> ParsedDoc:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError(
            "pdfplumber est requis pour lire les PDF : pip install pdfplumber"
        )

    doc = ParsedDoc(doc_id=path.name)
    pages_lines: list[list[tuple[str, float]]] = []
    tables_by_page: dict[int, list[str]] = {}

    with pdfplumber.open(str(path)) as pdf:
        doc.n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            try:
                words = page.extract_words(extra_attrs=["size"]) or []
            except Exception:
                words = []
            lines = _group_words_into_lines(words)
            pages_lines.append(lines)
            if not lines:
                doc.n_empty_pages += 1

            try:
                for tbl in (page.extract_tables() or []):
                    txt = _table_to_text(tbl)
                    if len(txt) > 40:
                        tables_by_page.setdefault(i, []).append(txt)
            except Exception:
                pass

    flat = [ln for page in pages_lines for ln in page]
    body = _dominant_size(flat)
    doc.body_font_size = body

    # Un titre : plus gros que le corps, et court.
    threshold = body * TITLE_SIZE_RATIO if body else float("inf")
    n_titles = 0
    current_section = ""
    buffer: list[str] = []

    def flush(page_no: int):
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            doc.blocks.append(Block(text=text, section=current_section, page=page_no))
        buffer = []

    for page_no, lines in enumerate(pages_lines):
        for text, size in lines:
            is_title = (
                size >= threshold
                and len(text) <= TITLE_MAX_CHARS
                and not text.endswith(".")
            )
            if is_title:
                flush(page_no)
                current_section = text
                n_titles += 1
            else:
                buffer.append(text)
        for txt in tables_by_page.get(page_no, []):
            flush(page_no)
            doc.blocks.append(
                Block(text=txt, section=current_section, is_table=True, page=page_no)
            )
    flush(len(pages_lines) - 1 if pages_lines else 0)

    doc.has_titles = n_titles >= 2
    return doc


def parse_text_file(path: Path) -> ParsedDoc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    doc = ParsedDoc(doc_id=path.name, n_pages=1)
    current_section = ""
    buffer: list[str] = []
    n_titles = 0

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                doc.blocks.append(
                    Block(text="\n".join(buffer).strip(), section=current_section)
                )
                buffer = []
            current_section = stripped.lstrip("#").strip()
            n_titles += 1
        else:
            buffer.append(line)
    if buffer:
        doc.blocks.append(Block(text="\n".join(buffer).strip(), section=current_section))

    doc.blocks = [b for b in doc.blocks if b.text]
    doc.has_titles = n_titles >= 2
    return doc


def parse_document(path: str | Path) -> ParsedDoc:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path)
    return parse_text_file(path)


# --------------------------------------------------------------------------
# 2. Découpage en phrases
# --------------------------------------------------------------------------

# Abréviations après lesquelles un point ne termine pas la phrase (FR + EN).
_ABBREV = {
    "m", "mm", "mme", "dr", "pr", "st", "ste", "art", "cf", "ex", "etc", "no", "n",
    "fig", "tel", "av", "bd", "ch", "vs", "mr", "mrs", "ms", "inc", "ltd", "co",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

_SENT_END = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    """Découpe en phrases. Simple mais robuste : pas de dépendance externe."""
    text = re.sub(r"[ \t]+", " ", text)
    parts = _SENT_END.split(text)

    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences:
            prev = sentences[-1]
            last_word = re.split(r"[\s(]", prev.rstrip("."))[-1].lower()
            # recolle si la « fin » de phrase est une abréviation ou une initiale
            if last_word in _ABBREV or re.fullmatch(r"[a-z]", last_word):
                sentences[-1] = prev + " " + part
                continue
            # recolle si la phrase précédente est trop courte pour en être une
            if len(prev) < 25 and not prev.endswith((":", "?", "!")):
                sentences[-1] = prev + " " + part
                continue
        sentences.append(part)

    # une ligne sans ponctuation finale (titre de liste, cellule) reste une unité
    return [s for s in sentences if s.strip()]


# --------------------------------------------------------------------------
# 3. Assemblage des phrases en morceaux
# --------------------------------------------------------------------------

def _pack(sentences: list[str], target: int, max_size: int) -> list[list[str]]:
    """Assemble des phrases jusqu'à la taille visée, sans jamais couper une phrase."""
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0

    for sent in sentences:
        s_len = len(sent) + 1
        # une phrase seule plus longue que le maximum : elle part telle quelle
        if s_len > max_size and not current:
            groups.append([sent])
            continue
        if current and size + s_len > target:
            groups.append(current)
            current, size = [sent], s_len
        else:
            current.append(sent)
            size += s_len

    if current:
        groups.append(current)
    return groups


def _merge_tiny(groups: list[list[str]], min_size: int) -> list[list[str]]:
    """Fusionne les groupes trop petits avec le suivant (ou le précédent)."""
    out: list[list[str]] = []
    for g in groups:
        if out and len(" ".join(g)) < min_size:
            out[-1].extend(g)
        else:
            out.append(g)
    if len(out) >= 2 and len(" ".join(out[0])) < min_size:
        out[1] = out[0] + out[1]
        out.pop(0)
    return out


def _make_chunk(doc_id, section, sentences, strategy, index, is_table=False) -> Chunk:
    body = " ".join(sentences).strip() if not is_table else sentences[0]
    header = doc_id if not section else f"{doc_id} > {section}"
    text = f"{header}\n\n{body}"
    return Chunk(
        text=text,
        body=body,
        doc_id=doc_id,
        section=section,
        strategy=strategy,
        index=index,
        is_table=is_table,
        n_chars=len(text),
        n_sentences=len(sentences) if not is_table else 1,
    )


# --------------------------------------------------------------------------
# 4. Stratégie "semantic" — embeddings via Ollama
# --------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Vectorise une liste de textes via Ollama. Essaie /api/embed, sinon /api/embeddings."""
    if not texts:
        return []
    try:
        out = _post_json(f"{OLLAMA_URL}/api/embed", {"model": model, "input": texts})
        vecs = out.get("embeddings")
        if vecs and len(vecs) == len(texts):
            return vecs
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError):
        pass

    vecs = []
    for t in texts:
        out = _post_json(f"{OLLAMA_URL}/api/embeddings", {"model": model, "prompt": t})
        vecs.append(out["embedding"])
    return vecs


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _semantic_groups(
    sentences: list[str], target: int, max_size: int, model: str
) -> list[list[str]]:
    """Coupe aux endroits où deux phrases consécutives se ressemblent le moins."""
    if len(sentences) < 3:
        return [sentences]

    vecs = embed_texts(sentences, model=model)
    sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]

    # Seuil relatif au document : les 25 % de frontières les moins ressemblantes.
    ordered = sorted(sims)
    threshold = ordered[max(0, int(len(ordered) * 0.25) - 1)]

    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]
    size = len(sentences[0])

    for i in range(1, len(sentences)):
        s_len = len(sentences[i]) + 1
        drop = sims[i - 1] <= threshold
        too_big = size + s_len > max_size
        big_enough = size >= target * 0.5

        if (drop and big_enough) or too_big:
            groups.append(current)
            current, size = [sentences[i]], s_len
        else:
            current.append(sentences[i])
            size += s_len

    if current:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------
# 5. Découpage d'un document
# --------------------------------------------------------------------------

def chunk_document(
    path: str | Path,
    strategy: str = "structural",
    target_size: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    embed_model: str = EMBED_MODEL,
    parsed: ParsedDoc | None = None,
) -> list[Chunk]:
    """Découpe un document selon la stratégie demandée. Renvoie une liste de Chunk."""
    doc = parsed or parse_document(path)
    max_size = int(target_size * MAX_CHUNK_FACTOR)
    chunks: list[Chunk] = []
    idx = 0

    # Repli automatique : pas de titres détectés => on ne peut pas faire de structurel.
    effective = strategy
    if strategy == "structural" and not doc.has_titles:
        effective = "fixed"

    if effective == "structural":
        # une section à la fois : on ne franchit jamais une frontière de section
        for block in doc.blocks:
            if block.is_table:
                chunks.append(
                    _make_chunk(doc.doc_id, block.section, [block.text],
                                strategy, idx, is_table=True)
                )
                idx += 1
                continue
            sents = split_sentences(block.text)
            if not sents:
                continue
            groups = _merge_tiny(_pack(sents, target_size, max_size), MIN_CHUNK)
            for g in groups:
                chunks.append(_make_chunk(doc.doc_id, block.section, g, strategy, idx))
                idx += 1

    elif effective == "semantic":
        for block in doc.blocks:
            if block.is_table:
                chunks.append(
                    _make_chunk(doc.doc_id, block.section, [block.text],
                                strategy, idx, is_table=True)
                )
                idx += 1
                continue
            sents = split_sentences(block.text)
            if not sents:
                continue
            groups = _semantic_groups(sents, target_size, max_size, embed_model)
            groups = _merge_tiny(groups, MIN_CHUNK)
            for g in groups:
                chunks.append(_make_chunk(doc.doc_id, block.section, g, strategy, idx))
                idx += 1

    else:  # "fixed"
        text_blocks = [b for b in doc.blocks if not b.is_table]
        table_blocks = [b for b in doc.blocks if b.is_table]
        full = "\n".join(b.text for b in text_blocks)
        sents = split_sentences(full)
        groups = _merge_tiny(_pack(sents, target_size, max_size), MIN_CHUNK)

        for i, g in enumerate(groups):
            # recouvrement : on reprend la dernière phrase du groupe précédent
            if overlap > 0 and i > 0:
                tail = groups[i - 1][-1]
                if len(tail) <= overlap:
                    g = [tail] + g
            chunks.append(_make_chunk(doc.doc_id, "", g, strategy, idx))
            idx += 1

        for b in table_blocks:
            chunks.append(
                _make_chunk(doc.doc_id, b.section, [b.text], strategy, idx, is_table=True)
            )
            idx += 1

    return chunks


def chunk_corpus(
    input_dir: str | Path,
    strategy: str = "structural",
    target_size: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    embed_model: str = EMBED_MODEL,
    verbose: bool = True,
) -> list[Chunk]:
    """Découpe tous les documents d'un dossier."""
    input_dir = Path(input_dir)
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not files:
        raise RuntimeError(f"Aucun document lisible dans {input_dir}")

    all_chunks: list[Chunk] = []
    for path in files:
        t0 = time.time()
        try:
            doc = parse_document(path)
            chunks = chunk_document(
                path, strategy=strategy, target_size=target_size,
                overlap=overlap, embed_model=embed_model, parsed=doc,
            )
        except Exception as exc:
            print(f"  ! {path.name} : {exc}", file=sys.stderr)
            continue
        all_chunks.extend(chunks)
        if verbose:
            flag = "titres" if doc.has_titles else "SANS titres"
            print(
                f"  {path.name:<52} {len(chunks):>4} morceaux  "
                f"({flag}, {doc.n_pages} p., {time.time() - t0:.1f} s)"
            )
    return all_chunks


# --------------------------------------------------------------------------
# 6. Statistiques et ligne de commande
# --------------------------------------------------------------------------

def stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {}
    sizes = sorted(c.n_chars for c in chunks)
    n = len(sizes)
    return {
        "n_chunks": n,
        "n_docs": len({c.doc_id for c in chunks}),
        "n_tables": sum(1 for c in chunks if c.is_table),
        "chars_min": sizes[0],
        "chars_median": sizes[n // 2],
        "chars_max": sizes[-1],
        "chars_mean": round(sum(sizes) / n),
        "pct_under_300": round(100 * sum(1 for s in sizes if s < 300) / n),
        "n_with_section": sum(1 for c in chunks if c.section),
    }


def _print_stats(label: str, chunks: list[Chunk]) -> None:
    s = stats(chunks)
    if not s:
        print(f"\n{label} : aucun morceau produit.")
        return
    print(f"\n--- {label} ---")
    print(f"  morceaux            : {s['n_chunks']} sur {s['n_docs']} documents")
    print(f"  taille (caractères) : min {s['chars_min']} | "
          f"médiane {s['chars_median']} | moyenne {s['chars_mean']} | max {s['chars_max']}")
    print(f"  morceaux < 300 car. : {s['pct_under_300']} %")
    print(f"  tableaux isolés     : {s['n_tables']}")
    print(f"  avec titre de section : {s['n_with_section']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Découpeur de documents pour l'Advisor")
    ap.add_argument("--input", required=True, help="dossier contenant les documents")
    ap.add_argument("--strategy", default="structural",
                    choices=["fixed", "structural", "semantic", "all"])
    ap.add_argument("--size", type=int, default=DEFAULT_TARGET,
                    help=f"taille visée en caractères (défaut {DEFAULT_TARGET})")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                    help="recouvrement, stratégie 'fixed' uniquement")
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--out", help="fichier JSON de sortie (facultatif)")
    ap.add_argument("--show", type=int, default=0,
                    help="afficher les N premiers morceaux")
    args = ap.parse_args()

    strategies = ["fixed", "structural", "semantic"] if args.strategy == "all" \
        else [args.strategy]

    results: dict[str, list[Chunk]] = {}
    for strat in strategies:
        print(f"\n=== Stratégie : {strat} (taille visée {args.size} caractères) ===")
        t0 = time.time()
        chunks = chunk_corpus(
            args.input, strategy=strat, target_size=args.size,
            overlap=args.overlap, embed_model=args.embed_model,
        )
        results[strat] = chunks
        _print_stats(strat, chunks)
        print(f"  durée totale : {time.time() - t0:.1f} s")

        for c in chunks[: args.show]:
            print(f"\n  [{c.doc_id} | {c.section or 'sans section'} | {c.n_chars} car.]")
            print("  " + c.body[:300].replace("\n", " ") + " ...")

    if args.out:
        payload = {
            strat: {"stats": stats(cs), "chunks": [c.to_dict() for c in cs]}
            for strat, cs in results.items()
        }
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nÉcrit : {args.out}")


if __name__ == "__main__":
    main()
