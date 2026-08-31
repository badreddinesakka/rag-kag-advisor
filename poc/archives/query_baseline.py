# -*- coding: utf-8 -*-
"""
query_baseline.py — la LIGNE DE BASE : le document entier dans le contexte.

Ni recherche vectorielle, ni graphe. On lit les fichiers, on colle tout dans le
prompt, on demande la réponse.

Pourquoi c'est la mesure la plus importante du banc d'essai
-----------------------------------------------------------
Le routeur affirme qu'en dessous de 15 000 mots un corpus « tient entièrement
dans une fenêtre de contexte » et qu'un graphe coûterait plus qu'il ne
rapporte. C'est une affirmation, pas une mesure.

Cette ligne de base la transforme en mesure. Si elle bat le RAG et le KAG sur
un petit corpus, la porte des 15 000 mots est justifiée par un chiffre. Si elle
perd, il faut revoir le seuil. Les deux résultats sont bons à écrire.

Attention : qwen2.5:7b a une fenêtre par défaut de 2048 tokens dans Ollama.
Sans num_ctx, le document est silencieusement tronqué et la ligne de base perd
pour une mauvaise raison. On le fixe donc explicitement.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from index_rag import OLLAMA_BASE
from profiler import extract_text
from prompts import FORMAT_JSON, PROMPTS

REQUEST_TIMEOUT = 900
NUM_CTX = 16_384   # fenêtre de contexte demandée à Ollama


def lire_corpus(dossier: str) -> tuple[str, dict]:
    """Concatène le texte de tous les fichiers du dossier."""
    chemins = sorted(Path(dossier).iterdir())
    morceaux, lus = [], []
    for p in chemins:
        if not p.is_file():
            continue
        texte, _, _ = extract_text(p.name, p.read_bytes())
        if texte and texte.strip():
            morceaux.append(f"=== {p.name} ===\n{texte.strip()}")
            lus.append(p.name)
    plein = "\n\n".join(morceaux)
    return plein, {"fichiers_lus": lus, "n_fichiers": len(lus)}


def repondre(dossier: str, nom_prompt: str, gen_model: str = "qwen2.5:7b",
             num_ctx: int = NUM_CTX) -> dict:
    contexte, infos = lire_corpus(dossier)
    if not contexte:
        return {"ok": False, "error": f"Aucun texte exploitable dans {dossier}."}

    prompt = PROMPTS[nom_prompt] % contexte

    payload = {
        "model": gen_model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    if FORMAT_JSON[nom_prompt]:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            reponse = json.loads(resp.read().decode("utf-8")).get("response", "")
    except Exception as e:
        return {"ok": False, "error": f"Ollama : {e}"}

    mots = len(contexte.split())
    # 1,3 token par mot, la même approximation que le profiler.
    tokens_est = int(mots * 1.3)

    resultat = {
        "ok": True,
        "architecture": "BASELINE",
        "prompt": nom_prompt,
        "reponse": reponse,
        "contexte_caracteres": len(contexte),
        "contexte_mots": mots,
        "tokens_estimes": tokens_est,
        "num_ctx_demande": num_ctx,
        "appels_llm_question": 1,
        **infos,
    }

    if tokens_est > num_ctx:
        resultat["avertissement"] = (
            f"Le corpus fait ~{tokens_est} tokens pour une fenêtre de {num_ctx}. "
            f"Ollama va TRONQUER : la ligne de base est faussée. Augmente "
            f"--num-ctx, ou constate que le corpus ne tient plus en contexte "
            f"— ce qui est en soi un argument pour RAG ou KAG.")

    return resultat


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Ligne de base : tout le document dans le contexte.")
    ap.add_argument("--input", required=True, help="dossier du corpus")
    ap.add_argument("--prompt", default="detaille", choices=["court", "detaille"])
    ap.add_argument("--gen-model", default="qwen2.5:7b")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    args = ap.parse_args()

    r = repondre(args.input, args.prompt, args.gen_model, args.num_ctx)
    if not r.get("ok"):
        raise SystemExit(f"[ÉCHEC] {r['error']}")
    if r.get("avertissement"):
        print(f"[!] {r['avertissement']}\n")

    print(f"{r['n_fichiers']} fichiers · {r['contexte_mots']} mots "
          f"(~{r['tokens_estimes']} tokens)\n")
    print(r["reponse"])
