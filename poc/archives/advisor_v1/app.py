# -*- coding: utf-8 -*-
"""
app.py — Interface utilisateur (Streamlit), v3.

Déroulé
-------
  1. Mesures rapides (profiler.py) : taille, langues, tableaux. Gratuit.
  2. Si le corpus est trop petit → décision immédiate, on ne réveille pas le LLM.
  3. Sinon → sondage (probe.py) : le LLM lit ~10 morceaux et en extrait les
     vraies relations. C'est cette mesure qui décide.
  4. Configuration complète + temps estimés.

Lancer :  streamlit run app.py
"""

import pandas as pd
import streamlit as st

import index_kag
import index_rag
from profiler import profile_corpus
from router import decide, needs_probe, HARD_MIN_WORDS_KAG
from estimator import estimate
from probe import probe_corpus, DEFAULT_MODEL
from decision import CONTRAINT, CONSEQUENCE, REGLE, MESURE
from router_preuve import choix_advisor, config_indexation, tient_dans_le_contexte

st.set_page_config(page_title="RAG / KAG Advisor", page_icon="🧭", layout="wide")

st.title("🧭 RAG / KAG Advisor")
st.caption(
    "Déposez un corpus : l'application le mesure, en extrait un échantillon de "
    "relations avec un LLM local, puis choisit RAG ou KAG et propose une "
    "configuration complète."
)

with st.sidebar:
    st.header("Paramètres")
    mutability = st.radio(
        "Le corpus sera-t-il mis à jour ?",
        options=["figé", "vivant"],
        help="Figé = indexé une fois. Vivant = mis à jour régulièrement. "
             "Un corpus vivant rend le graphe beaucoup moins intéressant.",
    )
    st.divider()
    st.subheader("Sondage LLM")
    use_probe = st.checkbox("Sonder le corpus avec le LLM", value=True,
                            help="Le LLM lit quelques morceaux et en extrait les vraies "
                                 "relations. C'est la mesure la plus fiable. "
                                 "Nécessite Ollama en local (~30 s).")
    probe_model = st.text_input("Modèle Ollama", value=DEFAULT_MODEL)
    n_chunks = st.slider("Morceaux analysés", 6, 40, 20)
    per_doc = st.slider("Morceaux par document", 1, 4, 2)
    st.divider()
    st.markdown(
        "**Note.** Les temps affichés sont des **estimations**, pas des mesures. "
        "Le temps réel dépend de votre matériel."
    )

files = st.file_uploader(
    "Documents du corpus (PDF, TXT, MD, JSON)",
    type=["pdf", "txt", "md", "json"],
    accept_multiple_files=True,
)
run = st.button("Analyser le corpus", type="primary", disabled=not files)


# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def profile_cached(payload: tuple) -> dict:
    return profile_corpus(list(payload))


def probe_cached(payload: tuple, model: str, chunks: int, per_doc: int) -> dict:
    """
    Sonde le corpus, en ne le refaisant pas si rien n'a changé.

    Le cache est géré à la main, dans st.session_state, et NON avec
    @st.cache_data. Raison : la barre de progression est un élément d'interface.
    Streamlit interdit d'en créer dans une fonction mise en cache, et au second
    passage (cache touché) la barre ne s'afficherait plus du tout.
    """
    key = ("probe", hash(payload), model, chunks, per_doc)
    if key in st.session_state:
        return st.session_state[key]

    bar = st.progress(0.0, text="Le LLM lit le corpus…")

    def on_progress(done, total):
        bar.progress(done / max(total, 1), text=f"Le LLM lit le corpus… {done}/{total}")

    result = probe_corpus(list(payload), model=model, n_chunks=chunks,
                          per_doc=per_doc, progress=on_progress)
    bar.empty()
    st.session_state[key] = result
    return result

def _n(x) -> str:
    return f"{x:,}".replace(",", " ")
def render_config_table(config: dict):
    rows = [(k, str(v)) for k, v in config.items() if not k.startswith("_")]
    st.table(pd.DataFrame(rows, columns=["Paramètre", "Valeur"]))


def render_preuve(choix: list):
    """
    La config, groupée par NIVEAU DE PREUVE.

    L'ancien tableau donnait le même poids visuel à « top_k = 5 » (une valeur
    tirée d'un article) et au découpage gagnant (comparé sur ce corpus). Le
    lecteur ne pouvait pas faire la différence. Ici il la voit d'un coup d'œil.
    """
    n_mes = sum(1 for c in choix if c.statut == MESURE)
    a, b, c_, d_ = st.columns(4)
    a.metric("Mesuré", n_mes, help="Comparé à des alternatives sur CE corpus.")
    b.metric("Conséquence", sum(1 for c in choix if c.statut == CONSEQUENCE),
             help="Découle directement d'un paramètre mesuré.")
    c_.metric("Réglé", sum(1 for c in choix if c.statut == REGLE),
              help="Choix raisonné, jamais vérifié ici.")
    d_.metric("Contraint", sum(1 for c in choix if c.statut == CONTRAINT),
              help="Imposé par le matériel ou les outils : pas un choix.")

    if n_mes == 0:
        st.warning(
            "Aucun paramètre n'a été mesuré sur ce corpus. La configuration "
            "ci-dessous est cohérente, mais rien ne dit qu'elle est la meilleure."
        )

    for statut, titre, aide in (
        (MESURE, "✅ Mesuré", "comparé à des alternatives sur ce corpus"),
        (CONSEQUENCE, "↳ Conséquence", "découle du paramètre mesuré au-dessus"),
        (REGLE, "⚠️ Réglé", "raisonné, mais jamais vérifié ici"),
        (CONTRAINT, "🔒 Contraint", "imposé — aucun choix possible"),
    ):
        groupe = [c for c in choix if c.statut == statut]
        if not groupe:
            continue
        st.markdown(f"**{titre}** — _{aide}_")
        st.table(pd.DataFrame(
            [(c.nom, str(c.valeur), c.phrase().split("—", 1)[-1].strip())
             for c in groupe],
            columns=["Paramètre", "Valeur", "Sur quoi ça repose"]))



def corpus_pour_estimation(corpus: dict, choix: list) -> dict:
    """
    Le corpus, avec un nombre de morceaux cohérent avec le découpage MESURÉ.

    estimator.py chiffre le temps à partir de « _n_chunks_est », que le routeur
    calcule en tokens AVANT toute mesure. Sur Ooredoo il annonçait 67 morceaux
    quand l'index en contenait 557 : les temps affichés étaient huit fois trop
    optimistes, et l'utilisateur le découvrait en lançant l'indexation.

    Ce n'est toujours qu'une estimation — le découpage réel dépend des blocs
    insécables — mais elle part de la bonne taille de morceau.
    """
    cfg = config_indexation(choix)
    taille = cfg.get("_chunk_chars")
    if not taille:
        return corpus
    chars = (corpus.get("total_words") or 0) * 6
    estime = max(1, round(chars / int(taille)))
    return {**corpus, "_n_chunks_mesure": estime}


def render_times(times: dict, n_chunks: int | None = None):
    col1, col2 = st.columns(2)
    col1.metric(times["transformation_label"], times["transformation_human"])
    col2.metric("Temps par question", times["query_human"])
    breakdown = times.get("transformation_breakdown")
    if breakdown:
        st.caption(" · ".join(f"{k} : {v}" for k, v in breakdown.items()))
    if n_chunks:
        st.caption(f"Estimation pour ~{n_chunks} morceaux, déduits de la taille "
                   f"de découpage mesurée. Le compte réel peut varier : chunker "
                   f"ne coupe ni un tableau ni un paragraphe.")


# ---------------------------------------------------------------------------
if run and files:
    st.session_state["payload"] = tuple((f.name, f.getvalue()) for f in files)

payload = st.session_state.get("payload")
if not payload:
    st.info("Déposez des documents puis cliquez sur **Analyser le corpus**.")
    st.stop()

with st.spinner("Mesure de la structure du corpus…"):
    profiled = profile_cached(payload)

corpus = profiled["corpus"]
if corpus is None:
    st.error("Aucun document exploitable (texte vide ou format non lu).")
    st.stop()

# --- 1. Fiche d'identité -----------------------------------------------------
st.header("1 · Fiche d'identité du corpus")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", corpus["n_docs"])
c2.metric("Mots (total)", f"{corpus['total_words']:,}".replace(",", " "))
c3.metric("Langues", ", ".join(corpus["languages"].keys()) or "—")
c4.metric("Multilingue", "oui" if corpus["is_multilingual"] else "non")

c5, c6, c7 = st.columns(3)
c5.metric("Longueur moyenne", f"{corpus['avg_doc_words']} mots")
c6.metric("Docs avec tableaux", f"{(corpus.get('docs_with_tables_frac') or 0):.0%}")
_h = corpus.get("homogeneity")
c7.metric("Régularité de forme", "—" if _h is None else f"{_h:.0%}",
          help="Moyenne de trois régularités de FORME : langue dominante, "
               "longueurs de documents, présence de tableaux. Ne dit rien du "
               "SUJET des documents.")
if corpus.get("homogeneity_note"):
    st.caption(f"Régularité de forme : {corpus['homogeneity_note']}")
if (corpus.get("ner_coverage") or 1.0) < 0.95:
    st.warning(
        f"Reconnaissance d'entités effectuée sur {corpus['ner_coverage']:.0%} "
        f"du corpus seulement. Les signaux d'entités (densité, entités "
        f"multi-documents, degré) sont donc SOUS-ESTIMÉS."
    )

# --- 2. Sondage --------------------------------------------------------------
probe = None
st.header("2 · Sondage : quelles relations contient vraiment ce corpus ?")

if not needs_probe(corpus):
    st.info(
        f"Corpus de {_n(corpus['total_words'])} mots, en dessous du seuil de "
        f"{_n(HARD_MIN_WORDS_KAG)} mots. Un graphe ne serait jamais rentable ici : "
        f"le sondage LLM est inutile, on ne le lance pas."
    )
elif not use_probe:
    st.warning(
        "Sondage désactivé dans la barre latérale. La décision retombera sur le "
        "comptage d'entités, beaucoup moins fiable."
    )
else:
    probe = probe_cached(payload, probe_model, n_chunks, per_doc)
    if not probe.get("available"):
        st.error(
            f"Sondage impossible : {probe.get('error')}\n\n"
            f"Vérifiez qu'Ollama tourne (`ollama serve`) et que le modèle est "
            f"téléchargé (`ollama pull {probe_model}`)."
        )
        probe = None
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Relations trouvées", probe["relations_kept"],
                  help="Relations réelles extraites par le LLM et vérifiées dans le texte.")
        p2.metric("Relations / 1000 tokens", probe["relations_per_1000_tokens"])
        p3.metric("Entités multi-documents", f"{probe['cross_doc_entity_share']:.0%}",
                  help="Part des entités qui reviennent dans au moins deux documents.")
        p4.metric("Types de relations réutilisés", f"{probe['relation_reuse']:.0%}",
                  help="Part des relations dont le type revient au moins deux fois. "
                       "Un vrai graphe réutilise ses types de relations.")

        st.caption(
            f"{probe['chunks_sampled']} morceaux lus sur {probe['docs_covered']} documents "
            f"({probe['chunks_per_doc_avg']} par document) · modèle {probe['model']} · "
            f"{probe['distinct_predicates']} types de relations différents · "
            f"{probe['relations_unverified']} relations non vérifiées dans le texte "
            f"({probe['unverified_rate']:.0%})."
        )

        with st.expander("Voir les relations extraites"):
            rows = [
                {"Sujet": r["sujet"], "Relation": r["relation"], "Objet": r["objet"]}
                for r in probe["sample_relations"]
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.write("Aucune relation retenue — c'est en soi une réponse : "
                         "ce corpus n'a pas de structure relationnelle.")
            if probe["top_predicates"]:
                st.markdown("**Relations les plus fréquentes**")
                st.table(pd.DataFrame(probe["top_predicates"],
                                      columns=["Relation", "Occurrences"]))

# --- 3. Décision -------------------------------------------------------------
with st.spinner("Mesure des découpages candidats…"):
    choix = choix_advisor(corpus, mutability=mutability, probe=probe,
                          files=list(payload))
arch = next(c.valeur for c in choix if c.nom == "architecture")

st.header("3 · Décision")

tient, serre, tokens, budget = tient_dans_le_contexte(corpus)
if tient:
    st.success(
        f"**Ne rien indexer.** Le corpus fait {_n(tokens)} tokens, le budget de "
        f"contexte en autorise {_n(budget)}. Tout tient dans le prompt : "
        f"découper puis rechercher ne peut que perdre de l'information."
    )
    if serre:
        st.warning("Marge serrée : un document de plus fera basculer vers le RAG.")
    st.header("4 · Configuration recommandée — CONTEXTE")
    render_preuve(choix)
    st.stop()

decision = decide(corpus, mutability=mutability, probe=probe)
left, right = st.columns([1, 2])
with left:
    st.metric("Architecture recommandée", arch)
    st.metric("Score d'adéquation", f"{decision['fit_score']}/100")
    st.progress(decision["fit_score"] / 100)
    st.caption(
        f"Score KAG {decision['kag_suitability']}/100 · seuil {decision['kag_threshold']} · "
        f"décision fondée sur : **{decision['decision_source']}**."
    )
with right:
    st.markdown("**Pourquoi cette architecture :**")
    for r in decision["decision_reasons"]:
        st.markdown(f"- {r}")

# --- 4. Configuration --------------------------------------------------------
st.header(f"4 · Configuration recommandée — {arch}")

render_preuve(choix)
st.divider()

if arch == "RAG":
    st.markdown(
        "Le RAG propose **trois versions**. Seuls le reranker et le nombre de "
        "morceaux changent ; le reste est fixé par la structure du corpus."
    )
    # Ces justifications viennent du routeur, AVANT la mesure. Celles qui
    # portent sur la taille des morceaux ou leur nombre sont périmées dès que
    # le découpage a été mesuré : les afficher telles quelles contredit le bloc
    # « mesuré » situé juste au-dessus, sur le même écran.
    mesure_faite = bool(config_indexation(choix).get("_chunk_strategy"))
    perimes = ("chunk_size", "morceaux →", "morceaux moyens")
    for r in decision["rag"]["common_reasons"]:
        if mesure_faite and any(m in r for m in perimes):
            st.markdown(f"- ~~_{r}_~~ — **remplacé par la mesure ci-dessus**")
        else:
            st.markdown(f"- _{r}_")

    tabs = st.tabs(["⚡ Vitesse", "⚖️ Équilibré (défaut)", "🎯 Qualité"])
    for tab, name in zip(tabs, ["vitesse", "équilibré", "qualité"]):
        with tab:
            block = decision["rag"]["versions"][name]
            # On n'affiche QUE les paramètres qui changent d'une version à
            # l'autre. Réafficher toute la config ici republiait chunk_size en
            # tokens et un top_k différent de celui du bloc de preuve : deux
            # tableaux, deux vérités, sur le même écran.
            varient = ("reranker", "top_k", "retrieve_k", "gen_verification")
            render_config_table({k: v for k, v in block["config"].items()
                                 if k in varient})
            st.caption("Seuls ces réglages changent entre les trois versions. "
                       "Le découpage, le modèle et l'index sont ceux du bloc "
                       "de preuve ci-dessus.")
            for r in block["reasons"]:
                st.markdown(f"- {r}")
            corpus_est = corpus_pour_estimation(corpus, choix)
            n_est = corpus_est.get("_n_chunks_mesure")
            cfg_est = dict(block["config"])
            if n_est:
                cfg_est["_n_chunks_est"] = n_est
            render_times(estimate("RAG", corpus_est, cfg_est), n_est)
else:
    st.markdown(
        "Le KAG est **dicté par la structure**. Le seul réglage est la mutabilité "
        "du corpus, qui fixe la lourdeur de construction."
    )
    block = decision["kag"]
    render_config_table(block["config"])
    for r in block["reasons"]:
        st.markdown(f"- {r}")
    render_times(estimate("KAG", corpus, block["config"]))
    st.info(
        "En KAG, le gros du travail est fait **une seule fois** à la construction. "
        "Ensuite, chaque question reste rapide."
    )

with st.expander("Voir aussi l'autre architecture (comparaison)"):
    if arch == "RAG":
        st.markdown("**Configuration KAG** (non retenue) :")
        render_config_table(decision["kag"]["config"])
        render_times(estimate("KAG", corpus, decision["kag"]["config"]))
    else:
        st.markdown("**Configuration RAG équilibrée** (non retenue) :")
        eq = decision["rag"]["versions"]["équilibré"]["config"]
        render_config_table(eq)
        render_times(estimate("RAG", corpus, eq))

# --- 5. Indexation -----------------------------------------------------------
# Jusqu'ici l'application ne faisait que MESURER et DÉCIDER. Cette section est
# la première qui ÉCRIT : elle transforme le corpus en base interrogeable, avec
# exactement les paramètres décidés au-dessus.
st.header("5 · Construire l'index")

st.caption(
    "Les paramètres ci-dessus ne sont plus des recommandations à recopier : "
    "ils sont appliqués tels quels. Le calcul est fait par Ollama, sur votre "
    "machine."
)

if arch == "RAG":
    st.markdown(
        "**Ce qui va être construit :** une collection Milvus. Le corpus est "
        "découpé, chaque morceau devient un vecteur, et le tout est indexé.\n\n"
        "Les trois versions (vitesse / équilibré / qualité) partagent **le même "
        "index** : elles ne diffèrent que par le reranker et le nombre de "
        "morceaux retenus, qui sont des réglages du moment de la question."
    )
    # config_indexation() porte « _chunk_strategy » et « _chunk_chars » quand
    # le découpage a été mesuré. Sans elle, l'indexation redécouperait à sa
    # façon et la mesure ne servirait à rien : la boucle resterait ouverte.
    index_config = dict(decision["rag"]["versions"]["équilibré"]["config"])
    index_config.update(config_indexation(choix))
    if index_config.get("_chunk_strategy"):
        st.success(f"Boucle fermée : l'index utilisera le découpage mesuré "
                   f"« {index_config['_chunk_strategy']} » à "
                   f"{index_config['_chunk_chars']} caractères.")
    else:
        st.info("Le découpage n'a pas été mesuré : l'indexation utilisera son "
                "découpage par défaut.")

    col_a, col_b = st.columns([2, 1])
    collection = col_a.text_input("Nom de la collection Milvus",
                                  value=index_rag.DEFAULT_COLLECTION)
    recreate = col_b.checkbox("Écraser si elle existe", value=False,
                              help="Sans cette case, l'application refuse "
                                   "d'indexer par-dessus une collection "
                                   "existante.")
    st.caption(f"Milvus : {index_rag.MILVUS_URI}")

    if st.button("Construire l'index Milvus", type="primary"):
        bar = st.progress(0.0, text="Démarrage…")

        def on_progress(done, total, msg):
            bar.progress(min(done / max(total, 1), 1.0), text=msg)

        with st.spinner("Indexation en cours…"):
            result = index_rag.index_corpus(
                list(payload), index_config, collection, recreate, on_progress
            )
        bar.empty()
        st.session_state["index_result"] = result

else:
    st.markdown(
        "**Ce qui va être construit :** un graphe Neo4j. Le corpus est découpé, "
        "un LLM en extrait les relations, les variantes d'un même nom sont "
        "regroupées, puis tout est écrit dans le graphe.\n\n"
        "C'est **long** : chaque morceau demande un appel au LLM. Ne fermez pas "
        "l'onglet pendant la construction."
    )
    index_config = dict(decision["kag"]["config"])
    index_config.update(config_indexation(choix))

    reset = st.checkbox("Vider le graphe avant de commencer", value=False,
                        help="Sans cette case, les nouvelles relations "
                             "s'ajoutent à celles déjà présentes.")
    st.caption(f"Neo4j : {index_kag.NEO4J_URI}")

    if not index_kag.NEO4J_PASSWORD:
        st.warning(
            "Le mot de passe Neo4j n'est pas défini. Renseignez la variable "
            "d'environnement `NEO4J_PASSWORD` (dans `docker-compose.yml`, ou "
            "avec `$env:NEO4J_PASSWORD = \"...\"` sous PowerShell)."
        )

    if st.button("Construire le graphe Neo4j", type="primary"):
        bar = st.progress(0.0, text="Démarrage…")

        def on_progress(done, total, msg):
            bar.progress(min(done / max(total, 1), 1.0), text=msg)

        with st.spinner("Construction du graphe en cours…"):
            result = index_kag.index_corpus(
                list(payload), index_config, probe, reset, on_progress
            )
        bar.empty()
        st.session_state["index_result"] = result

# Le résultat est gardé en session : Streamlit relance tout le script à chaque
# interaction, et sans cela le compte rendu disparaîtrait au premier clic.
result = st.session_state.get("index_result")
if result:
    if result.get("ok"):
        st.success("Index construit.")
        rows = [(k, str(v)) for k, v in result.items() if k != "ok"]
        st.table(pd.DataFrame(rows, columns=["Mesure", "Valeur"]))
        if result.get("avertissement"):
            st.warning(result["avertissement"])
    else:
        st.error(result.get("error", "Échec, sans message."))

# --- 6. Détail ---------------------------------------------------------------
with st.expander("Détail des mesures (JSON)"):
    st.markdown("**Profil du corpus**")
    st.json(corpus)
    if probe:
        st.markdown("**Sondage LLM**")
        st.json({k: v for k, v in probe.items() if k != "sample_relations"})