# -*- coding: utf-8 -*-
"""
router_preuve.py — met une étiquette de preuve sur chaque sortie de router.py.

Un paramètre n'est pas qu'une valeur : c'est une valeur PLUS ce qui la soutient.
Ce fichier range chaque sortie du routeur en quatre niveaux — mesuré,
conséquence, réglé, contraint — et refuse de présenter un `if` et une mesure de
la même façon.

CE QUI CHANGE DANS CETTE VERSION (advisor2)
===========================================
LA TROISIÈME ARCHITECTURE « CONTEXTE » EST SUPPRIMÉE, avec ses quatre
constantes : MODELE_GENERATION, FENETRE_TOKENS, PART_RESERVEE, PART_SERREE.

Pourquoi, alors qu'elle avait raison sur le RFP. Ce n'est pas la règle qui était
fausse — « si le corpus tient dans la fenêtre, ne l'indexe pas » reste juste, et
la mesure du 30/08 l'a confirmée (RAG 92 % de précision contre 83 % au KAG sur
un corpus qui n'avait besoin ni de l'un ni de l'autre).

Le problème est ce dont elle a besoin pour trancher : la taille de la fenêtre du
modèle de génération. Ce nombre ne se déduit pas du corpus. Il décrit une
INSTALLATION. Écrit en dur, il vaut 32 768 sur la machine de développement et
n'a aucune raison de valoir cela ailleurs — or c'est lui, et lui seul, qui
décidait entre « n'indexe rien » et « indexe ». Une règle juste alimentée par
une constante fausse produit une décision fausse, et elle la produit sans
prévenir.

L'Advisor ne répond donc plus que RAG ou KAG, à partir de ce que le corpus dit
vraiment : ses entités, leurs liens, sa connectivité entre documents.

À GARDER POUR LE RAPPORT : l'observation reste valable même si l'outil ne la
fait plus. Sur un corpus qui tient en contexte, les deux architectures indexées
perdent. C'est un résultat de mesure, pas une fonctionnalité.

Usage :
    python router_preuve.py --profile profil.json [--probe sonde.json]
"""

from __future__ import annotations

import json
from pathlib import Path

import router
from decision import (config, consequence, contraint, diagnostic, mesure,
                      regle, resume)

def _choix_rag(c: dict, chunk_chars_mesure: int | None = None) -> list:
    """
    Les sorties RAG, chacune avec son niveau de preuve.

    TROIS CORRECTIONS D'INCOHÉRENCE, repérées en relisant le rapport PDF —
    des contradictions invisibles à l'écran, où les paramètres défilaient sans
    qu'on les lise côte à côte.

    1. « chunk_size » EN TOKENS DISPARAÎT quand un découpage a été mesuré.
       Le rapport affichait « chunk_chars 2400 caractères » et « chunk_size
       650 tokens » dans deux blocs différents : deux tailles pour la même
       chose, dont une seule fait foi. On ne garde que celle qui est appliquée.

    2. « reranker » NE PEUT PAS ÊTRE « aucun » ET « activé ». La justification
       venait de la variante « équilibré », qui l'active parfois ; la valeur
       venait de la config réelle. On rédige maintenant la phrase à partir de
       la valeur, pas l'inverse.

    3. « retrieve_k » NE PEUT PAS SE JUSTIFIER PAR UN RERANKER ABSENT.
       Sa raison invoquait « recette standard avec reranker » alors que le
       reranker valait « aucun ».
    """
    # Un découpage mesuré rend la taille en tokens caduque : elle décrivait un
    # candidat, pas ce qui sera appliqué.
    # La taille mesurée vient du bloc découpage, pas de la config RAG :
    # aller la chercher dans « c » donnait 0, d'où un recouvrement nul.
    mesure_faite = bool(chunk_chars_mesure)
    reranker = c.get("reranker") or "aucun"
    actif = reranker != "aucun"

    sorties = [
        # TYPE, pas nom. Et « réglé », pas « contraint » : rien ne l'impose,
        # c'est la langue du corpus qui le justifie. La dimension du vecteur
        # ne figure plus ici — elle dépend du modèle que l'utilisateur
        # choisira, donc elle appartient à la section « non décidé ».
        regle("embedding_type", c["embedding_model"],
              "type déduit de la langue du corpus ; le NOM du modèle est "
              "choisi par l'utilisateur parmi ceux dont il dispose"),
        contraint("distance", c["_metric_type"],
                  "cosinus, standard pour des vecteurs normalisés"),
        # PLUS « CONTRAINT ». Le seuil FLAT/HNSW est mécanique, mais il
        # s'applique à n_chunks_est — une estimation que la ligne suivante
        # déclare peu fiable. Une contrainte assise sur un chiffre douteux
        # n'est pas une contrainte : c'est une règle, et il faut le dire.
        regle("index_type", c["_index_type"],
              f"seuil mécanique appliqué à une ESTIMATION (~{c['_n_chunks_est']} "
              f"morceaux). Si le découpage réel s'en écarte beaucoup, l'index "
              f"choisi peut ne plus être le bon"),
        regle("n_chunks_est", c["_n_chunks_est"],
              "estimation calculée en TOKENS avant le découpage réel, qui se "
              "fait en CARACTÈRES. L'écart observé va jusqu'à un facteur huit : "
              "à lire comme un ordre de grandeur, jamais comme un compte"),
    ]

    if not mesure_faite:
        sorties.append(
            regle("chunk_size", c["chunk_size"],
                  "fourchette de la littérature, en tokens, jamais vérifiée sur "
                  "ce corpus — remplacée par chunk_chars dès qu'un découpage "
                  "est mesuré"))

    # LE RECOUVREMENT SUIT L'UNITÉ DU DÉCOUPAGE RETENU.
    # Le routeur le calcule en tokens (15 % de chunk_size). Dès qu'un
    # découpage est mesuré, la taille passe en CARACTÈRES : afficher un
    # recouvrement de 98 tokens à côté d'une taille de 2400 caractères mettait
    # deux unités dans la même phrase, et le chiffre était faux dans les deux.
    if mesure_faite:
        taille_car = int(chunk_chars_mesure)
        sorties.append(
            consequence("chunk_overlap_chars", round(taille_car * 0.15),
                        f"15 % des {taille_car} caractères du découpage mesuré. "
                        f"La proportion reste un usage courant, jamais vérifié ; "
                        f"seule l'unité est désormais cohérente"))
    else:
        sorties.append(
            regle("chunk_overlap", c["chunk_overlap"],
                  "15 % de la taille en tokens, usage courant"))

    sorties += [
        regle("top_k", c["top_k"],
              "valeur de la littérature — aucune alternative essayée ici"),
        regle("retrieve_k", c["retrieve_k"],
              ("candidats à soumettre au reranker AVANT la coupe à top_k. "
               "Valeur non vérifiée, et sans intérêt si aucun reranker n'est "
               "finalement installé" if actif
               else "sans reranker, les candidats récupérés sont ceux qu'on "
                    "garde : retrieve_k suit top_k. Non vérifié")),
        regle("reranker", reranker,
              ("RECOMMANDÉ par une règle de langue, jamais comparé au cas sans "
               "reranker. L'Advisor ne nomme aucun modèle : mesurez le vôtre "
               "avec mesure_reranker.py avant de l'adopter" if actif
               else "écarté par une règle, pas par une mesure : le gain n'a "
                    "pas été essayé sur ce corpus")),
        regle("gen_verification", c["gen_verification"],
              "second appel LLM — coût et gain non mesurés"),
    ]
    return sorties


def _choix_kag(c: dict, chunk_chars_mesure: int | None = None) -> list:
    """
    Les sorties KAG, chacune avec son niveau de preuve.

    LE MODÈLE D'EXTRACTION A DISPARU DE CETTE LISTE. Il y figurait en
    « contraint : seul modèle d'extraction téléchargé dans Ollama » — une
    phrase vraie sur une machine et fausse partout ailleurs. Il est désormais
    un argument obligatoire d'index_kag.py, et il apparaît dans la section
    « ce que l'Advisor ne décide pas » du rapport.

    Ce n'est pas un détail de présentation : le choix du modèle d'extraction
    pèse plus lourd que tous les paramètres ci-dessous réunis. Mesuré sur un
    même document, trois modèles ont donné 24, 86 et 131 exigences.
    """
    sorties = [
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

    # Même correction que côté RAG : quand un découpage est mesuré, il vaut des
    # CARACTÈRES. Laisser à côté une taille en tokens met deux unités dans la
    # même page pour la même chose.
    if chunk_chars_mesure:
        # La taille mesurée est DÉJÀ affichée par le bloc découpage, commun aux
        # deux architectures. La répéter sous un autre nom donnait deux lignes
        # pour la même valeur. On retire seulement la taille en tokens, devenue
        # caduque, et on porte la réserve propre au KAG dans un paramètre à
        # part — car elle mérite d'être lue, elle.
        sorties = [x for x in sorties if x.nom != "chunk_size"]
        # LE RECOUVREMENT KAG N'ÉTAIT AFFICHÉ NULLE PART.
        # Le RAG montrait chunk_overlap_chars, le KAG rien — pourtant
        # index_kag.py en applique un, à 10 % de la taille du morceau. Une
        # valeur appliquée mais jamais montrée est exactement ce que le
        # quatrième bloc du rapport doit éliminer.
        # La proportion diffère du RAG (10 % contre 15 %) : c'est un choix
        # d'origine, non vérifié, et la phrase le dit.
        sorties.insert(1, regle(
            "chunk_overlap_chars_kag", int(chunk_chars_mesure) // 10,
            f"10 % des {int(chunk_chars_mesure)} caractères du morceau. "
            f"Proportion plus faible que côté RAG (15 %), sans qu'aucune "
            f"mesure ne justifie l'écart : une relation coupée entre deux "
            f"morceaux est perdue définitivement, pas seulement moins bien "
            f"retrouvée"))
        sorties.insert(2, diagnostic(
            "decoupage:aptitude_relationnelle", "non mesurée",
            "chunk_quality note la PROPRETÉ d'un morceau, pas son aptitude à "
            "porter une relation. Un tableau coupé entre son en-tête et ses "
            "lignes reste propre et devient inexploitable pour l'extraction de "
            "triplets. Aucune mesure ne couvre ce risque"))
    return sorties


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
        # Un candidat qui n'a pas pu être noté est une OBSERVATION, pas un
        # paramètre : personne n'applique « decoupage:sémantique ».
        sortie.append(diagnostic(f"decoupage:{r['name']}", "non mesuré",
                                 r["raison"]))
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
    res = router.decide(corpus, mutability=mutability, probe=probe)
    archi = res["architecture"]

    # UNE DÉCISION SERRÉE N'EST PAS UNE DÉCISION.
    # Mesuré sur le corpus telecom : score 77 contre un seuil de 75, par le
    # comptage d'entités — la méthode que le routeur qualifie lui-même de peu
    # fiable, sur des signaux sous-estimés faute de couverture. Deux points
    # d'écart, dans ces conditions, ne distinguent rien : l'outil affichait
    # « KAG » avec le même aplomb qu'un score de 95.
    # On le dit désormais, au lieu de laisser croire à un verdict.
    score  = res["kag_suitability"]
    seuil  = res["kag_threshold"]
    marge  = abs(score - seuil)
    repli  = "repli" in (res.get("decision_source") or "").lower()
    serre  = marge <= (15 if repli else 8)

    if serre:
        # Le détail est développé dans l'encadré de la section 2 du rapport.
        # Le répéter ici en entier faisait lire deux fois la même chose.
        raison_archi = (f"DÉCISION SERRÉE : {score}/100 contre un seuil de "
                        f"{seuil}, soit {marge} point(s) d'écart"
                        + (" par la méthode de repli" if repli else "")
                        + " — voir l'avertissement en section 2")
    else:
        raison_archi = (f"score KAG {score}/100 contre un seuil de {seuil} — "
                        f"pondérations posées à la main, jamais confrontées à "
                        f"un graphe réel")

    tete = [regle("architecture", archi, raison_archi)]

    corps = _choix_decoupage(res["chunking"], dossier, embeddings)

    # Un découpage a-t-il vraiment été mesuré ? Si oui, la taille en tokens de
    # la littérature n'a plus lieu d'être affichée : elle décrirait un candidat
    # écarté, à côté de la taille réellement appliquée.
    chunk_chars_mesure = next((c.valeur for c in corps
                               if c.nom == "chunk_chars"), None)

    corps += _choix_rag(res["rag"]["versions"]["équilibré"]["config"],
                        chunk_chars_mesure) if archi == "RAG" \
        else _choix_kag(res["kag"]["config"], chunk_chars_mesure)

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
    # L'Advisor ne fournit plus qu'un TYPE. Le NOM du modèle vient de la
    # ligne de commande de l'indexation : on ne le devine pas ici.
    sortie["_embedding_type"]  = v.get("embedding_type")
    sortie["_embedding_dim"]   = None
    sortie["_index_type"]      = v.get("index_type")
    sortie["_metric_type"]     = v.get("distance")
    sortie["_n_chunks_est"]    = v.get("n_chunks_est")
    if v.get("chunk_strategy") and v.get("chunk_chars"):
        sortie["_chunk_strategy"] = v["chunk_strategy"]
        sortie["_chunk_chars"]    = int(v["chunk_chars"])
    if v.get("chunk_overlap_chars") is not None:
        sortie["_chunk_overlap_chars"] = int(v["chunk_overlap_chars"])
    if v.get("chunk_overlap_chars_kag") is not None:
        sortie["_chunk_overlap_chars_kag"] = int(v["chunk_overlap_chars_kag"])
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