# -*- coding: utf-8 -*-
"""
rapport.py — le rapport PDF de l'Advisor.

Ce que ce fichier produit, et pourquoi
--------------------------------------
L'écran de l'interface disparaît dès qu'on ferme l'onglet. Une configuration
qu'on ne peut pas relire trois semaines plus tard, ou montrer à un collègue,
n'est pas un livrable.

Le rapport reprend donc TOUT ce que l'interface affiche, avec la même règle :
chaque paramètre est présenté AVEC son niveau de preuve. Un `if` et une mesure
ne se lisent pas de la même façon sur le papier non plus.

Une section est ajoutée par rapport à l'écran : « ce que l'Advisor NE décide
PAS ». Elle liste explicitement les paramètres laissés à l'utilisateur — le
modèle, la fenêtre de contexte, le coût acceptable d'un reranker. Sans elle,
un lecteur pourrait croire que la configuration est complète, et découvrir les
manques au moment de l'indexation.

Dépendance
----------
reportlab. Si le paquet n'est pas là, `disponible()` renvoie False et
l'interface propose le rapport en texte brut plutôt que de planter :
    pip install reportlab
"""

from __future__ import annotations

import io
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
        TableStyle,
    )
    REPORTLAB = True
except ImportError:
    REPORTLAB = False


# Les quatre niveaux de preuve, dans l'ordre où ils comptent : ce qui est
# mesuré d'abord, ce qui n'est pas un choix en dernier.
NIVEAUX = [
    ("mesuré",      "Mesuré — comparé à des alternatives sur ce corpus",
     "Plusieurs candidats ont été essayés et notés. C'est le seul niveau où "
     "l'Advisor a vérifié son propre conseil."),
    ("conséquence", "Conséquence — découle d'une mesure",
     "Ces valeurs n'ont pas été comparées : elles suivent mécaniquement d'un "
     "paramètre qui l'a été."),
    ("réglé",       "Réglé — raisonné, jamais vérifié",
     "Une règle a été appliquée. Elle est défendable, elle n'a pas été "
     "confrontée à ce corpus."),
    ("contraint",   "Contraint — pas un choix",
     "Imposé par un outil ou une dépendance."),
    ("diagnostic",  "Diagnostic — observations, pas des réglages",
     "Ni valeurs ni choix : ces lignes expliquent pourquoi une mesure manque. "
     "Elles ne comptent pas dans le décompte ci-dessus."),
]

# Les diagnostics ne sont pas des paramètres : les inclure dans le total
# faisait paraître l'Advisor moins mesuré qu'il ne l'est.
NIVEAUX_PARAMETRES = ("mesuré", "conséquence", "réglé", "contraint")

NON_DECIDE = [
    ("Modèle de génération",
     "Aucun corpus ne dit quel modèle rédige le mieux. À choisir parmi ceux "
     "que sert votre installation."),
    ("Modèle d'embedding",
     "L'Advisor recommande un TYPE (multilingue ou monolingue). Le nom, la "
     "dimension et la vitesse dépendent de ce qui est installé."),
    ("Modèle d'extraction du graphe",
     "Même raison. Et l'écart entre deux modèles est énorme : mesuré à un "
     "facteur cinq sur le nombre d'exigences extraites d'un même document."),
    ("Fenêtre de contexte",
     "Propriété du modèle, pas du corpus. Elle conditionne la taille du "
     "contexte qu'on peut envoyer."),
    ("Coût acceptable d'un reranker",
     "Le GAIN se mesure sur le corpus, le COÛT dépend de votre carte "
     "graphique. Un même reranker peut valoir son prix ou non selon la "
     "machine."),
    ("Limite de génération (num_predict)",
     "Trop basse, le JSON est coupé en plein milieu et le morceau est compté "
     "comme vide, sans message d'erreur."),
]


def disponible() -> bool:
    """reportlab est-il installé ?"""
    return REPORTLAB


# ---------------------------------------------------------------------------
# VERSION TEXTE — toujours disponible, sans dépendance
# ---------------------------------------------------------------------------
def rapport_texte(corpus: dict, choix: list, architecture: str,
                  decision: dict | None = None,
                  probe: dict | None = None) -> str:
    """Le même contenu que le PDF, en texte brut. Sert de repli."""
    L = []
    L.append("RAPPORT DE CONFIGURATION — ADVISOR")
    L.append(f"Généré le {datetime.now():%d/%m/%Y à %H:%M}")
    L.append("")
    L.append("=" * 68)
    L.append("1. LE CORPUS")
    L.append("=" * 68)
    for cle, libelle in _LIGNES_CORPUS:
        if corpus.get(cle) is not None:
            L.append(f"  {libelle:<34} {_valeur(corpus, cle)}")

    L.append("")
    L.append("=" * 68)
    L.append(f"2. ARCHITECTURE RECOMMANDÉE : {architecture}")
    L.append("=" * 68)
    if decision:
        L.append(f"  Score KAG {decision.get('kag_suitability')}/100 "
                 f"contre un seuil de {decision.get('kag_threshold')}")
        L.append(f"  Décision fondée sur : {decision.get('decision_source')}")
        L.append("")
        for r in decision.get("decision_reasons", []):
            L.append(f"  - {r}")

    L.append("")
    L.append("=" * 68)
    L.append("3. CONFIGURATION, PAR NIVEAU DE PREUVE")
    L.append("=" * 68)
    for statut, titre, note in NIVEAUX:
        groupe = [c for c in choix if c.statut == statut]
        if not groupe:
            continue
        L.append("")
        L.append(titre.upper())
        L.append(f"  ({note})")
        for c in groupe:
            L.append(f"  · {c.phrase()}")

    L.append("")
    L.append("=" * 68)
    L.append("4. CE QUE L'ADVISOR NE DÉCIDE PAS")
    L.append("=" * 68)
    L.append("  Ces paramètres ne se déduisent d'aucun corpus : ils décrivent")
    L.append("  une installation. Les taire serait plus honnête que les inventer,")
    L.append("  mais il faut les choisir avant d'indexer.")
    L.append("")
    for nom, pourquoi in NON_DECIDE:
        L.append(f"  · {nom}")
        L.append(f"      {pourquoi}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# VERSION PDF
# ---------------------------------------------------------------------------
_LIGNES_CORPUS = [
    ("n_docs",                  "Documents"),
    ("total_words",             "Mots"),
    ("total_tokens_est",        "Tokens estimés"),
    ("avg_doc_words",           "Longueur moyenne"),
    ("languages",               "Langues"),
    ("docs_with_titles_frac",   "Documents avec titres"),
    ("section_chars_median",    "Taille médiane des sections"),
    ("table_chars_share",       "Part de texte en tableaux"),
    ("distinct_entities",       "Entités distinctes"),
    ("avg_entity_degree",       "Degré moyen des entités"),
    ("cross_doc_connectivity",  "Connectivité entre documents"),
    ("homogeneity",             "Régularité de forme"),
]


def _valeur(corpus: dict, cle: str) -> str:
    v = corpus.get(cle)
    if v is None:
        return "—"
    if isinstance(v, dict):
        return " · ".join(f"{k} {p:.0%}" for k, p in v.items())
    if isinstance(v, float):
        return f"{v:.0%}" if 0 <= v <= 1 and cle.endswith(("frac", "share",
                                                           "connectivity",
                                                           "homogeneity")) \
            else f"{v:.2f}"
    return f"{v:,}".replace(",", " ") if isinstance(v, int) else str(v)


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Titre", parent=s["Title"], fontSize=20,
                         spaceAfter=4, alignment=TA_LEFT))
    s.add(ParagraphStyle("SousTitre", parent=s["Normal"], fontSize=9,
                         textColor=colors.HexColor("#666666"), spaceAfter=18))
    s.add(ParagraphStyle("H", parent=s["Heading2"], fontSize=13,
                         textColor=colors.HexColor("#1a1a1a"),
                         spaceBefore=16, spaceAfter=6))
    s.add(ParagraphStyle("Note", parent=s["Normal"], fontSize=8.5,
                         textColor=colors.HexColor("#555555"),
                         spaceAfter=8, leading=11))
    s.add(ParagraphStyle("Corps", parent=s["Normal"], fontSize=9.5, leading=13))
    s.add(ParagraphStyle("Cellule", parent=s["Normal"], fontSize=8.5, leading=11))
    return s


def _tableau(donnees, largeurs, styles_sup=None):
    t = Table(donnees, colWidths=largeurs, repeatRows=1)
    base = [
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, 0), 0.7, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fafafa")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(base + (styles_sup or [])))
    return t


def rapport_pdf(corpus: dict, choix: list, architecture: str,
                decision: dict | None = None,
                probe: dict | None = None,
                nom_corpus: str = "") -> bytes:
    """Rend le rapport en PDF. Lève ImportError si reportlab manque."""
    if not REPORTLAB:
        raise ImportError("reportlab n'est pas installé : pip install reportlab")

    tampon = io.BytesIO()
    doc = SimpleDocTemplate(tampon, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Rapport de configuration — Advisor")
    s = _styles()
    story = []

    # --- en-tête ----------------------------------------------------------
    story.append(Paragraph("Rapport de configuration", s["Titre"]))
    sous = f"Advisor · généré le {datetime.now():%d/%m/%Y à %H:%M}"
    if nom_corpus:
        sous += f" · corpus : {nom_corpus}"
    story.append(Paragraph(sous, s["SousTitre"]))

    story.append(Paragraph(
        "Chaque paramètre est présenté <b>avec ce qui le soutient</b>. Une "
        "valeur mesurée et une valeur posée par une règle ne se lisent pas de "
        "la même façon, et rien dans ce rapport ne les confond.", s["Corps"]))
    story.append(Spacer(1, 10))

    # --- 1. le corpus -----------------------------------------------------
    story.append(Paragraph("1 · Le corpus", s["H"]))
    lignes = [["Mesure", "Valeur"]]
    lignes += [[lib, _valeur(corpus, cle)] for cle, lib in _LIGNES_CORPUS
               if corpus.get(cle) is not None]
    story.append(_tableau(lignes, [95 * mm, 79 * mm]))

    couv = corpus.get("ner_coverage")
    if couv is not None and couv < 0.95:
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"<b>Réserve.</b> La reconnaissance d'entités n'a porté que sur "
            f"{couv:.0%} du corpus. Les signaux d'entités — densité, entités "
            f"multi-documents, degré — sont donc sous-estimés, et le score KAG "
            f"avec eux.", s["Note"]))

    # --- 2. l'architecture ------------------------------------------------
    story.append(Paragraph(f"2 · Architecture recommandée : {architecture}", s["H"]))
    if decision:
        story.append(Paragraph(
            f"Score KAG <b>{decision.get('kag_suitability')}/100</b> contre un "
            f"seuil de <b>{decision.get('kag_threshold')}</b>. Décision fondée "
            f"sur : {decision.get('decision_source')}.", s["Corps"]))
        story.append(Spacer(1, 6))
        for r in decision.get("decision_reasons", []):
            story.append(Paragraph(f"• {r}", s["Corps"]))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 4))
        # Une marge de deux points sur une méthode de repli ne distingue rien.
        # L'écrire en petit à côté du reste reviendrait à le cacher.
        _score = decision.get("kag_suitability")
        _seuil = decision.get("kag_threshold")
        _repli = "repli" in (decision.get("decision_source") or "").lower()
        if _score is not None and _seuil is not None and \
                abs(_score - _seuil) <= (15 if _repli else 8):
            _t = Table([[Paragraph(
                f"<b>Décision serrée : {abs(_score - _seuil)} point(s) "
                f"d'écart avec le seuil.</b> "
                + ("Et fondée sur le comptage d'entités, méthode que le "
                   "routeur tient lui-même pour peu fiable. " if _repli else "")
                + "L'autre architecture n'est pas écartée : elle n'a pas été "
                  "essayée. Sur ce corpus, construire les deux et les comparer "
                  "n'est pas une amélioration possible, c'est la seule façon "
                  "de savoir.", s["Corps"])]], colWidths=[174 * mm])
            _t.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#fff4e5")),
                ("BOX",         (0, 0), (-1, -1), 0.8, colors.HexColor("#e0a030")),
                ("TOPPADDING",  (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(Spacer(1, 6))
            story.append(_t)
            story.append(Spacer(1, 6))

        story.append(Paragraph(
            "<b>Ce choix est une règle, pas une mesure.</b> Les pondérations du "
            "score ont été posées à la main et n'ont pas été confrontées à un "
            "graphe réel sur ce corpus. Construire les deux architectures et "
            "les comparer reste le seul moyen de le vérifier.", s["Note"]))

    if probe and probe.get("available"):
        story.append(Spacer(1, 8))
        lignes = [["Sondage LLM", "Valeur"],
                  ["Relations retenues", str(probe.get("relations_kept"))],
                  ["Relations / 1000 tokens",
                   str(probe.get("relations_per_1000_tokens"))],
                  ["Entités multi-documents",
                   f"{probe.get('cross_doc_entity_share', 0):.0%}"],
                  ["Types de relations réutilisés",
                   f"{probe.get('relation_reuse', 0):.0%}"],
                  ["Documents couverts",
                   f"{probe.get('docs_covered')} sur {probe.get('docs_total')}"],
                  ["Relations non vérifiées",
                   f"{probe.get('unverified_rate', 0):.0%}"]]
        story.append(_tableau(lignes, [95 * mm, 79 * mm]))
        if probe.get("docs_covered", 0) < probe.get("docs_total", 1):
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                "Le sondage n'a pas lu tous les documents : les signaux "
                "ci-dessus décrivent l'échantillon, pas le corpus entier.",
                s["Note"]))

    # --- 3. la configuration ---------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3 · Configuration, par niveau de preuve", s["H"]))

    compte = {st: len([c for c in choix if c.statut == st])
              for st, _, _ in NIVEAUX}
    total = sum(compte.get(st, 0) for st in NIVEAUX_PARAMETRES)
    story.append(Paragraph(
        f"<b>{compte.get('mesuré', 0)} paramètre(s) mesuré(s) sur {total}.</b> "
        f"Le reste est raisonné ou imposé — c'est écrit ici plutôt que masqué.",
        s["Corps"]))
    story.append(Spacer(1, 8))

    for statut, titre, note in NIVEAUX:
        groupe = [c for c in choix if c.statut == statut]
        if not groupe:
            continue
        bloc = [Paragraph(titre, s["H"]), Paragraph(note, s["Note"])]
        if statut == "diagnostic":
            lignes = [["Observation", "Ce qu'elle dit"]]
            for c in groupe:
                lignes.append([Paragraph(f"<b>{c.nom}</b>", s["Cellule"]),
                               Paragraph(c.raison, s["Cellule"])])
            bloc.append(_tableau(lignes, [52 * mm, 122 * mm]))
        else:
            lignes = [["Paramètre", "Valeur", "Sur quoi ça repose"]]
            for c in groupe:
                raison = c.phrase().split("—", 1)[-1].strip()
                lignes.append([
                    Paragraph(f"<b>{c.nom}</b>", s["Cellule"]),
                    Paragraph(str(c.valeur), s["Cellule"]),
                    Paragraph(raison, s["Cellule"]),
                ])
            bloc.append(_tableau(lignes, [38 * mm, 30 * mm, 106 * mm]))
        story.append(KeepTogether(bloc))
        story.append(Spacer(1, 6))

    # --- 4. ce qui n'est pas décidé --------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4 · Ce que l'Advisor ne décide pas", s["H"]))
    story.append(Paragraph(
        "Ces paramètres ne se déduisent d'aucun corpus : ils décrivent une "
        "<b>installation</b>, pas un texte. L'Advisor s'abstient plutôt que "
        "d'écrire une valeur qui serait vraie sur une machine et fausse sur "
        "toutes les autres — mais il faut les choisir avant d'indexer.",
        s["Corps"]))
    story.append(Spacer(1, 8))

    lignes = [["Paramètre", "Pourquoi il n'est pas décidé ici"]]
    lignes += [[Paragraph(f"<b>{n}</b>", s["Cellule"]),
                Paragraph(p, s["Cellule"])] for n, p in NON_DECIDE]
    story.append(_tableau(lignes, [52 * mm, 122 * mm]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Une estimation de durée fiable se mesure, elle ne se calcule pas.</b> "
        "Lancez trois morceaux, chronométrez, multipliez : c'est plus sûr que "
        "n'importe quelle formule. Sur un même document et un même matériel, "
        "deux modèles de taille comparable ont donné 79 et 157 secondes par "
        "morceau — un facteur deux qu'aucun modèle de coût ne prédit.",
        s["Note"]))

    doc.build(story)
    return tampon.getvalue()