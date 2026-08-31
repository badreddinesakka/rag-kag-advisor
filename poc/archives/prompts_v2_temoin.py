# -*- coding: utf-8 -*-
"""
prompts.py — le prompt du banc d'essai (v2).

CE QUI CHANGE PAR RAPPORT A LA v1
=================================
Le prompt court est SUPPRIME. Il ne produisait pas de liste identifiable, donc
sa precision n'etait pas mesurable : une case sur deux du tableau restait vide.
Le supprimer divise par deux le temps de calcul sans rien perdre.

Le prompt detaille est retravaille sur quatre points, tires de ce que le banc
d'essai a mesure :

1. INTERDICTION DE REGROUPER. C'est le levier le plus fort. Dans
   extract_criteres.py, cette seule consigne fait passer de 17 lignes a 158 :
   « CCNA et CCNP obligatoires » doit donner DEUX lignes, pas une. Sans elle,
   le modele condense et le rappel s'effondre.

2. ATTENTE DE VOLUME ANNONCEE. Le modele s'arretait spontanement vers 17-19
   lignes. Lui dire qu'un appel d'offres en contient plusieurs dizaines le
   pousse a continuer au lieu de conclure.

3. PARCOURS ORDONNE. « Du debut a la fin, passage par passage » : le modele
   traite mal le milieu d'un long contexte (mesure : la ligne de base, qui voit
   tout le document, plafonne a 26 %). Un ordre de parcours explicite limite la
   casse.

4. STATUT A TROIS VALEURS IMPOSEES. La v1 laissait le champ libre : le modele
   recopiait le mot du document (« must », « shall », « required »,
   « preferably ») et on obtenait douze valeurs differentes la ou le RFP n'en
   connait que trois.

Le prompt reste STRICTEMENT IDENTIQUE entre RAG, KAG et ligne de base. Le mot
« CONTEXTE » est volontairement neutre : ecrire « EXTRAITS » d'un cote et
« FAITS » de l'autre suffirait a casser la comparaison.
"""

from __future__ import annotations

PROMPT_DETAILLE = """Tu es un analyste d'appel d'offres. On construit une \
MATRICE DE CONFORMITÉ : une ligne par exigence, que le fournisseur devra ensuite \
cocher « conforme » ou « non conforme ».

Recense TOUTES les exigences présentes dans le CONTEXTE ci-dessous.

MÉTHODE
Parcours le CONTEXTE du début à la fin, passage par passage. Pour chaque \
passage, note ce qu'il exige avant de passer au suivant. Ne survole pas la fin.

CE QU'EST UNE EXIGENCE
Une contrainte VÉRIFIABLE : un seuil chiffré, une certification, un diplôme, \
une durée, un délai, un format imposé, un livrable, une condition d'exclusion, \
une capacité technique. Chaque puce, chaque ligne de tableau, chaque phrase \
contenant « must », « shall », « should », « required » ou « Yes » en est une.

CE QUI N'EN EST PAS
Une intention générale (« solution fiable », « bonne qualité »), un titre de \
section seul, un fragment de phrase sans verbe, ou la réponse attendue du \
fournisseur (« Fully compliant ») — ce sont des cases à remplir, pas des \
exigences.

RÈGLES STRICTES
- NE REGROUPE JAMAIS. « CCNA et CCNP obligatoires » donne DEUX lignes. \
« 5 ans en réseau et 3 ans en facturation » donne DEUX lignes. Une exigence \
groupée est une exigence perdue.
- Chaque ligne doit se comprendre seule, sans relire le document.
- Recopie les valeurs exactes : 5 ans, 3 WD, 20k bills/hour, ISO 27001, Bac+5.
- Le champ « statut » ne peut valoir QUE : obligatoire, recommande, \
non precise. Traduis : must / shall / required / mandatory / Yes → \
obligatoire ; should / recommended / preferably → recommande ; rien de clair \
→ non precise. N'écris jamais le mot anglais du document.
- N'écris QUE ce qui figure dans le CONTEXTE. N'ajoute rien qui vienne de tes \
connaissances générales sur les appels d'offres.

VOLUME ATTENDU
Un appel d'offres contient plusieurs dizaines d'exigences. Si tu en as listé \
moins de vingt, c'est que tu as sauté des passages : reprends le CONTEXTE et \
continue. Ne t'arrête que lorsque tu as parcouru le dernier passage.

FORMAT
Réponds UNIQUEMENT avec du JSON, sans commentaire :
{"criteres": [{"critere": "...", "valeur": "...", "statut": "...", \
"categorie": "..."}]}

categorie parmi : entreprise, equipe, solution, dossier, sla, cout.
valeur : le seuil, la durée ou le nom exact ; "—" s'il n'y en a pas.

CONTEXTE :
\"\"\"
%s
\"\"\"
"""


PROMPTS = {
    "detaille": PROMPT_DETAILLE,
}

# format=json est demande a Ollama : le prompt exige du JSON, autant le
# contraindre au niveau du decodage plutot que d'esperer que le modele obeisse.
FORMAT_JSON = {
    "detaille": True,
}


# ---------------------------------------------------------------------------
# REQUETES DE RECHERCHE (cote RAG uniquement)
# ---------------------------------------------------------------------------
# « quels sont les criteres ? » ressemble a tout et a rien : la recherche
# vectorielle ne trouve rien de precis. On lance plusieurs recherches courtes,
# une par famille d'exigences, et on fusionne les passages obtenus.
#
# Ces requetes ne sont PAS le prompt : elles ne servent qu'a aller chercher le
# contexte. Le KAG utilise les memes mots pour filtrer le graphe.
#
# EQUILIBRAGE (v2) : la v1 comptait six requetes sur le fournisseur et l'equipe
# pour une seule sur la solution. Le rappel mesure s'en ressentait — 77 % sur la
# categorie « equipe » contre 21 % sur « solution ». Les familles sont
# desormais couvertes a parts egales.
REQUETES_RECHERCHE = [
    # entreprise
    "critères d'éligibilité du fournisseur",
    "références clients et expérience télécom",
    # équipe
    "certifications requises CCNA CCNP ISO 27001",
    "années d'expérience et diplôme exigés",
    # solution
    "exigences techniques de la solution",
    "format des fichiers produits XML PDF CSV",
    "architecture, performance et reprise après sinistre",
    "interface de personnalisation et application de gestion",
    # dossier
    "livrables et format de la réponse",
    "matrice de conformité et motifs d'exclusion",
    # sla et coût
    "délais SLA de réponse et de correction",
    "coûts, licences, maintenance et TCO",
]

# Mots-cles derives des requetes, pour filtrer le graphe KAG.
MOTS_CLES_GRAPHE = [
    "critere", "criteria", "eligib", "certif", "ccna", "ccnp", "iso", "itil",
    "experience", "ans", "years", "diplome", "degree", "bac", "master",
    "exclu", "disqualif", "compliant", "conformite", "sla", "response",
    "livrable", "deliverable", "cout", "cost", "licence", "license",
    "maintenance", "reference", "telecom", "vendor", "bidder", "fournisseur",
    "obligatoire", "mandatory", "requis", "required", "must", "shall",
    "xml", "pdf", "csv", "architecture", "performance", "support",
]
