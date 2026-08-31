"""
prompts_rfp.py — Requetes de recherche et prompt d'extraction, une famille par categorie.

Deux idees a retenir :

1. LES REQUETES SONT LE VRAI LEVIER.
   En RAG, un morceau jamais remonte est perdu pour toujours. On ecrit donc
   ~10 requetes par categorie, pas une seule, pour balayer le document.
   Les requetes sont en ANGLAIS parce que le document est en anglais.

2. LE PROMPT RAG N'EST PAS LE PROMPT "DOCUMENT ENTIER".
   En contexte plein, le modele s'arrete trop tot : il faut le pousser
   ("il y en a des dizaines"). En RAG il ne voit que 2 ou 3 bouts de texte :
   la meme consigne le ferait inventer. Ici on fait donc l'inverse —
   aucun objectif de volume, et la liste vide est une reponse correcte.

Les categories sont FERMEES : entreprise, equipe, solution, dossier.
Le modele ne doit jamais en creer une autre.
"""

CATEGORIES = {
    # ------------------------------------------------------------------
    "entreprise": {
        "definition": (
            "Requirements about the BIDDING COMPANY itself as a legal, financial "
            "and commercial entity: eligibility, age, tax and legal standing, "
            "certifications held by the company, financial health, past customer "
            "references, sector experience."
        ),
        "compte": [
            "The company must have existed for at least 5 years at the submission date.",
            "The bidder must provide two reference accounts with telecom operators.",
        ],
        "compte_pas": [
            "A team member must hold a CCNA certification. -> that is 'equipe'.",
            "The response must be submitted through IVALUA Sourcing. -> that is 'dossier'.",
        ],
        "requetes": [
            "eligibility criteria for legal persons",
            "years of existence of the company at submission date",
            "regular tax situation declarations settled",
            "compulsory liquidation receivership not eligible",
            "ISO 27001 certification of the company",
            "reference accounts telecom operators last 10 years",
            "experience in the telecommunication sector",
            "financial performance annual report profit growth",
            "bidder company background name address years in business",
            "warranties power and authority good industry practice",
        ],
    },
    # ------------------------------------------------------------------
    "equipe": {
        "definition": (
            "Requirements about the PEOPLE proposed by the vendor: positions, "
            "years of individual experience, personal certifications, academic "
            "degrees, technical skills, CVs, training of Ooredoo staff, local "
            "support presence, staff turnover."
        ),
        "compte": [
            "At least one team member must hold a CCNP certification.",
            "Team members must demonstrate 5 years of network engineering experience.",
        ],
        "compte_pas": [
            "The company must be ISO 27001 certified. -> that is 'entreprise'.",
            "The solution must produce a unified XML file. -> that is 'solution'.",
        ],
        "requetes": [
            "required position network billing systems engineer mandatory",
            "minimum years of professional experience per domain",
            "network engineering telecom billing systems experience",
            "required certifications CCNA CCNP ITIL lead implementer",
            "required technical skills proficiency level expert advanced",
            "Cisco IOS BGP OSPF Oracle SQL Java Linux skills",
            "academic formation degree master bachelor equivalence",
            "CVs of proposed team members education certifications copies",
            "project team roles responsibilities solution certified onsite offsite",
            "training services engineering support teams instructors French",
            "turnover of personnel first contract year percentage",
            "local support team displacement capacity offshore",
        ],
    },
    # ------------------------------------------------------------------
    "solution": {
        "definition": (
            "Requirements about the PRODUCT to be delivered: functional and "
            "technical capabilities, output formats, architecture, performance, "
            "technology freshness, supported release lifetime."
        ),
        "compte": [
            "The solution must produce the bill in PDF format.",
            "The target architecture must be in disaster recovery mode.",
        ],
        "compte_pas": [
            "The licensing model must be described in the proposal. -> that is 'dossier'.",
            "Instructors must be qualified and experienced. -> that is 'equipe'.",
        ],
        "requetes": [
            "graphical interactive interface for bill design customization",
            "produce unified XML file for each bill",
            "insert bill details in database tables statistics",
            "produce bill in PDF format send bills by email",
            "call details CSV format single PDF for splitted large accounts",
            "web application for solution management monitoring reporting",
            "target architecture disaster recovery active active passive",
            "industrialized standard preconfigured automated scalable reliable",
            "performance PDF generation slow server CPU memory usage",
            "solution based on old or obsolete technologies excluded",
            "proposed release supported at least 3 years modules dependency",
            "full automation full integration faster time to market",
        ],
    },
    # ------------------------------------------------------------------
    "dossier": {
        "definition": (
            "Requirements about the PROPOSAL ITSELF and the commercial terms: "
            "how to submit, file formats and names, compliance matrix, ordering "
            "of answers, signature, validity, pricing breakdown, licensing "
            "documentation, support SLA figures, exclusion causes, confidentiality."
        ),
        "compte": [
            "The offer must remain firm for 30 days from bid close date.",
            "Initial response to a critical incident: 1 hour.",
        ],
        "compte_pas": [
            "The solution must send bills by email. -> that is 'solution'.",
            "The company must not be in liquidation. -> that is 'entreprise'.",
        ],
        "requetes": [
            "responses sent through IVALUA sourcing system electronic copy",
            "marketing brochures will not be considered as responses",
            "deliverables format document name folder name PDF Excel",
            "requirements compliance matrix functional technical security",
            "compliance statement fully partially not compliant clause by clause",
            "document and page number cross reference in compliance column",
            "respond in the same order as sections rejected",
            "references to external documents or web sites not considered",
            "proposal signed by authorized official RFP number identified",
            "firm offers 30 days prices no greater than other customers",
            "budgetary pricing itemized model travel costs maintenance years 4 and 5",
            "total cost of ownership three years calculation details",
            "licensing model tiered PAYU named floating modular cost",
            "RACI matrix escalation matrix issue tracking system",
            "SLA response initial response workaround final solution working days",
            "excluded from tender evaluation commercial offer not opened",
            "non disclosure agreement confidentiality third parties",
            "late responses disqualified deadline",
        ],
    },
}


# ----------------------------------------------------------------------
# Le prompt d'extraction, specialise pour le RAG
# ----------------------------------------------------------------------

GABARIT = """/no_think
You extract requirements from a Request For Proposal (RFP) issued by a telecom operator.

You are given a few PASSAGES retrieved by a search engine. They are NOT the whole
document. They may be out of order, irrelevant, or cut in the middle of a sentence.

TASK
List only the requirements of the category "{categorie}" that are EXPLICITLY written
in the passages below.

CATEGORY "{categorie}"
{definition}

Counts as "{categorie}":
{compte}

Does NOT count:
{compte_pas}

RULES
1. If the passages contain no requirement of this category, answer exactly
   {{"criteres": []}}. An empty list is a CORRECT answer. Never invent a
   requirement to fill the list.
2. Never state a number of requirements to reach. Extract what is written, nothing more.
3. Treat each passage on its own. Do not combine two passages, and do not guess what
   comes before or after them. If a sentence is cut, ignore it.
4. One line = one requirement. Never merge two requirements into one line. A table row
   listing three skills is three requirements.
5. Copy figures, durations and proper names exactly as written: "5 years",
   "20k bills / Hour", "ISO 27001", "1 WD", "Bac+5".
6. "section" must be copied from the [number title] tag at the start of the passage.
   Never invent a section number.
7. "statut": "obligatoire" if the text says must / shall / required / mandatory /
   is not eligible; "recommande" if it says should / recommended / preferably /
   is recommended; "non precise" otherwise.
8. Answer in JSON only. No text before, no text after, no markdown fence, no comment.
9. Write "critere" and "valeur" in French. Keep technical names in their original form.

OUTPUT FORMAT
{{"criteres": [{{"categorie": "{categorie}", "critere": "...", "valeur": "...", "statut": "...", "section": "..."}}]}}

PASSAGES
{passages}
"""


def construire_prompt(categorie: str, passages: str) -> str:
    """Assemble le prompt final pour une categorie et un lot de passages."""
    c = CATEGORIES[categorie]
    return GABARIT.format(
        categorie=categorie,
        definition=c["definition"],
        compte="\n".join(f"  - {x}" for x in c["compte"]),
        compte_pas="\n".join(f"  - {x}" for x in c["compte_pas"]),
        passages=passages,
    )


def toutes_les_requetes():
    """[(categorie, requete), ...] pour toutes les categories."""
    return [(cat, r) for cat, c in CATEGORIES.items() for r in c["requetes"]]


if __name__ == "__main__":
    for cat, c in CATEGORIES.items():
        print(f"{cat:<12} {len(c['requetes'])} requetes")
    print(f"{'TOTAL':<12} {len(toutes_les_requetes())} requetes")
