# -*- coding: utf-8 -*-
"""
prompts.py - les prompts d'extraction (v8).

CE QUI CHANGE PAR RAPPORT A LA v7 : UN PROMPT PROPRE AU KAG.
============================================================

Le prompt RAG n'est pas retouche. On ajoute PROMPT_KAG a cote.

POURQUOI DEUX PROMPTS. Les versions v1 a v3 revendiquaient un prompt UNIQUE
entre RAG, KAG et ligne de base, au nom de la comparabilite : si le prompt
change, l'ecart mesure melange l'effet de l'architecture et celui du prompt.
Ce principe est abandonne depuis la v4, sur demande de l'encadrant : « le
prompt c'est pour un RAG, pas pour un LLM ».

Une fois ce pas franchi, il serait incoherent de donner au KAG un prompt ecrit
pour des extraits de texte. Le KAG ne voit pas de phrases : il voit des
triplets sujet -> relation -> objet, qui ont PERDU la phrase d'origine.
Lui dire « ne complete jamais une phrase tronquee » n'a aucun sens, et la
regle anti-copie non plus, puisqu'il n'y a plus de source a recopier.

A ECRIRE DANS LE RAPPORT. Le banc d'essai ne mesure donc plus des
architectures toutes choses egales par ailleurs : il mesure des CHAINES
COMPLETES, chacune avec le prompt qui lui convient. C'est defendable - c'est
ce qu'on deploierait en vrai - mais il faut le dire, sinon le chiffre est
trompeur.

CE QUI CHANGE DANS LE PROMPT KAG PAR RAPPORT AU PROMPT RAG
==========================================================
- Le contexte est presente comme un ensemble de FAITS, pas d'extraits.
- Le risque n'est plus la RECOPIE mais l'INVENTION : un triplet est pauvre, le
  modele est tente de reconstituer la phrase qu'il croit deviner. La consigne
  centrale devient « n'affirme que ce que les triplets affirment ».
- Les triplets peuvent etre du bruit. Mesure du 30/08 sur le graphe du RFP :
  environ 13 des 34 premiers predicats sont des titres du sommaire
  (definition, criteria, landscape, portfolio, warranties...). Le modele doit
  pouvoir les ignorer, donc la consigne le dit explicitement.
- La regle d'eclatement des enumerations reste, parce qu'un objet de triplet
  peut lui-meme contenir une liste.
- Les 10 categories, la regle d'arbitrage, les statuts et le format de sortie
  sont IDENTIQUES au prompt RAG. C'est ce qui permet de comparer les deux
  sorties ligne a ligne.

CONTEXTE DE MESURE DU GRAPHE (30/08), utile pour lire les resultats
==================================================================
11 morceaux de 3679 caracteres, 179 relations, 283 noeuds, 14,5 min de
construction. Mais 257 entites sur 283 - soit 91 % - ont un degre de 1 : elles
n'apparaissent que dans un seul triplet et ne se relient a rien. Ce n'est pas
un reseau, c'est un sac de faits isoles. Deux causes mesurees : la resolution
d'entites « basique » n'a fusionne que 4 entites sur 289, et
cross_doc_connectivity vaut 0 puisqu'il n'y a qu'un document.
Ne pas s'attendre a ce que le KAG exploite des chemins : il n'y en a pas.

CE QUI VIENT DES VERSIONS PRECEDENTES (cote RAG)
================================================
- v7 : REGLE ANTI-COPIE, six mots consecutifs maximum. Mesure : mediane de
  mots recopies 42 % -> 35 %, reformulation nette 27 % -> 39 %. La regle aide
  mais n'est pas respectee a la lettre (il reste des copies de 14 a 17 mots).
- v6 : LES 10 CATEGORIES. Les 5 du paragraphe 5.2 du RFP couvrent la SOLUTION
  mais pas le FOURNISSEUR : « operational » ramassait 70 % des exigences.
  Mesure v6 : le plus gros bloc tombe a 20 %.
- v5 : SPLIT INTERNAL ENUMERATIONS (paragraphe 10.0 de 1 a 6 lignes, total de
  124 a 160) et WHO IS BOUND (le glossaire du paragraphe 18 ne produit plus
  rien).
- v4 : PROMPT ECRIT POUR UN RAG, sans aucune ATTENTE DE VOLUME.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# PROMPT RAG
# ---------------------------------------------------------------------------
# Deux emplacements a remplir, dans cet ordre :
#   1er %s = l'etiquette autorisee (ex. « P7 »)
#   2e  %s = le passage etiquete

PROMPT_RAG = """You are analysing a telecom Request For Proposal (RFP).

You do NOT see the whole document. Below is ONE EXCERPT returned by a search \
engine. It is not in document order. It may start or end in the middle of a \
sentence. It may contain no requirement at all.

Your task: list every requirement stated in this excerpt.

A requirement is anything the bidder must do, provide, prove, respect or comply \
with: an obligation, a numeric threshold, a certification, a diploma, a \
duration, a deadline, an imposed format, a mandatory document, an eligibility \
condition, an exclusion condition, a technical capability.

WHO IS BOUND
A requirement constrains the BIDDER. Before writing a line, ask who must act.
NOT requirements, even when they sound formal:
- what Ooredoo itself does, owns, provides or has already decided (its history, \
its awards, its current systems, the hardware it will supply);
- glossaries, abbreviation lists and definitions;
- descriptions of the existing solution and its limitations;
- pointers to other documents ("the attached file describes...", "outlined in \
the attached file"): the pointer is not a deliverable, and the bidder is not \
being asked to supply that file;
- section headings alone, sentence fragments with no verb, general intentions \
("reliable solution", "good quality");
- the answer the supplier will later tick ("Fully compliant").
Never rewrite one of these as "The bidder must...". If nothing in the excerpt \
binds the bidder, return an empty list.

WORKING WITH ONE EXCERPT
- Extract only what is written in front of you. Never complete a truncated \
sentence, and never add anything from your general knowledge of tenders.
- Do not expect any particular number of requirements. A short excerpt may \
yield one, a table may yield fifteen, prose may yield none.

NEVER GROUP
"CCNA and CCNP required" gives TWO requirements. "5 years in networking and 3 \
years in billing" gives TWO requirements. A grouped requirement is a lost \
requirement.

SPLIT INTERNAL ENUMERATIONS
This is the most common mistake, so read it twice. A single sentence often \
hides several requirements behind a comma list or behind the words \
"including", "such as", "as follows", "and", or a colon. Produce ONE \
requirement PER ITEM of the list, never one sentence that repeats the list.

Wrong (one line):
  "The bidder must include company name and address, contact details, \
headquarter address, years in business and previous company names."
Right (five lines):
  "The bidder must provide its company name and address."
  "The bidder must provide a contact name, title, address, email and phone."
  "The bidder must provide its headquarter address if different."
  "The bidder must state the number of years in business under this name."
  "The bidder must state its previous company names."

Apply the same to bullet lists: every bullet is at least one requirement.

TABLES
A table row is a requirement in its own right. Read the column headers to \
understand the row: in a certification table, "Yes" in a "Mandatory" column \
makes that certification mandatory. One requirement per row, never a summary \
of the table.

CATEGORIES
Every requirement receives exactly one category from this closed list of ten. \
Never invent a category. Never leave it empty.

security
  Protection of information, and the ISO 27001 certification. Non-disclosure \
agreement, confidentiality of the RFP and its documents, disclosure to third \
parties and sub-contractors.
  ISO 27001 ALWAYS belongs here, whether it is required of the company or of a \
team member, and even when it appears in a table of professional \
certifications next to CCNA or ITIL.

infrastructure
  Hosting and physical or cloud resources. Disaster recovery mode \
(active-active or active-passive), hardware provided by the operator, compute, \
network and storage allocation from the private cloud, and anything the \
document places under an "IT Infrastructure" heading.

sla
  Response and resolution times only. Initial response, workaround, final \
solution, and the priority levels attached to them (critical, major, medium, \
minor).

hr
  The people the vendor assigns to the project. Mandatory positions, years of \
professional experience per domain, professional certifications (CCNA, CCNP, \
ITIL and any other EXCEPT ISO 27001), required proficiency levels in a named \
technology, academic degrees and their equivalences, CVs to be submitted, and \
personnel turnover limits.

financial
  Money. Budgetary quote, itemized pricing, travel costs, maintenance years \
included or charged separately, total cost of ownership, licensing model and \
schemes, per-module licence costs, price warranties, firm-offer duration, and \
the company's financial statements and profit history.

eligibility
  Whether the company is allowed to bid at all, and who it is. Years of \
existence, tax situation, liquidation and receivership, company profile and \
addresses, previous company names, contractual warranties, reference accounts, \
and experience in the telecom sector.

functional
  What the solution must produce or let a business user do. Bill design and \
support for future design changes, unified XML file per bill, PDF bill output, \
call details in CSV, sending bills by email, single PDF for splitted large \
accounts, graphical interactive interface for bill design customization.

technical
  How the solution is built and how it performs. Throughput and server \
resource usage, insertion of bill details into database tables, automation and \
integration capabilities, the web application for managing, running, \
monitoring and reporting.
  A certification or a person's skill level is NOT technical, it is hr. This \
category is about the SOLUTION, not about the team.

delivery
  Executing the project once won. Implementation and transition schedule, \
project team roles and responsibilities, onsite versus offsite activities, \
skills transfer and handover, methodology, tools, change control, \
communication methods, training services and materials, support and \
maintenance organisation, RACI and escalation matrices, issue tracking, local \
or offshore support teams.

process
  Applying for the tender. Deliverables and their formats, imposed document \
names and folder structure, submission channel, deadlines and late responses, \
order of sections, compliance matrix rules and compliance statements, grounds \
for exclusion from the evaluation, ownership of the responses, and signature \
of the proposal.
  Use process only when no other category applies.

TIE-BREAK
If a requirement seems to fit more than one category, apply this order and stop \
at the first match: security, infrastructure, sla, hr, financial, eligibility, \
functional, technical, delivery, process.

REFORMULATION
Rewrite each requirement as one clear sentence of your own. Do not copy the \
source sentence and do not simply prefix it with "The bidder must".

HARD LIMIT: never copy more than SIX consecutive words from the excerpt. If a \
passage cannot be shortened, change the order of its parts or split it into \
several requirements until no run of seven identical words remains.

The limit does not apply to a single unbreakable name: a certification \
("ISO 27001 Lead Implementer/Auditor"), a product, a file name, a proper noun. \
Copy those in full.

Keep unchanged: figures, units, dates, durations, percentages, acronyms, \
certification names, product names, proper nouns. Everything around them must \
be your own wording.

Wrong (18 consecutive words copied, only a prefix added):
  "The bidder must include the company name, contact name and title, phone \
number, email address and brief project description for each reference."
Right (split, and rephrased):
  "For each reference, the bidder must give the client company name."
  "For each reference, the bidder must name a contact person and their title."
  "For each reference, the bidder must give a phone number and an email address."
  "For each reference, the bidder must summarise the project briefly."

STATUS
The "status" field can ONLY be: mandatory, recommended, unspecified.
Map: must / shall / required / Yes -> mandatory; should / recommended / \
preferably -> recommended; nothing clear -> unspecified.
Never write the English word taken from the document in that field.

OUTPUT FORMAT
Answer with JSON only, no commentary, no markdown fences:
{"requirements": [{"passage": "...", "requirement": "...", "category": "...", \
"status": "..."}]}

"passage" must be exactly: %s
"category" must be exactly one of: security, infrastructure, sla, hr, \
financial, eligibility, functional, technical, delivery, process.

EXCERPT:
\"\"\"
%s
\"\"\"
"""


# ---------------------------------------------------------------------------
# PROMPT KAG
# ---------------------------------------------------------------------------
# Memes deux emplacements que le prompt RAG :
#   1er %s = l'etiquette autorisee (ex. « G3 »)
#   2e  %s = le groupe de triplets etiquete

PROMPT_KAG = """You are analysing facts extracted from a telecom Request For \
Proposal (RFP).

You do NOT see the document. Below is a GROUP OF FACTS pulled from a knowledge \
graph, each written as: subject -> relation -> object. These triplets were \
produced automatically from the RFP. The original sentences are gone.

Your task: list every requirement that these facts actually state.

A requirement is anything the bidder must do, provide, prove, respect or comply \
with: an obligation, a numeric threshold, a certification, a diploma, a \
duration, a deadline, an imposed format, a mandatory document, an eligibility \
condition, an exclusion condition, a technical capability.

NEVER INVENT
This is the main danger here. A triplet is poor and incomplete, and you will be \
tempted to reconstruct the sentence you think it came from. Do not.
- State only what the triplets assert. If a triplet gives a subject and an \
object but no clear obligation, it is not a requirement.
- Never add a threshold, a duration or a name that no triplet contains.
- Never merge two unrelated triplets into one invented requirement.

IGNORE THE NOISE
The automatic extraction also captured table-of-contents entries and section \
headings. Triplets whose relation is a bare noun such as "definition", \
"criteria", "description", "landscape", "portfolio", "architecture", \
"requirements", "schedule", "background", "warranties", "performance" or \
"cost" carry no obligation: skip them.
Also skip facts about Ooredoo itself (its history, its awards, its capital, its \
current systems and their limitations), and definitions of abbreviations. \
A requirement constrains the BIDDER.
If the group contains no requirement at all, return an empty list.

SPLIT ENUMERATIONS
A single triplet object may hide several requirements behind a comma list or \
behind "including", "such as", "and". Produce ONE requirement PER ITEM.
"required certifications -> are -> CCNA and CCNP" gives TWO requirements.

WRITING
Write each requirement as one clear, self-contained sentence in English. Keep \
figures, units, durations, percentages, acronyms, certification names and \
proper nouns exactly as the triplets give them.

CATEGORIES
Every requirement receives exactly one category from this closed list of ten. \
Never invent a category. Never leave it empty.

security
  Protection of information, and the ISO 27001 certification. Non-disclosure \
agreement, confidentiality of the RFP and its documents, disclosure to third \
parties and sub-contractors.
  ISO 27001 ALWAYS belongs here, whether it is required of the company or of a \
team member, and even when it sits next to CCNA or ITIL.

infrastructure
  Hosting and physical or cloud resources. Disaster recovery mode \
(active-active or active-passive), hardware provided by the operator, compute, \
network and storage allocation from the private cloud.

sla
  Response and resolution times only. Initial response, workaround, final \
solution, and the priority levels attached to them (critical, major, medium, \
minor).

hr
  The people the vendor assigns to the project. Mandatory positions, years of \
professional experience per domain, professional certifications (CCNA, CCNP, \
ITIL and any other EXCEPT ISO 27001), required proficiency levels in a named \
technology, academic degrees, CVs to be submitted, personnel turnover limits.

financial
  Money. Budgetary quote, itemized pricing, travel costs, maintenance years, \
total cost of ownership, licensing model and schemes, per-module licence costs, \
price warranties, firm-offer duration, financial statements and profit history.

eligibility
  Whether the company is allowed to bid at all, and who it is. Years of \
existence, tax situation, liquidation and receivership, company profile and \
addresses, previous company names, contractual warranties, reference accounts, \
experience in the telecom sector.

functional
  What the solution must produce or let a business user do. Bill design and \
future design changes, unified XML file per bill, PDF bill output, call details \
in CSV, sending bills by email, single PDF for splitted large accounts, \
graphical interface for bill design customization.

technical
  How the solution is built and how it performs. Throughput and server \
resource usage, insertion of bill details into database tables, automation and \
integration capabilities, the web application for managing, running, \
monitoring and reporting.
  A certification or a person's skill level is NOT technical, it is hr.

delivery
  Executing the project once won. Implementation and transition schedule, team \
roles, onsite versus offsite activities, skills transfer and handover, \
methodology, tools, change control, training services and materials, support \
and maintenance organisation, RACI and escalation matrices, issue tracking, \
local or offshore support teams.

process
  Applying for the tender. Deliverables and their formats, imposed document \
names and folder structure, submission channel, deadlines, order of sections, \
compliance matrix rules, grounds for exclusion, ownership of the responses, \
signature of the proposal.
  Use process only when no other category applies.

TIE-BREAK
If a requirement seems to fit more than one category, apply this order and stop \
at the first match: security, infrastructure, sla, hr, financial, eligibility, \
functional, technical, delivery, process.

STATUS
The "status" field can ONLY be: mandatory, recommended, unspecified.
Map: must / shall / required / Yes -> mandatory; should / recommended / \
preferably -> recommended; nothing clear -> unspecified.
Triplets often lose the modal verb. When nothing in the triplet indicates how \
strongly the thing is required, write unspecified rather than guessing.

OUTPUT FORMAT
Answer with JSON only, no commentary, no markdown fences:
{"requirements": [{"passage": "...", "requirement": "...", "category": "...", \
"status": "..."}]}

"passage" must be exactly: %s
"category" must be exactly one of: security, infrastructure, sla, hr, \
financial, eligibility, functional, technical, delivery, process.

FACTS:
\"\"\"
%s
\"\"\"
"""


PROMPTS = {
    "rag": PROMPT_RAG,
    "kag": PROMPT_KAG,
}

# format=json est demande a Ollama : le prompt exige du JSON, autant le
# contraindre au decodage plutot que d'esperer que le modele obeisse.
FORMAT_JSON = {
    "rag": True,
    "kag": True,
}

CLES_ATTENDUES = ("passage", "requirement", "category", "status")

# L'ORDRE EST CELUI DE LA REGLE D'ARBITRAGE, du plus specifique au plus general.
# Le repli du code doit utiliser CATEGORIE_DEFAUT, pas le premier element :
# une categorie non reconnue tombe dans « process », le defaut.
CATEGORIES = (
    "security",
    "infrastructure",
    "sla",
    "hr",
    "financial",
    "eligibility",
    "functional",
    "technical",
    "delivery",
    "process",
)

CATEGORIE_DEFAUT = "process"

STATUTS = ("mandatory", "recommended", "unspecified")


# ---------------------------------------------------------------------------
# REQUETES DE RECHERCHE (cote RAG uniquement)
# ---------------------------------------------------------------------------
# Ces requetes ne sont PAS le prompt. Elles servent uniquement a aller chercher
# les passages ; le prompt travaille ensuite sur chaque passage separement.
#
# MESURE DU 29/08 : ces 21 requetes avec retrieve_k=10 remontent 81 passages
# distincts sur les 89 de l'index, soit 91 % de couverture.
REQUETES_RECHERCHE = [
    # --- functional ---------------------------------------------------------
    "bill design changes and business requirements",
    "unified XML file, PDF bill and CSV call details",
    "single PDF bill for splitted large accounts, sending bills by email",
    # --- technical ----------------------------------------------------------
    "bill generation throughput and server resource usage",
    "web application for managing monitoring and reporting the solution",
    "required technical skills Cisco Oracle Java Linux XML",
    # --- process ------------------------------------------------------------
    "deliverables format, document names and folder structure",
    "compliance matrix statements and grounds for exclusion",
    # --- eligibility --------------------------------------------------------
    "bidder eligibility, years in business and financial performance",
    "client references and experience in the telecom sector",
    # --- hr -----------------------------------------------------------------
    "required certifications CCNA CCNP and academic degree",
    "years of professional experience of the project team",
    # --- sla ----------------------------------------------------------------
    "SLA initial response and final correction times",
    # --- delivery -----------------------------------------------------------
    "training services, number of persons and language",
    "support, maintenance, RACI and escalation matrix",
    "implementation schedule, project team and skills transfer",
    # --- financial ----------------------------------------------------------
    "pricing, licensing model, maintenance years and TCO",
    # --- security -----------------------------------------------------------
    "confidentiality, non-disclosure agreement and third parties",
    "ISO 27001 certification",
    # --- infrastructure -----------------------------------------------------
    "disaster recovery active-active or active-passive architecture",
    "hardware, compute network and storage from the private cloud",
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