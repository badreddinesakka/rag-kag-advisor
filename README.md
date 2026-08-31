# RAG & KAG — Advisor et extraction d'exigences

Stage de deuxième année, Ooredoo Tunisie · ENIT / UTM · 2025-2026

Deux systèmes qui répondent à des questions sur un ensemble de documents, et
un outil qui aide à choisir entre les deux.

---

## Le problème

Vous avez des documents. Vous voulez un système qui réponde à des questions
dessus. Deux approches existent.

**RAG** — on découpe les documents, on range les morceaux dans une base
vectorielle, et à chaque question on va chercher les passages qui ressemblent.

**KAG** — on lit tous les documents une fois, on en extrait des faits sous
forme de triplets, on construit un graphe de connaissances, et on interroge ce
graphe.

Laquelle choisir ? Ça dépend des documents. Et jusqu'ici, ça se décidait à
l'intuition.

---

## Le projet, en deux parties

### 1 · L'Advisor — `advisor2/`

Un outil qui lit un corpus et recommande une configuration : RAG ou KAG, avec
la stratégie de découpage, la taille des morceaux et les paramètres associés.

Sa particularité tient en une idée : **chaque paramètre est affiché avec ce qui
le justifie.** Un réglage issu d'une mesure et un réglage posé par une règle ne
se lisent pas de la même façon, et l'outil ne les confond jamais.

Cinq niveaux de preuve :

| Niveau | Ce que ça veut dire |
|---|---|
| **Mesuré** | plusieurs candidats essayés et notés sur ce corpus |
| **Conséquence** | découle mécaniquement d'un paramètre mesuré |
| **Réglé** | une règle a été appliquée, jamais vérifiée ici |
| **Contraint** | imposé par un outil, aucun choix possible |
| **Diagnostic** | une observation, pas un réglage |

L'outil dit aussi ce qu'il **ne** décide **pas**. Le nom d'un modèle, la taille
d'une fenêtre de contexte ou la puissance d'une carte graphique décrivent une
installation, pas un corpus : ces valeurs sont demandées à l'utilisateur plutôt
qu'inventées.

À la fin, un rapport PDF téléchargeable reprend tout, y compris les réserves.

**Interface :**

```bash
cd advisor2
streamlit run app.py
```

### 2 · L'extraction d'exigences — `poc/`

Un cas concret : extraire automatiquement les exigences d'un appel d'offres
(*Request For Proposal*) et les ranger par catégorie.

Le document est un RFP réel : quinze pages, une dizaine de tableaux, des
exigences dispersées entre le texte courant, les listes à puces et les
tableaux.

Deux chaînes ont été construites sur ce même document, **une par
architecture**, pour pouvoir les comparer :

- une chaîne RAG, sur un index vectoriel Milvus
- une chaîne KAG, sur un graphe Neo4j

Chacune produit un fichier JSON : une ligne par exigence, avec sa catégorie,
son caractère obligatoire ou non, et la section du document dont elle vient.

Une troisième brique évalue les deux sorties sans jeu de test écrit à la main :
elle relit chaque exigence produite en la confrontant au document d'origine.

---

## Structure du dépôt

```
advisor2/     l'Advisor : profil du corpus, décision, rapport PDF
poc/          l'extraction d'exigences : RAG, KAG, évaluation
  archives/   travaux antérieurs conservés pour référence
corpus/       les documents de test
```

`poc/` utilise des modules d'`advisor2/` (lecture des documents, découpage,
accès à Milvus et Neo4j). Il faut donc rendre `advisor2/` visible avant de
lancer les scripts du PoC :

```bash
export PYTHONPATH=./advisor2      # Linux, macOS
$env:PYTHONPATH = ".\advisor2"    # Windows PowerShell
```

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
```

Trois services doivent tourner à côté, ils ne s'installent pas avec pip :

- **Ollama** — sert les modèles de langue et d'embedding
- **Milvus** — la base vectorielle, côté RAG
- **Neo4j** — la base de graphe, côté KAG

Le détail est dans [INSTALLATION.md](INSTALLATION.md).

---

## Un mot sur les choix

Deux partis pris traversent tout le projet.

**Aucun nom de modèle n'est écrit dans le code.** Un nom de modèle décrit ce
qui est installé sur une machine. Écrit en dur, il devient faux dès qu'on
change d'environnement — et il a l'apparence d'une recommandation alors que
rien ne le justifie. Les modèles sont donc des arguments obligatoires : les
scripts s'arrêtent avec un message clair plutôt que de travailler avec un
modèle que personne n'a choisi.

**Ce qui n'est pas mesuré est annoncé comme tel.** L'Advisor pourrait afficher
une configuration complète et lisse. Il préfère dire « cette valeur vient d'une
règle, elle n'a pas été vérifiée sur votre corpus ». C'est moins spectaculaire,
et beaucoup plus utile à qui doit décider ensuite.

---

## Licence et usage

Travail réalisé dans le cadre d'un stage. Les documents du corpus ne sont pas
inclus dans ce dépôt.