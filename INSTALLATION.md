# Installation depuis zéro

Windows + PowerShell. Faites les étapes dans l'ordre, et vérifiez chacune avant
de passer à la suivante.

---

## 1. L'environnement Python

Placez-vous dans le dossier `stage2`, puis :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell refuse d'exécuter le script, autorisez-le pour cette session
seulement :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Vous devez voir `(.venv)` au début de la ligne. Vérifiez que c'est bien ce
Python qui répond :

```powershell
python -c "import sys; print(sys.executable)"
```

Le chemin affiché doit contenir `stage2\.venv`. S'il pointe ailleurs,
l'activation n'a pas fonctionné — ne continuez pas.

---

## 2. Les paquets

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
```

La dernière ligne est **obligatoire** et souvent oubliée. C'est le modèle de
reconnaissance d'entités ; ce n'est pas un paquet pip ordinaire, il ne peut pas
figurer dans `requirements.txt`.

Vérification :

```powershell
python -c "import spacy; spacy.load('xx_ent_wiki_sm'); print('spaCy OK')"
python -c "import pdfplumber, pymilvus, neo4j, streamlit, reportlab; print('paquets OK')"
```

`reportlab` sert au rapport PDF de l'Advisor. Sans lui, l'interface propose le
rapport en texte brut plutôt que de planter.

**Facultatif — le reranker.** `sentence-transformers` et `torch` pèsent environ
2,5 Go et ne servent qu'à `mesure_reranker.py`. Tout le reste tourne sans eux.

```powershell
pip install sentence-transformers torch
```

---

## 3. Le chemin des modules

`poc/` réutilise des modules d'`advisor2/` : lecture des documents, découpage,
accès à Milvus et Neo4j. Il faut donc rendre `advisor2/` visible avant de
lancer un script du PoC.

**À refaire dans chaque nouvelle fenêtre PowerShell :**

```powershell
$env:PYTHONPATH = "C:\chemin\vers\stage2\advisor2"
```

Sans cette ligne, les scripts de `poc/` s'arrêtent sur
`ModuleNotFoundError: No module named 'index_rag'`.

Les scripts d'`advisor2/` n'en ont pas besoin quand on les lance depuis leur
propre dossier.

Pour que VS Code cesse de signaler ces imports comme introuvables, un fichier
`.vscode/settings.json` contient déjà :

```json
{ "python.analysis.extraPaths": ["./advisor2"] }
```

---

## 4. Ollama — les modèles locaux

Ollama s'installe à part, depuis https://ollama.com. Ce n'est pas un paquet
Python.

```powershell
ollama serve          # à laisser tourner dans une fenêtre séparée
```

Il vous faut **deux modèles** : un pour les embeddings, un pour la génération.
Le projet n'en impose aucun — c'est un choix délibéré, expliqué plus bas.

```powershell
ollama pull <un-modele-d-embedding>
ollama pull <un-modele-de-generation>
```

Vérification :

```powershell
ollama list
curl http://localhost:11434/api/tags
```

### Aucun nom de modèle n'est écrit dans le code

C'est le parti pris principal du projet. Un nom de modèle décrit ce qui est
installé sur une machine, pas ce dont un corpus a besoin. Écrit en dur, il
devient faux dès qu'on change d'environnement, et il prend l'apparence d'une
recommandation alors que rien ne le justifie.

Les scripts attendent donc le modèle en argument, et s'arrêtent avec un message
clair si vous l'oubliez :

| Script | Argument |
|---|---|
| `probe.py` | `--model` |
| `index_rag.py` | `--embed-model` |
| `index_kag.py` | `--extraction-model` |
| `mesure_topk.py` | `--llm` et `--embedding` |
| `embed_compare.py` | `--models` (au moins deux) |
| `mesure_reranker.py` | `--reranker` et `--embedding` |

Dans l'interface Streamlit, le champ « Modèle Ollama » est vide au départ, pour
la même raison. Tant qu'il l'est, le sondage LLM ne part pas et la décision
retombe sur le comptage d'entités — méthode moins fiable, que le rapport
signale explicitement.

### Une note sur le matériel

Un modèle de 7 milliards de paramètres pèse environ 4,7 Go. Sur une carte de
4 Go de VRAM, il ne tient pas entièrement : une partie tourne sur le
processeur, et tout ralentit.

Les durées varient énormément d'un modèle à l'autre. Sur un même document et un
même matériel, l'extraction de triplets a pris entre 79 et 157 secondes par
morceau selon le modèle — un facteur deux qu'aucune formule ne prédit.

**Mesurez avant de vous engager.** `index_kag.py --max-chunks 3` analyse trois
morceaux et affiche les secondes par morceau. Multipliez, et vous saurez dans
quoi vous vous lancez.

---

## 5. Milvus et Neo4j — les bases

Elles tournent dans Docker.

```powershell
docker compose up -d
docker ps
```

Vous devez voir quatre conteneurs actifs : `neo4j`, `milvus-standalone`,
`milvus-minio` et `milvus-etcd`.

Le mot de passe Neo4j n'est pas dans le code. Il faut le poser dans
l'environnement, **à chaque nouvelle fenêtre PowerShell** :

```powershell
$env:NEO4J_PASSWORD = "votre_mot_de_passe"
```

Variables reconnues, toutes optionnelles sauf la dernière :

| Variable | Défaut |
|---|---|
| `OLLAMA_URL` | `http://localhost:11434` |
| `MILVUS_URI` | `http://localhost:19530` |
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASSWORD` | *(vide — à définir)* |
| `CHUNKER_EMBED_MODEL` | *(vide — sinon le découpage sémantique est écarté)* |

---

## 6. Vérifier que tout marche

### Sans Ollama ni Docker

Ces trois commandes ne demandent aucun service. Si elles passent, l'installation
Python est bonne.

```powershell
cd advisor2
python profiler.py --input ..\corpus\rfp --out profil.json
python router.py --profile profil.json
python chunk_quality.py --input ..\corpus\rfp --from-router profil.json --no-embed
```

Lancez-les **une par une**. Et n'écrivez jamais
`python profiler.py > profil.json` : PowerShell écrit alors le fichier en
UTF-16, illisible ensuite. C'est pour cela que `--out` existe.

### Avec Ollama

```powershell
python probe.py --input ..\corpus\rfp --model <votre-modele>
python chunk_quality.py --input ..\corpus\rfp --from-router profil.json
```

### L'interface complète

```powershell
cd advisor2
streamlit run app.py
```

Chargez des documents, saisissez un modèle Ollama dans la barre latérale, et
laissez l'outil dérouler. À la fin, un rapport PDF est téléchargeable.

### Le PoC d'extraction

```powershell
$env:PYTHONPATH = "C:\chemin\vers\stage2\advisor2"
cd poc
python query_rag.py --collection <votre-collection> --out exigences.json
```

---

## 7. Si ça casse

| Symptôme | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'index_rag'` | `$env:PYTHONPATH` manquant — voir l'étape 3. |
| `UnicodeDecodeError: 0xff in position 0` | Fichier écrit avec `>` de PowerShell, en UTF-16. Utilisez `--out`. |
| `Unexpected UTF-8 BOM` en lisant un JSON | `Set-Content -Encoding UTF8` ajoute un BOM que Python refuse. Utilisez `[System.IO.File]::WriteAllText(...)`. |
| `ner_available: false` | `python -m spacy download xx_ent_wiki_sm` manquant. |
| `sémantique : aucun morceau produit` | Ollama ne répond pas, ou aucun modèle d'embedding n'a été fourni. |
| `Ollama injoignable` | `ollama serve` n'est pas lancé. |
| `the following arguments are required: --embed-model` | Normal : aucun modèle par défaut. Voir l'étape 4. |
| Mot de passe Neo4j vide | `$env:NEO4J_PASSWORD` à redéfinir dans chaque fenêtre. |
| `field text not exist` côté Milvus | La collection interrogée a un autre schéma que celui attendu. |
| Réponse LLM vide, ou `Expecting ',' delimiter` | Sortie tronquée : le JSON coupé en plein milieu devient illisible. Augmentez `num_predict`. |
| PDF indisponible dans l'interface | `pip install reportlab`. |