# advisor2

Version de l'Advisor dont **aucune décision ne dépend de la machine**.

`advisor/` reste intact à côté : c'est l'archive de la version précédente.

---

## Le principe

L'Advisor décide à partir du **corpus**, et de rien d'autre.

Un nom de modèle, une taille de fenêtre, une vitesse de carte graphique
décrivent une installation. Écrits en dur, ils sont vrais sur une machine et
faux sur toutes les autres — et l'utilisateur ne le voit pas, parce qu'ils ont
la forme d'une recommandation.

Conséquence : **si une valeur ne se déduit pas du corpus, l'Advisor ne la
décide pas.** Elle devient un argument obligatoire, ou elle disparaît.

---

## Ce qui a changé par rapport à `advisor/`

### La troisième architecture « CONTEXTE » est supprimée

`router_preuve.py` répondait parfois « n'indexe rien : tout tient dans la
fenêtre du modèle ». La règle est bonne. Mais elle avait besoin de la taille de
cette fenêtre — 32 768 en dur — pour trancher. Ce nombre décrit une
installation, pas un corpus.

Quatre constantes disparaissent avec elle : `MODELE_GENERATION`,
`FENETRE_TOKENS`, `PART_RESERVEE`, `PART_SERREE`.

L'Advisor répond désormais RAG ou KAG.

### Plus aucun nom de modèle dans le code

| Fichier | Avant | Maintenant |
|---|---|---|
| `router.py` | `EMBEDDING_MODEL = "bge-m3"` | recommande un **type** : multilingue ou monolingue |
| `router.py` | `EXTRACTION_MODEL_HEAVY/LIGHT` | supprimés |
| `probe.py` | `DEFAULT_MODEL = "qwen2.5:7b"` | `--model` obligatoire |
| `mesure_topk.py` | `MODELE_LLM`, `MODELE_EMB` | `--llm` et `--embedding` obligatoires |
| `embed_compare.py` | `MODELES_DEFAUT = [...]` | `--models` obligatoire |
| `mesure_reranker.py` | `RERANKER_DEFAUT = "BAAI/..."` | `--reranker` et `--embedding` obligatoires |
| `chunker.py` | repli sur `bge-m3` | erreur explicite si aucun modèle |
| `index_rag.py` | repli sur `bge-m3` | erreur explicite |
| `index_kag.py` | repli sur `qwen2.5:7b` | `--extraction-model` obligatoire |

Le principe est le même partout : **échouer avec un message clair plutôt que
travailler avec un modèle que personne n'a choisi.**

### L'embedding devient un type, pas un nom

`router.py` répond `multilingue` ou `monolingue`. Le corpus dit s'il faut un
modèle capable de plusieurs langues ; il ne dit rien sur ce qui est installé.

La dimension du vecteur n'est plus imposée non plus : elle dépend du modèle que
l'utilisateur choisit.

---

## Le seul fichier qui dépend encore de la machine

`estimator.py`, et il le dit lui-même en tête.

Un temps de calcul ne se déduit pas d'un texte. Ses constantes ont donc le
statut le plus faible du projet : ni mesurées, ni universelles.

Elles ont au moins été **recalibrées** sur des mesures réelles. La version
précédente annonçait des minutes pour une construction KAG qui en a pris des
heures.

Mesures du 30/08, 11 morceaux de 3 679 caractères, RTX A2000 4 Go :

| Modèle | Secondes par morceau |
|---|---|
| qwen2.5:7b | 79 |
| mistral:7b | 110 |
| qwen3:8b | 157 |

Un facteur deux entre deux modèles de taille comparable, sur le même matériel
et le même texte. Aucune formule ne prédit cela.

**La seule estimation fiable reste celle qu'on mesure** : lancer trois morceaux
avec `--max-chunks 3`, chronométrer, multiplier. `index_kag.py` affiche les
secondes par morceau à la fin de chaque run.

---

## Utilisation

```bash
export PYTHONPATH=chemin/vers/advisor2

# 1. profiler le corpus
python profiler.py --input corpus/ --out profil.json

# 2. sonder (le modèle est obligatoire)
python probe.py --input corpus/ --model <ton-modèle-ollama>

# 3. décider
python router_preuve.py --profile profil.json --input corpus/ --out config.json

# 4. indexer
python index_rag.py --input corpus/ --config config.json --embed-model <ton-modèle>
python index_kag.py --input corpus/ --config config.json --extraction-model <ton-modèle>

# 5. mesurer (facultatif)
python embed_compare.py   --input corpus/ --models <m1> <m2>
python mesure_topk.py     --input corpus/ --llm <m> --embedding <m>
python mesure_reranker.py --input corpus/ --questions questions_auto.json \
                          --reranker <m> --embedding <m>
```

L'interface :

```bash
streamlit run app.py
```

---

## Ce que l'Advisor décide, et ce qu'il ne décide pas

**Il décide** — l'architecture RAG ou KAG, la stratégie de découpage, la taille
des morceaux, le nombre de morceaux attendu, le mode d'ontologie, la résolution
d'entités, la détection de communautés, le type de modèle d'embedding, le top-k,
le gain d'un reranker.

**Il ne décide pas** — quel modèle utiliser, quelle fenêtre de contexte, si le
coût d'un reranker est acceptable, quelle limite de génération poser.

Ces quatre-là ne se lisent pas dans un corpus. Les taire est plus honnête que
les inventer.
