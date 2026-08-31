# Banc d'essai — extraction de critères d'un appel d'offres

Généré le 17/08/2026 à 11:47. Liste de référence : 70 critères.

## Résultats

| Système | Rappel moyen | Écart entre runs | Précision | Annoncés | Contexte (mots) | Durée moy. |
|---|---|---|---|---|---|---|
| KAG | 50% | ±17% | 42% | 106 | 1989 | 682s |

## Comment lire ce tableau

- **Rappel** : part des critères de la liste de référence retrouvés dans la réponse. C'est la mesure principale.
- **Écart entre runs** : différence entre le meilleur et le pire run. Un écart large veut dire que la case ne conclut rien.
- **Précision** : part des critères annoncés qui correspondent à un critère réel de la liste de référence.
- **Annoncés** : nombre de lignes produites. Un rappel faible avec beaucoup de lignes annoncées signale un système bavard et imprécis ; avec peu de lignes, un système avare mais juste.
- **Contexte (mots)** : quantité d'information vue par le système. Si deux systèmes voient des quantités très différentes, la comparaison n'est pas à armes égales et il faut le dire.

## Coût, à ne pas oublier

| Système | Appels LLM à la construction | Appels LLM par question |
|---|---|---|
| RAG | 0 | 1 |
| KAG | 1 par morceau du corpus | 1 |
| Ligne de base | 0 | 1 |

Si le KAG répond mieux, ce n'est pas forcément le graphe qui est meilleur : c'est peut-être simplement que le LLM a déjà lu tout le corpus une fois, à la construction. Le RAG, lui, n'en voit que quelques morceaux au moment de la question.

## Ce qu'il reste à faire à la main

Les lignes « à vérifier » du fichier JSON ne sont PAS des inventions prouvées. Chacune est soit une invention, soit un critère absent de la liste de référence, soit une reformulation que les mots-clés n'ont pas reconnue. Il faut les lire une par une avant de conclure sur la précision.