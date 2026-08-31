# -*- coding: utf-8 -*-
"""
decision.py — la brique de base de l'Advisor.

Un paramètre de config n'est pas juste une valeur : c'est une valeur PLUS ce qui
la soutient. Aujourd'hui l'Advisor affiche « top_k = 5 » et « découpage
structurel » de la même façon, alors que le premier vient d'un `if` et le second
d'une mesure. Ce module force à dire lequel est lequel.

Trois niveaux, et rien d'autre :
  contraint — pas un choix (matériel, outil, dépendance manquante)
  réglé     — un choix raisonné, jamais vérifié sur ce corpus
  mesuré    — comparé à des alternatives sur ce corpus

Ce fichier ne mesure rien lui-même. Il enregistre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONTRAINT   = "contraint"
REGLE       = "réglé"
MESURE      = "mesuré"
CONSEQUENCE = "conséquence"   # découle d'un paramètre mesuré


@dataclass
class Choix:
    nom: str
    valeur: object
    statut: str
    raison: str
    essais: dict = field(default_factory=dict)   # candidat -> note, si mesuré
    egalite: bool = False                        # aucun gagnant net

    def phrase(self) -> str:
        """Une ligne lisible, formulée différemment selon le niveau de preuve."""
        if self.statut == CONTRAINT:
            return f"{self.nom} = {self.valeur} — imposé : {self.raison}"
        if self.statut == REGLE:
            return f"{self.nom} = {self.valeur} — NON MESURÉ : {self.raison}"
        if self.statut == CONSEQUENCE:
            return f"{self.nom} = {self.valeur} — découle d'une mesure : {self.raison}"
        detail = ", ".join(f"{k} {v:.3f}" for k, v in self.essais.items())
        # Une raison explicite l'emporte sur la phrase automatique : certains
        # verdicts ne se résument pas à « il gagne » — un candidat peut avoir la
        # meilleure note ET être écarté pour son coût.
        if self.raison:
            return f"{self.nom} = {self.valeur} — {self.raison} ({detail})."
        if self.egalite:
            return (f"{self.nom} = {self.valeur} — égalité entre candidats "
                    f"({detail}) : on garde le plus simple.")
        return f"{self.nom} = {self.valeur} — mesuré, il gagne ({detail})."


def contraint(nom: str, valeur, raison: str) -> Choix:
    return Choix(nom, valeur, CONTRAINT, raison)


def regle(nom: str, valeur, raison: str) -> Choix:
    return Choix(nom, valeur, REGLE, raison)


def consequence(nom: str, valeur, raison: str) -> Choix:
    """
    Ni un choix, ni une mesure : une valeur qui DÉCOULE d'un paramètre mesuré.

    La taille des morceaux n'a pas été comparée à des alternatives — c'est la
    taille du découpage qui a gagné. La ranger en « contraint » ferait croire
    qu'aucun choix n'était possible, et effacerait la mesure qui l'a produite.
    """
    return Choix(nom, valeur, CONSEQUENCE, raison)


def mesure(nom: str, essais: list[tuple], ecart_min: float = 0.03) -> Choix:
    """
    essais : [(candidat, note), ...] DANS L'ORDRE DU PLUS SIMPLE AU PLUS COMPLEXE.
             Cet ordre est ce qui départage en cas d'égalité.

    Règle d'égalité : si le premier et le deuxième sont séparés de moins de
    `ecart_min`, on refuse de couronner. Sur quelques dizaines d'éléments, un
    écart plus petit n'est pas distinguable du bruit. On garde alors le candidat
    le plus simple, et on le dit.
    """
    if not essais:
        raise ValueError(f"{nom} : aucun essai, impossible de mesurer.")

    rang = {c: i for i, (c, _) in enumerate(essais)}
    classe = sorted(essais, key=lambda cn: (-cn[1], rang[cn[0]]))

    egalite = len(classe) > 1 and (classe[0][1] - classe[1][1]) < ecart_min
    gagnant = min((c for c, _ in classe[:2]), key=lambda c: rang[c]) if egalite else classe[0][0]

    return Choix(nom, gagnant, MESURE, "", dict(essais), egalite)


def resume(choix: list[Choix]) -> str:
    """La config, groupée par niveau de preuve. C'est le livrable de l'Advisor."""
    lignes = []
    for statut, titre in ((MESURE,      "MESURÉ — comparé sur ce corpus"),
                          (CONSEQUENCE, "CONSÉQUENCE — découle d'une mesure"),
                          (REGLE,       "RÉGLÉ — raisonné, jamais vérifié"),
                          (CONTRAINT,   "CONTRAINT — pas un choix")):
        groupe = [c for c in choix if c.statut == statut]
        if not groupe:
            continue
        lignes.append(f"\n{titre}")
        lignes += [f"  · {c.phrase()}" for c in groupe]
    return "\n".join(lignes)


def config(choix: list[Choix]) -> dict:
    """Les valeurs seules, pour le code en aval (index_rag, index_kag)."""
    return {c.nom: c.valeur for c in choix}


if __name__ == "__main__":
    exemple = [
        contraint("embedding_model", "bge-m3", "seul modèle servi par Ollama"),
        regle("chunk_overlap", 98, "15 % de la taille, usage courant"),
        mesure("decoupage", [("fixe", 0.871), ("structurel", 0.828)]),
        mesure("top_k", [("3", 0.62), ("5", 0.81), ("10", 0.82)]),
    ]
    print(resume(exemple))