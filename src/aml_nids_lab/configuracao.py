"""Parâmetros visíveis no início do notebook."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class ErroConfiguracao(ValueError):
    """Indica um parâmetro incompatível com a prática."""


CENARIOS_PAYLOAD = (
    ("payload_10_day17", "+10%"),
    ("payload_50_day17", "+50%"),
    ("payload_100_day17", "+100%"),
)
CONJUNTOS = ("clean_day17",) + tuple(
    nome for nome, _ in CENARIOS_PAYLOAD
)


@dataclass(frozen=True, slots=True)
class ParametrosLab:
    """Os seis parâmetros que o participante pode alterar."""

    n_amostras: int
    limiar: float
    epsilon: float
    semente: int
    grupo_caracteristicas: str
    iteracoes_pgd: int


def _inteiro_positivo(valor: Any, nome: str) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int) or valor <= 0:
        raise ErroConfiguracao(f"{nome} deve ser um inteiro positivo.")
    return valor


def _numero_positivo(valor: Any, nome: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErroConfiguracao(f"{nome} deve ser numérico.")
    numero = float(valor)
    if not math.isfinite(numero) or numero <= 0.0:
        raise ErroConfiguracao(f"{nome} deve ser positivo.")
    return numero


def criar_parametros(
    *,
    n_amostras: int,
    limiar: float,
    epsilon: float,
    semente: int,
    grupo_caracteristicas: str,
    iteracoes_pgd: int,
) -> ParametrosLab:
    """Valida e reúne os parâmetros definidos no notebook."""

    quantidade = _inteiro_positivo(n_amostras, "N_AMOSTRAS")
    limite = _numero_positivo(epsilon, "EPSILON")
    passos = _inteiro_positivo(iteracoes_pgd, "ITERACOES_PGD")
    if isinstance(semente, bool) or not isinstance(semente, int) or semente < 0:
        raise ErroConfiguracao("SEMENTE deve ser um inteiro não negativo.")
    if isinstance(limiar, bool) or not isinstance(limiar, (int, float)):
        raise ErroConfiguracao("LIMIAR_DECISAO deve ser numérico.")
    limiar_numerico = float(limiar)
    if not math.isfinite(limiar_numerico) or not 0.0 < limiar_numerico < 1.0:
        raise ErroConfiguracao("LIMIAR_DECISAO deve estar entre zero e um.")
    grupo = (
        grupo_caracteristicas.strip().casefold()
        if isinstance(grupo_caracteristicas, str)
        else ""
    )
    if grupo not in {"volume", "tempo", "todas"}:
        raise ErroConfiguracao(
            "GRUPO_CARACTERISTICAS deve ser 'volume', 'tempo' ou 'todas'."
        )
    return ParametrosLab(
        n_amostras=quantidade,
        limiar=limiar_numerico,
        epsilon=limite,
        semente=semente,
        grupo_caracteristicas=grupo,
        iteracoes_pgd=passos,
    )
