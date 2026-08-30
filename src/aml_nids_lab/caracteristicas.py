"""Características e escalonamento usados pelo detector.

O detector recebe 49 valores na ordem registrada em ``model/artifact.json``.
Este módulo mantém essa ordem, os grupos usados pelos ataques e a transformação
MinMax aplicada antes da MLP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ErroCaracteristicas(ValueError):
    """Indica uma definição de características ou uma matriz incompatível."""


def ler_artefato(raiz: str | Path) -> Mapping[str, Any]:
    """Lê ``model/artifact.json`` a partir da raiz da prática."""

    raiz = Path(raiz).expanduser().resolve()
    caminho = raiz / "model" / "artifact.json"
    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            artefato = json.load(arquivo)
    except (OSError, json.JSONDecodeError) as erro:
        raise ErroCaracteristicas(f"Não foi possível ler {caminho}: {erro}.") from erro
    if not isinstance(artefato, Mapping) or artefato.get("schema_version") != 1:
        raise ErroCaracteristicas("model/artifact.json deve usar schema_version=1.")
    return artefato


def _numeros(
    valor: Any,
    nome: str,
    quantidade: int,
) -> tuple[float, ...]:
    if not isinstance(valor, list) or len(valor) != quantidade:
        raise ErroCaracteristicas(
            f"{nome} deve conter {quantidade} valores numéricos."
        )
    try:
        import numpy as np

        numeros = np.asarray(valor, dtype=np.float64)
    except (ImportError, TypeError, ValueError) as erro:
        raise ErroCaracteristicas(f"{nome} deve conter valores numéricos.") from erro
    if not np.isfinite(numeros).all():
        raise ErroCaracteristicas(f"{nome} contém NaN ou infinito.")
    return tuple(float(item) for item in numeros)


def _matriz(valores: Any, largura: int) -> Any:
    try:
        import numpy as np

        matriz = np.asarray(valores, dtype=np.float64)
    except (ImportError, TypeError, ValueError) as erro:
        raise ErroCaracteristicas("As características devem ser numéricas.") from erro
    if matriz.ndim != 2 or matriz.shape[1] != largura:
        raise ErroCaracteristicas(
            f"A matriz deve ter o formato (n, {largura}); recebeu {matriz.shape}."
        )
    if not np.isfinite(matriz).all():
        raise ErroCaracteristicas("A matriz contém NaN ou infinito.")
    return matriz


@dataclass(frozen=True, slots=True)
class Caracteristicas:
    """Ordem das entradas, grupos manipuláveis e scaler fixo."""

    nomes: tuple[str, ...]
    grupos: Mapping[str, tuple[str, ...]]
    escala: tuple[float, ...]
    deslocamento: tuple[float, ...]

    def normalizar(self, valores: Any) -> Any:
        """Aplica ``X * scale_ + min_``, a transformação do MinMaxScaler."""

        import numpy as np

        matriz = _matriz(valores, len(self.nomes))
        resultado = matriz * np.asarray(self.escala, dtype=np.float64)
        resultado += np.asarray(self.deslocamento, dtype=np.float64)
        resultado.setflags(write=False)
        return resultado

    def desnormalizar(self, valores: Any) -> Any:
        """Converte vetores normalizados para as unidades das características."""

        import numpy as np

        matriz = _matriz(valores, len(self.nomes))
        resultado = (
            matriz - np.asarray(self.deslocamento, dtype=np.float64)
        ) / np.asarray(self.escala, dtype=np.float64)
        resultado.setflags(write=False)
        return resultado

    def mascara(self, grupo: str) -> Any:
        """Cria a máscara binária de ``volume``, ``tempo`` ou ``todas``."""

        import numpy as np

        nome = grupo.strip().casefold() if isinstance(grupo, str) else ""
        if nome == "todas":
            habilitadas = set(self.nomes)
        elif nome == "tempo":
            habilitadas = set(self.grupos["time"])
        elif nome == "volume":
            habilitadas = set(self.grupos["volume"])
        else:
            raise ErroCaracteristicas(
                "O grupo deve ser 'volume', 'tempo' ou 'todas'."
            )
        mascara = np.asarray(
            [1.0 if nome in habilitadas else 0.0 for nome in self.nomes],
            dtype=np.float32,
        )
        mascara.setflags(write=False)
        return mascara


def carregar_caracteristicas(raiz: str | Path) -> Caracteristicas:
    """Carrega a ordem das características, seus grupos e o scaler fixo."""

    artefato = ler_artefato(raiz)
    bruto = artefato.get("features")
    if not isinstance(bruto, Mapping):
        raise ErroCaracteristicas("O artefato não define features.")

    nomes_brutos = bruto.get("names")
    if not isinstance(nomes_brutos, list) or any(
        not isinstance(nome, str) or not nome for nome in nomes_brutos
    ):
        raise ErroCaracteristicas("features.names deve conter nomes não vazios.")
    nomes = tuple(nomes_brutos)
    if len(nomes) != 49 or len(set(nomes)) != 49:
        raise ErroCaracteristicas("O detector exige 49 características distintas.")

    grupos_brutos = bruto.get("groups")
    if not isinstance(grupos_brutos, Mapping) or set(grupos_brutos) != {
        "time",
        "volume",
    }:
        raise ErroCaracteristicas("Os grupos devem ser time e volume.")
    grupos: dict[str, tuple[str, ...]] = {}
    for grupo in ("time", "volume"):
        valores = grupos_brutos[grupo]
        if not isinstance(valores, list) or any(nome not in nomes for nome in valores):
            raise ErroCaracteristicas(f"O grupo {grupo} contém nomes inválidos.")
        grupos[grupo] = tuple(valores)
    if set(grupos["time"]) & set(grupos["volume"]):
        raise ErroCaracteristicas("Os grupos time e volume devem ser disjuntos.")
    if set(grupos["time"]) | set(grupos["volume"]) != set(nomes):
        raise ErroCaracteristicas("Os grupos devem cobrir as 49 características.")

    scaler = artefato.get("scaler")
    if not isinstance(scaler, Mapping):
        raise ErroCaracteristicas("O artefato não define o scaler.")
    if scaler.get("feature_range") != [0.0, 1.0] or scaler.get("clip") is not False:
        raise ErroCaracteristicas("O scaler deve usar o intervalo [0, 1] sem clipping.")
    escala = _numeros(scaler.get("scale_"), "scaler.scale_", len(nomes))
    deslocamento = _numeros(scaler.get("min_"), "scaler.min_", len(nomes))
    if any(valor <= 0.0 for valor in escala):
        raise ErroCaracteristicas("scaler.scale_ deve conter fatores positivos.")

    return Caracteristicas(
        nomes=nomes,
        grupos=grupos,
        escala=escala,
        deslocamento=deslocamento,
    )
