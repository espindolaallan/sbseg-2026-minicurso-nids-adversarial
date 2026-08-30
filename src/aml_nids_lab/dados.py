"""Entrada do pipeline: recortes de fluxos já extraídos.

Os CSVs contêm as características produzidas antes da prática. O carregamento
confere o arquivo indicado no manifesto e entrega arrays na ordem esperada pelo
detector.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .caracteristicas import Caracteristicas


class ErroDados(ValueError):
    """Indica um recorte ausente ou incompatível com o pipeline."""


def _somente_leitura(array: Any) -> Any:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ConjuntoFluxos:
    """IDs, características e rótulos alinhados de um recorte."""

    nome: str
    caminho: Path
    nomes_caracteristicas: tuple[str, ...]
    ids: tuple[str, ...]
    x: Any
    y: Any
    pacotes_retorno: Any

    def __len__(self) -> int:
        return len(self.ids)

    def selecionar(self, indices: Sequence[int]) -> "ConjuntoFluxos":
        """Seleciona linhas pela posição e preserva seu alinhamento."""

        import numpy as np

        posicoes = np.asarray(indices)
        if posicoes.ndim != 1 or posicoes.dtype.kind not in "iu":
            raise ErroDados("Os índices devem formar um vetor de inteiros.")
        if ((posicoes < 0) | (posicoes >= len(self))).any():
            raise ErroDados("Há um índice fora do conjunto.")
        lista = [int(indice) for indice in posicoes]
        return ConjuntoFluxos(
            nome=self.nome,
            caminho=self.caminho,
            nomes_caracteristicas=self.nomes_caracteristicas,
            ids=tuple(self.ids[indice] for indice in lista),
            x=_somente_leitura(self.x[posicoes].copy()),
            y=_somente_leitura(self.y[posicoes].copy()),
            pacotes_retorno=_somente_leitura(self.pacotes_retorno[posicoes].copy()),
        )

    def indices_dos_ids(self, ids: Sequence[str]) -> Any:
        """Converte IDs locais em posições do recorte."""

        import numpy as np

        posicao = {identificador: indice for indice, identificador in enumerate(self.ids)}
        ausentes = [identificador for identificador in ids if identificador not in posicao]
        if ausentes:
            raise ErroDados(f"IDs ausentes no conjunto: {ausentes[:5]}.")
        return np.asarray([posicao[identificador] for identificador in ids], dtype=np.int64)


def _carregar_manifesto(raiz: Path) -> Mapping[str, Any]:
    caminho = raiz / "data" / "manifest.json"
    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            manifesto = json.load(arquivo)
    except (OSError, json.JSONDecodeError) as erro:
        raise ErroDados(f"Não foi possível ler {caminho}: {erro}.") from erro
    if not isinstance(manifesto, Mapping) or manifesto.get("schema_version") != 2:
        raise ErroDados("data/manifest.json deve usar schema_version=2.")
    if not isinstance(manifesto.get("datasets"), Mapping):
        raise ErroDados("O manifesto não define os recortes.")
    return manifesto


def carregar_dados(
    raiz: str | Path,
    nome: str,
    *,
    caracteristicas: Caracteristicas,
) -> ConjuntoFluxos:
    """Carrega um recorte e confere arquivo, colunas, rótulos e alinhamento."""

    import numpy as np
    import pandas as pd

    raiz = Path(raiz).expanduser().resolve()
    manifesto = _carregar_manifesto(raiz)
    conjuntos = manifesto["datasets"]
    if nome not in conjuntos:
        raise ErroDados(f"Conjunto desconhecido: {nome!r}.")
    entrada = conjuntos[nome]
    if not isinstance(entrada, Mapping) or not isinstance(entrada.get("output"), Mapping):
        raise ErroDados(f"Metadados incompletos para {nome}.")

    arquivo = entrada.get("file")
    if not isinstance(arquivo, str) or not arquivo:
        raise ErroDados(f"Nome de arquivo inválido para {nome}.")
    pasta_dados = (raiz / "data").resolve()
    caminho = (pasta_dados / arquivo).resolve()
    try:
        caminho.relative_to(pasta_dados)
    except ValueError as erro:
        raise ErroDados(f"Recorte fora de data/: {caminho}.") from erro
    saida = entrada["output"]
    try:
        resumo = hashlib.sha256(caminho.read_bytes()).hexdigest()
        tamanho = caminho.stat().st_size
    except OSError as erro:
        raise ErroDados(f"Não foi possível ler {caminho}: {erro}.") from erro
    if tamanho != saida.get("size_bytes"):
        raise ErroDados(f"{arquivo} não tem o tamanho indicado no manifesto.")
    if resumo != saida.get("sha256"):
        raise ErroDados(f"{arquivo} não corresponde ao manifesto.")

    definicao = caracteristicas
    esquema = manifesto.get("schema")
    if not isinstance(esquema, Mapping) or esquema.get("feature_count") != len(
        definicao.nomes
    ):
        raise ErroDados("O esquema dos dados não corresponde ao detector.")
    id_coluna = esquema.get("id_column")
    auxiliares = esquema.get("auxiliary_columns")
    if not isinstance(id_coluna, str) or auxiliares != [
        "attackCategory",
        "label",
        "packetTotalCountBwd",
    ]:
        raise ErroDados("O esquema das colunas auxiliares é incompatível.")
    colunas = (id_coluna, *definicao.nomes, *auxiliares)

    try:
        tabela = pd.read_csv(
            caminho,
            compression="gzip",
            dtype={id_coluna: "string", "attackCategory": "string"},
            low_memory=False,
        )
    except Exception as erro:
        raise ErroDados(f"Não foi possível carregar {arquivo}: {erro}.") from erro
    if tuple(map(str, tabela.columns)) != colunas or tabela.empty:
        raise ErroDados(f"{arquivo} não preserva o esquema esperado.")

    ids = tuple(str(valor) for valor in tabela[id_coluna].tolist())
    if tabela[id_coluna].isna().any() or any(
        not valor or valor != valor.strip() for valor in ids
    ):
        raise ErroDados(f"{arquivo} contém IDs vazios.")
    if len(set(ids)) != len(ids):
        raise ErroDados(f"{arquivo} contém IDs duplicados.")
    try:
        x = tabela.loc[:, list(definicao.nomes)].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float64, copy=True)
        y_bruto = pd.to_numeric(tabela["label"], errors="raise").to_numpy(
            dtype=np.float64, copy=True
        )
        pacotes = pd.to_numeric(
            tabela["packetTotalCountBwd"], errors="raise"
        ).to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as erro:
        raise ErroDados(f"{arquivo} contém valores não numéricos.") from erro
    if not np.isfinite(x).all() or not np.isfinite(y_bruto).all() or not np.isfinite(pacotes).all():
        raise ErroDados(f"{arquivo} contém NaN ou infinito.")
    if not np.equal(y_bruto, np.floor(y_bruto)).all() or not np.isin(y_bruto, [0, 1]).all():
        raise ErroDados(f"{arquivo} contém rótulos fora de {{0, 1}}.")
    if (pacotes < 0).any() or not np.equal(pacotes, np.floor(pacotes)).all():
        raise ErroDados(f"{arquivo} contém contagens de pacotes inválidas.")

    try:
        posicoes_retorno = [
            definicao.nomes.index(nome)
            for nome in ("ipTotalLengthBwd", "octetTotalCountBwd")
        ]
    except ValueError as erro:
        raise ErroDados("Faltam características de atividade reversa.") from erro
    atividade_reversa = np.column_stack((pacotes, x[:, posicoes_retorno])) > 0
    if np.any(atividade_reversa != atividade_reversa[:, [0]]):
        raise ErroDados(
            f"{arquivo}: packetTotalCountBwd, ipTotalLengthBwd e "
            "octetTotalCountBwd devem indicar atividade conjuntamente."
        )
    y = y_bruto.astype(np.int64)

    categorias_esquema = esquema.get("categories")
    rotulos_esquema = esquema.get("labels")
    if not isinstance(categorias_esquema, Mapping) or not isinstance(
        rotulos_esquema, Mapping
    ):
        raise ErroDados("O manifesto não define categorias e rótulos.")
    normal, dos = categorias_esquema.get("normal"), categorias_esquema.get("dos")
    if (
        not isinstance(normal, str)
        or not isinstance(dos, str)
        or rotulos_esquema.get("normal") != 0
        or rotulos_esquema.get("attack") != 1
    ):
        raise ErroDados("As categorias e os rótulos do manifesto são incompatíveis.")
    categorias = tuple(str(valor) for valor in tabela["attackCategory"].tolist())
    if tabela["attackCategory"].isna().any() or set(categorias) - {normal, dos}:
        raise ErroDados(f"{arquivo} contém categorias inválidas.")
    y_categorias = np.asarray([1 if valor == dos else 0 for valor in categorias])
    if not np.array_equal(y, y_categorias):
        raise ErroDados(f"{arquivo}: attackCategory e label não concordam.")
    if nome.startswith("payload_") and not np.all(y == 1):
        raise ErroDados(f"{arquivo} deve conter somente fluxos DoS.")

    return ConjuntoFluxos(
        nome=nome,
        caminho=caminho,
        nomes_caracteristicas=definicao.nomes,
        ids=ids,
        x=_somente_leitura(x),
        y=_somente_leitura(y),
        pacotes_retorno=_somente_leitura(pacotes),
    )
