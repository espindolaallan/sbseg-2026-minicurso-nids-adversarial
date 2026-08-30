"""Métricas e verificações usadas para avaliar o NIDS e os ataques.

ASR é calculada somente para vetores pareados no espaço de características.
Os conjuntos reextraídos após manipulação de payload são avaliados por Recall
e FNR. As oito relações de coerência tabular são aplicadas nas unidades
originais das características.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


IdFluxo = str | int
TOLERANCIA_ABSOLUTA = 1e-3
TOLERANCIA_RELATIVA = 1e-5

NOMES_VERIFICACOES = (
    "total_length_equals_octet_count",
    "forward_length_equals_octet_count",
    "backward_length_equals_octet_count",
    "length_variance_equals_squared_stdev",
    "overall_minimum_mean_maximum_order",
    "forward_minimum_mean_maximum_order",
    "backward_minimum_mean_maximum_order",
    "inactive_backward_features_are_zero",
)

CARACTERISTICAS_REVERSAS = (
    "bwdBytesAvg",
    "bwdBytesPerMicroseconds",
    "bwdJitterMilliseconds",
    "interPacketTimeSecondsMaxBwd",
    "interPacketTimeSecondsMeanBwd",
    "interPacketTimeSecondsStdevBwd",
    "interPacketTimeSecondsSumBwd",
    "ipTotalLengthBwd",
    "ipTotalLengthMeanBwd",
    "ipTotalLengthStdevBwd",
    "maximumIpTotalLengthBwd",
    "minimumIpTotalLengthBwd",
    "octetTotalCountBwd",
)

CARACTERISTICAS_OBRIGATORIAS = frozenset(
    {
        "ipTotalLength",
        "octetTotalCount",
        "ipTotalLengthFwd",
        "octetTotalCountFwd",
        "ipTotalLengthVar",
        "ipTotalLengthStdev",
        "minimumIpTotalLength",
        "ipTotalLengthMean",
        "maximumIpTotalLength",
        "minimumIpTotalLengthFwd",
        "ipTotalLengthMeanFwd",
        "maximumIpTotalLengthFwd",
        *CARACTERISTICAS_REVERSAS,
    }
)


class ErroAvaliacao(ValueError):
    """Indica entradas incompatíveis com uma métrica."""


class ErroPareamento(ErroAvaliacao):
    """Indica tentativa de calcular ASR sem pareamento um-para-um."""


class ErroCoerencia(ValueError):
    """Indica registros incompletos ou numericamente inválidos."""


@dataclass(frozen=True, slots=True)
class MetricasClassificacao:
    populacao: int
    tp: int
    tn: int
    fp: int
    fn: int
    acuracia: float | None
    precisao: float | None
    recall: float | None
    fnr: float | None
    fpr: float | None
    f1: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "population": self.populacao,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": self.acuracia,
            "precision": self.precisao,
            "recall": self.recall,
            "fnr": self.fnr,
            "fpr": self.fpr,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class MetricasPayload:
    """Recall e FNR de um conjunto reextraído após manipulação de payload."""

    conjunto: str
    nivel: str
    total_dos: int
    tp: int
    fn: int
    recall: float | None
    fnr: float | None

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "dataset_id": self.conjunto,
            "nominal_factor": self.nivel,
            "total_dos": self.total_dos,
            "tp": self.tp,
            "fn": self.fn,
            "recall": self.recall,
            "fnr": self.fnr,
        }


@dataclass(frozen=True, slots=True)
class ResultadoASR:
    """ASR condicional sobre a população elegível do espaço de características."""

    elegiveis: int
    saidas: int
    evasoes: int
    asr: float | None
    evasoes_coerentes: int | None
    taxa_evasao_coerente: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "eligible": self.elegiveis,
            "outputs": self.saidas,
            "evasions": self.evasoes,
            "asr": self.asr,
            "coherent_evasions": self.evasoes_coerentes,
            "coherent_evasion_rate": self.taxa_evasao_coerente,
        }


@dataclass(frozen=True, slots=True)
class CustoPerturbacao:
    """Quantidade de características alteradas (L0) e maior alteração (L-infinito)."""

    saidas: int
    l0_minimo: int | None
    l0_mediana: float | None
    l0_maximo: int | None
    linf_minimo: float | None
    linf_mediana: float | None
    linf_maximo: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "outputs": self.saidas,
            "changed_features_min": self.l0_minimo,
            "changed_features_median": self.l0_mediana,
            "changed_features_max": self.l0_maximo,
            "linf_min": self.linf_minimo,
            "linf_median": self.linf_mediana,
            "linf_max": self.linf_maximo,
        }


@dataclass(frozen=True, slots=True)
class ResultadoRegra:
    nome: str
    status: str


@dataclass(frozen=True, slots=True)
class RelatorioCoerencia:
    ids_coerentes: tuple[IdFluxo, ...]
    resumo_regras: Mapping[str, Mapping[str, int]]
    total_registros: int
    registros_coerentes: int
    registros_incoerentes: int


def _divisao(numerador: int | float, denominador: int | float) -> float | None:
    return float(numerador / denominador) if denominador else None


def _rotulos_binarios(
    valores: Iterable[int], nome: str, permitidos: set[int]
) -> tuple[int, ...]:
    resultado = tuple(valores)
    invalidos = []
    for valor in resultado:
        if isinstance(valor, bool) or not isinstance(valor, int) or valor not in permitidos:
            if repr(valor) not in invalidos:
                invalidos.append(repr(valor))
    if invalidos:
        raise ErroAvaliacao(f"{nome} contém rótulos inesperados: {invalidos}.")
    return resultado


def metricas_classificacao(
    rotulos: Iterable[int],
    predicoes: Iterable[int],
    *,
    rotulo_ataque: int = 1,
    rotulo_normal: int = 0,
) -> MetricasClassificacao:
    """Calcula a matriz de confusão e as métricas binárias do NIDS."""

    if rotulo_ataque == rotulo_normal:
        raise ErroAvaliacao("Os rótulos de ataque e normal devem ser distintos.")
    permitidos = {rotulo_ataque, rotulo_normal}
    verdade = _rotulos_binarios(rotulos, "rotulos", permitidos)
    previstos = _rotulos_binarios(predicoes, "predicoes", permitidos)
    if len(verdade) != len(previstos):
        raise ErroAvaliacao("Rótulos e predições devem ter o mesmo tamanho.")

    tp = sum(r == rotulo_ataque and p == rotulo_ataque for r, p in zip(verdade, previstos))
    tn = sum(r == rotulo_normal and p == rotulo_normal for r, p in zip(verdade, previstos))
    fp = sum(r == rotulo_normal and p == rotulo_ataque for r, p in zip(verdade, previstos))
    fn = sum(r == rotulo_ataque and p == rotulo_normal for r, p in zip(verdade, previstos))
    precisao = _divisao(tp, tp + fp)
    recall = _divisao(tp, tp + fn)
    if precisao is None or recall is None or precisao + recall == 0:
        f1 = None if precisao is None or recall is None else 0.0
    else:
        f1 = 2.0 * precisao * recall / (precisao + recall)
    return MetricasClassificacao(
        populacao=len(verdade),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        acuracia=_divisao(tp + tn, len(verdade)),
        precisao=precisao,
        recall=recall,
        fnr=_divisao(fn, tp + fn),
        fpr=_divisao(fp, fp + tn),
        f1=f1,
    )


def metricas_payload(
    rotulos: Iterable[int],
    predicoes: Iterable[int],
    *,
    conjunto: str,
    nivel: str,
    rotulo_ataque: int = 1,
    rotulo_normal: int = 0,
) -> MetricasPayload:
    """Calcula Recall e FNR para conjuntos reextraídos sem pareamento."""

    if not conjunto or not nivel:
        raise ErroAvaliacao("conjunto e nivel devem ser informados.")
    metricas = metricas_classificacao(
        rotulos,
        predicoes,
        rotulo_ataque=rotulo_ataque,
        rotulo_normal=rotulo_normal,
    )
    return MetricasPayload(
        conjunto=conjunto,
        nivel=nivel,
        total_dos=metricas.tp + metricas.fn,
        tp=metricas.tp,
        fn=metricas.fn,
        recall=metricas.recall,
        fnr=metricas.fnr,
    )


def _ids_unicos(ids: Iterable[IdFluxo], nome: str) -> tuple[IdFluxo, ...]:
    valores = tuple(ids)
    if any(
        isinstance(identificador, bool) or not isinstance(identificador, (str, int))
        for identificador in valores
    ):
        raise ErroAvaliacao(f"{nome} deve conter strings ou inteiros.")
    if len(set(valores)) != len(valores):
        raise ErroAvaliacao(f"{nome} deve conter IDs únicos.")
    return valores


def calcular_asr(
    ids_elegiveis: Iterable[IdFluxo],
    predicoes_atacadas: Mapping[IdFluxo, int],
    *,
    pareado: bool,
    ids_validos: Iterable[IdFluxo] | None = None,
    rotulo_ataque: int = 1,
    rotulo_normal: int = 0,
) -> ResultadoASR:
    """Calcula ASR na amostra elegível pareada.

    Quando ``ids_validos`` é informado, a evasão coerente exige predição normal
    e aprovação nas oito verificações.
    """

    if not pareado:
        raise ErroPareamento("ASR exige correspondência entre fluxo limpo e modificado.")
    if rotulo_ataque == rotulo_normal:
        raise ErroAvaliacao("Os rótulos de ataque e normal devem ser distintos.")
    elegiveis = _ids_unicos(ids_elegiveis, "ids_elegiveis")
    conjunto_elegivel = set(elegiveis)
    inesperados = sorted(set(predicoes_atacadas) - conjunto_elegivel, key=repr)
    if inesperados:
        raise ErroAvaliacao(f"Predições recebidas para IDs não elegíveis: {inesperados}.")
    permitidos = {rotulo_ataque, rotulo_normal}
    invalidas = {
        identificador: valor
        for identificador, valor in predicoes_atacadas.items()
        if isinstance(valor, bool) or not isinstance(valor, int) or valor not in permitidos
    }
    if invalidas:
        raise ErroAvaliacao(f"Predições adversariais inválidas: {invalidas}.")

    evasoes_coerentes: int | None = None
    taxa_coerente: float | None = None
    if ids_validos is not None:
        validos = _ids_unicos(ids_validos, "ids_validos")
        ids_validos_inesperados = sorted(set(validos) - conjunto_elegivel, key=repr)
        if ids_validos_inesperados:
            raise ErroAvaliacao(
                f"Validade recebida para IDs não elegíveis: {ids_validos_inesperados}."
            )
        conjunto_valido = set(validos)
        evasoes_coerentes = sum(
            predicoes_atacadas.get(identificador) == rotulo_normal
            and identificador in conjunto_valido
            for identificador in elegiveis
        )
        taxa_coerente = _divisao(evasoes_coerentes, len(elegiveis))

    saidas = len(predicoes_atacadas)
    evasoes = sum(
        predicoes_atacadas.get(identificador) == rotulo_normal
        for identificador in elegiveis
    )
    denominador = len(elegiveis)
    return ResultadoASR(
        elegiveis=denominador,
        saidas=saidas,
        evasoes=evasoes,
        asr=_divisao(evasoes, denominador),
        evasoes_coerentes=evasoes_coerentes,
        taxa_evasao_coerente=taxa_coerente,
    )


def custo_perturbacao(
    ids: Iterable[IdFluxo],
    vetores_limpos: Iterable[Iterable[int | float]],
    vetores_atacados: Iterable[Iterable[int | float]],
    *,
    tolerancia: float = 1e-6,
) -> CustoPerturbacao:
    """Calcula L0 e L-infinito sobre as saídas efetivamente produzidas."""

    if isinstance(tolerancia, bool) or not isinstance(tolerancia, (int, float)):
        raise ErroAvaliacao("tolerancia deve ser numérica.")
    tolerancia_valor = float(tolerancia)
    if not math.isfinite(tolerancia_valor) or tolerancia_valor < 0.0:
        raise ErroAvaliacao("tolerancia deve ser finita e não negativa.")

    identificadores = _ids_unicos(ids, "ids")
    limpos = tuple(tuple(linha) for linha in vetores_limpos)
    atacados = tuple(tuple(linha) for linha in vetores_atacados)
    if len(identificadores) != len(limpos) or len(limpos) != len(atacados):
        raise ErroAvaliacao("IDs, vetores limpos e atacados devem ter a mesma cardinalidade.")

    contagens: list[int] = []
    linf: list[float] = []
    for identificador, limpo, atacado in zip(identificadores, limpos, atacados):
        if len(limpo) != len(atacado):
            raise ErroAvaliacao(f"Vetores do registro {identificador!r} têm dimensões diferentes.")
        alteradas = 0
        maior_delta = 0.0
        for coluna, (antes, depois) in enumerate(zip(limpo, atacado)):
            if isinstance(antes, bool) or isinstance(depois, bool):
                raise ErroAvaliacao(
                    f"Valor não numérico no registro {identificador!r}, característica {coluna}."
                )
            try:
                antes_valor = float(antes)
                depois_valor = float(depois)
            except (TypeError, ValueError, OverflowError) as erro:
                raise ErroAvaliacao(
                    f"Valor não numérico no registro {identificador!r}, característica {coluna}."
                ) from erro
            if not math.isfinite(antes_valor) or not math.isfinite(depois_valor):
                raise ErroAvaliacao(
                    f"Valor não finito no registro {identificador!r}, característica {coluna}."
                )
            delta = abs(depois_valor - antes_valor)
            alteradas += delta > tolerancia_valor
            maior_delta = max(maior_delta, delta)
        contagens.append(alteradas)
        linf.append(maior_delta)

    return CustoPerturbacao(
        saidas=len(contagens),
        l0_minimo=min(contagens) if contagens else None,
        l0_mediana=float(statistics.median(contagens)) if contagens else None,
        l0_maximo=max(contagens) if contagens else None,
        linf_minimo=min(linf) if linf else None,
        linf_mediana=float(statistics.median(linf)) if linf else None,
        linf_maximo=max(linf) if linf else None,
    )


def _numero(registro: Mapping[str, object], campo: str) -> float:
    if campo not in registro:
        raise ErroCoerencia(f"Característica obrigatória ausente: {campo}.")
    valor = registro[campo]
    if isinstance(valor, bool):
        raise ErroCoerencia(f"{campo} deve ser numérica e finita.")
    try:
        convertido = float(valor)
    except (TypeError, ValueError) as erro:
        raise ErroCoerencia(f"{campo} deve ser numérica e finita.") from erro
    if not math.isfinite(convertido):
        raise ErroCoerencia(f"{campo} deve ser numérica e finita.")
    return convertido


def _tolerancia(
    *operandos: float, absoluta: float, relativa: float
) -> float:
    return absoluta + relativa * max(1.0, *(abs(valor) for valor in operandos))


def _igualdade(
    nome: str,
    esquerdo: float,
    direito: float,
    absoluta: float,
    relativa: float,
) -> ResultadoRegra:
    residuo = abs(esquerdo - direito)
    tolerancia = _tolerancia(esquerdo, direito, absoluta=absoluta, relativa=relativa)
    status = "failed" if residuo > tolerancia else "passed"
    return ResultadoRegra(nome, status)


def _ordenacao(
    nome: str,
    minimo: float,
    media: float,
    maximo: float,
    absoluta: float,
    relativa: float,
) -> ResultadoRegra:
    residuo = max(minimo - media, media - maximo, 0.0)
    tolerancia = _tolerancia(minimo, media, maximo, absoluta=absoluta, relativa=relativa)
    status = "failed" if residuo > tolerancia else "passed"
    return ResultadoRegra(nome, status)


def _nao_aplicavel(nome: str) -> ResultadoRegra:
    return ResultadoRegra(nome, "not_applicable")


def _verificar_registro(
    registro: Mapping[str, object],
    *,
    contagem_reversa: object | None,
    id_registro: IdFluxo | None,
    absoluta: float,
    relativa: float,
) -> tuple[IdFluxo | None, bool, tuple[ResultadoRegra, ...]]:
    ausentes = sorted(CARACTERISTICAS_OBRIGATORIAS - set(registro))
    if ausentes:
        raise ErroCoerencia(f"Características obrigatórias ausentes: {ausentes}.")
    valores = {
        campo: _numero(registro, campo) for campo in CARACTERISTICAS_OBRIGATORIAS
    }
    if contagem_reversa is None:
        quantidade_reversa = _numero(registro, "packetTotalCountBwd")
    else:
        quantidade_reversa = _numero({"contagem_reversa": contagem_reversa}, "contagem_reversa")
    if quantidade_reversa < 0:
        raise ErroCoerencia("A contagem de pacotes reversos não pode ser negativa.")

    regras = [
        _igualdade(
            NOMES_VERIFICACOES[0], valores["ipTotalLength"],
            valores["octetTotalCount"], absoluta, relativa
        ),
        _igualdade(
            NOMES_VERIFICACOES[1], valores["ipTotalLengthFwd"],
            valores["octetTotalCountFwd"], absoluta, relativa
        ),
        _igualdade(
            NOMES_VERIFICACOES[2], valores["ipTotalLengthBwd"],
            valores["octetTotalCountBwd"], absoluta, relativa
        ),
        _igualdade(
            NOMES_VERIFICACOES[3], valores["ipTotalLengthVar"],
            valores["ipTotalLengthStdev"] ** 2, absoluta, relativa
        ),
        _ordenacao(
            NOMES_VERIFICACOES[4], valores["minimumIpTotalLength"],
            valores["ipTotalLengthMean"], valores["maximumIpTotalLength"],
            absoluta, relativa
        ),
        _ordenacao(
            NOMES_VERIFICACOES[5], valores["minimumIpTotalLengthFwd"],
            valores["ipTotalLengthMeanFwd"], valores["maximumIpTotalLengthFwd"],
            absoluta, relativa
        ),
    ]
    if quantidade_reversa > 0:
        regras.append(
            _ordenacao(
                NOMES_VERIFICACOES[6], valores["minimumIpTotalLengthBwd"],
                valores["ipTotalLengthMeanBwd"], valores["maximumIpTotalLengthBwd"],
                absoluta, relativa
            )
        )
        regras.append(_nao_aplicavel(NOMES_VERIFICACOES[7]))
    else:
        regras.append(_nao_aplicavel(NOMES_VERIFICACOES[6]))
        maior_valor_reverso = max(
            abs(valores[campo]) for campo in CARACTERISTICAS_REVERSAS
        )
        regras.append(
            _igualdade(
                NOMES_VERIFICACOES[7],
                maior_valor_reverso,
                0.0,
                absoluta,
                relativa,
            )
        )

    return (
        id_registro,
        not any(regra.status == "failed" for regra in regras),
        tuple(regras),
    )


def verificar_coerencia(
    registros: Iterable[Mapping[str, object]],
    *,
    contagens_reversas: Sequence[object] | None = None,
    ids: Sequence[IdFluxo] | None = None,
    tolerancia_absoluta: float = TOLERANCIA_ABSOLUTA,
    tolerancia_relativa: float = TOLERANCIA_RELATIVA,
) -> RelatorioCoerencia:
    """Aplica as oito verificações e resume resultados por regra."""

    if (
        tolerancia_absoluta < 0
        or tolerancia_relativa < 0
        or not math.isfinite(tolerancia_absoluta)
        or not math.isfinite(tolerancia_relativa)
    ):
        raise ErroCoerencia("As tolerâncias devem ser finitas e não negativas.")
    valores = tuple(registros)
    if contagens_reversas is not None and len(contagens_reversas) != len(valores):
        raise ErroCoerencia("As contagens reversas devem acompanhar os registros.")
    if ids is not None and len(ids) != len(valores):
        raise ErroCoerencia("Os IDs devem acompanhar os registros.")
    if ids is not None and len(set(ids)) != len(ids):
        raise ErroCoerencia("Os IDs dos registros devem ser únicos.")

    avaliados = []
    for indice, registro in enumerate(valores):
        contagem = None if contagens_reversas is None else contagens_reversas[indice]
        identificador = None if ids is None else ids[indice]
        avaliados.append(
            _verificar_registro(
                registro,
                contagem_reversa=contagem,
                id_registro=identificador,
                absoluta=tolerancia_absoluta,
                relativa=tolerancia_relativa,
            )
        )

    resumo: dict[str, dict[str, int]] = {
        nome: {status: 0 for status in ("passed", "failed", "not_applicable")}
        for nome in NOMES_VERIFICACOES
    }
    for _, _, regras in avaliados:
        for regra in regras:
            resumo[regra.nome][regra.status] += 1
    ids_coerentes = tuple(
        identificador
        for identificador, coerente, _ in avaliados
        if coerente and identificador is not None
    )
    coerentes = sum(coerente for _, coerente, _ in avaliados)
    return RelatorioCoerencia(
        ids_coerentes=ids_coerentes,
        resumo_regras=resumo,
        total_registros=len(avaliados),
        registros_coerentes=coerentes,
        registros_incoerentes=len(avaliados) - coerentes,
    )
