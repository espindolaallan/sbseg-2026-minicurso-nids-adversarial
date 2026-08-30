"""Conecta as etapas científicas mostradas no notebook.

Dados, normalização e inferência permanecem visíveis no notebook. Este módulo
organiza a amostra comum, os ataques e a avaliação de coerência.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ataques import (
    AmostraAtaque,
    ResultadoAtaques,
    ResultadoMetodoAtaque,
    executar_fgm_pgd,
    fluxos_elegiveis,
    selecionar_amostra,
)
from .avaliacao import (
    CustoPerturbacao,
    MetricasClassificacao,
    MetricasPayload,
    RelatorioCoerencia,
    ResultadoASR,
    calcular_asr,
    custo_perturbacao,
    metricas_classificacao,
    verificar_coerencia,
)
from .caracteristicas import Caracteristicas
from .configuracao import CENARIOS_PAYLOAD, CONJUNTOS, ParametrosLab
from .dados import ConjuntoFluxos
from .detector import adaptar_para_art, inferir


class ErroExperimento(RuntimeError):
    """Indica uma inconsistência entre etapas do experimento."""


@dataclass(frozen=True, slots=True)
class PipelinePreparado:
    parametros: ParametrosLab
    caracteristicas: Caracteristicas
    dados: Mapping[str, ConjuntoFluxos]
    modelo: Any


@dataclass(frozen=True, slots=True)
class ResultadoBaseline:
    """Métricas limpas e verdadeiros positivos DoS elegíveis."""

    metricas: MetricasClassificacao
    ids_elegiveis: tuple[str, ...]
    predicoes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelecaoAtaques:
    avaliacao: AmostraAtaque
    dados_limpos: ConjuntoFluxos
    mascara: Any
    ids_atacados: tuple[str, ...]
    vetores_para_ataque: Any
    contagens_reversas_ataque: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AtaquesExecutados:
    resultado: ResultadoAtaques
    contrato_art: Mapping[str, Any]

    @property
    def fgm(self) -> ResultadoMetodoAtaque:
        return self.resultado.por_metodo("FGM")

    @property
    def pgd(self) -> ResultadoMetodoAtaque:
        return self.resultado.por_metodo("PGD")


@dataclass(frozen=True, slots=True)
class ResultadoAtaqueAvaliado:
    status: str
    execucao: ResultadoMetodoAtaque
    metricas: ResultadoASR
    custo: CustoPerturbacao | None
    validade: RelatorioCoerencia | None


@dataclass(frozen=True, slots=True)
class AvaliacaoAtaques:
    fgm: ResultadoAtaqueAvaliado
    pgd: ResultadoAtaqueAvaliado
    validade_amostra_limpa: RelatorioCoerencia
    contrato_art: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ResultadoPayloadAvaliado:
    metricas: MetricasPayload
    validade: RelatorioCoerencia


def preparar_pipeline(
    parametros: ParametrosLab,
    caracteristicas: Caracteristicas,
    dados: Mapping[str, ConjuntoFluxos],
    modelo: Any,
) -> PipelinePreparado:
    """Reúne os objetos carregados nas etapas anteriores do notebook."""

    if not isinstance(parametros, ParametrosLab):
        raise ErroExperimento("parametros deve ser um ParametrosLab.")
    if not isinstance(caracteristicas, Caracteristicas) or modelo is None:
        raise ErroExperimento("Características e detector devem estar carregados.")
    if set(dados) != set(CONJUNTOS):
        raise ErroExperimento("Os quatro recortes devem estar carregados.")
    for nome, conjunto in dados.items():
        if (
            not isinstance(conjunto, ConjuntoFluxos)
            or conjunto.nome != nome
            or conjunto.nomes_caracteristicas != caracteristicas.nomes
        ):
            raise ErroExperimento(f"Recorte incompatível: {nome}.")
    return PipelinePreparado(parametros, caracteristicas, dict(dados), modelo)


def _validar_baseline(
    conjunto: ConjuntoFluxos,
    baseline: ResultadoBaseline,
) -> None:
    """Confere o vínculo entre a avaliação limpa e seus IDs elegíveis."""

    if not isinstance(baseline, ResultadoBaseline):
        raise ErroExperimento("baseline deve ser um ResultadoBaseline.")
    predicoes = tuple(baseline.predicoes)
    ids = tuple(map(str, baseline.ids_elegiveis))
    if len(predicoes) != len(conjunto) or any(
        type(valor) is not int or valor not in (0, 1) for valor in predicoes
    ):
        raise ErroExperimento("As predições do baseline devem ser binárias.")
    rotulos = tuple(map(int, conjunto.y))
    metricas_esperadas = metricas_classificacao(rotulos, predicoes)
    ids_esperados = tuple(
        map(str, fluxos_elegiveis(conjunto.ids, rotulos, predicoes))
    )
    if baseline.metricas != metricas_esperadas or ids != ids_esperados:
        raise ErroExperimento("O baseline não corresponde ao recorte limpo.")


def selecionar_ataques(
    pipeline: PipelinePreparado,
    baseline: ResultadoBaseline,
    vetores_limpos: Any,
) -> SelecaoAtaques:
    """Prepara a amostra comum a FGM minimal e PGD."""

    import numpy as np

    parametros = pipeline.parametros
    conjunto = pipeline.dados["clean_day17"]
    matriz = np.asarray(vetores_limpos)
    forma = (len(conjunto), len(pipeline.caracteristicas.nomes))
    if matriz.shape != forma or not np.isfinite(matriz).all():
        raise ErroExperimento(f"Os vetores limpos devem ter formato {forma}.")
    esperados = pipeline.caracteristicas.normalizar(conjunto.x)
    if not np.allclose(matriz, esperados, rtol=1e-6, atol=1e-8):
        raise ErroExperimento(
            "Os vetores limpos devem preservar a ordem do recorte clean_day17."
        )
    _validar_baseline(conjunto, baseline)
    avaliacao = selecionar_amostra(
        baseline.ids_elegiveis,
        parametros.n_amostras,
        semente=parametros.semente,
    )
    if not avaliacao.ids:
        raise ErroExperimento("A amostra de avaliação ficou vazia.")

    indices = conjunto.indices_dos_ids(tuple(map(str, avaliacao.ids)))
    dados_limpos = conjunto.selecionar(indices)
    selecionados = matriz[indices].copy()
    mascara = pipeline.caracteristicas.mascara(
        parametros.grupo_caracteristicas
    )
    vetores_ataque = np.clip(selecionados, 0.0, 1.0)
    vetores_ataque.setflags(write=False)
    ids_atacados = tuple(map(str, dados_limpos.ids))
    contagens = tuple(float(valor) for valor in dados_limpos.pacotes_retorno)
    return SelecaoAtaques(
        avaliacao,
        dados_limpos,
        mascara,
        ids_atacados,
        vetores_ataque,
        contagens,
    )


def executar_ataques(
    pipeline: PipelinePreparado,
    selecao: SelecaoAtaques,
) -> AtaquesExecutados:
    """Executa FGM minimal e PGD sobre os mesmos IDs."""

    import numpy as np
    import torch

    np.random.seed(pipeline.parametros.semente)
    torch.manual_seed(pipeline.parametros.semente)
    adaptador = adaptar_para_art(pipeline.modelo, pipeline.parametros.limiar)
    resultado = executar_fgm_pgd(
        adaptador.estimador,
        selecao.vetores_para_ataque,
        selecao.ids_atacados,
        epsilon=pipeline.parametros.epsilon,
        mascara=selecao.mascara,
        iteracoes_pgd=pipeline.parametros.iteracoes_pgd,
    )
    return AtaquesExecutados(resultado, dict(adaptador.contrato))


def _avaliar_coerencia(
    matriz: Any,
    caracteristicas: Caracteristicas,
    contagens: Sequence[Any],
    ids: Sequence[str],
) -> RelatorioCoerencia:
    registros = []
    for indice, linha in enumerate(matriz):
        valores = tuple(float(valor) for valor in linha)
        if len(valores) != len(caracteristicas.nomes):
            raise ErroExperimento(f"Vetor incompatível na linha {indice}.")
        registros.append(dict(zip(caracteristicas.nomes, valores)))
    return verificar_coerencia(registros, contagens_reversas=contagens, ids=ids)


def _avaliar_metodo(
    pipeline: PipelinePreparado,
    selecao: SelecaoAtaques,
    execucao: ResultadoMetodoAtaque,
) -> ResultadoAtaqueAvaliado:
    """Avalia predições, coerência e magnitude das saídas do ataque."""

    predicoes: dict[str, int] = {}
    validade: RelatorioCoerencia | None = None
    custo: CustoPerturbacao | None = None
    avaliacao_concluida = False

    if execucao.disponivel and execucao.vetores is not None:
        try:
            ids_saida = tuple(map(str, execucao.ids_saida))
            inferencia = inferir(
                pipeline.modelo,
                execucao.vetores,
                pipeline.parametros.limiar,
            )
            rotulos = tuple(int(valor) for valor in inferencia.rotulos)
            if len(rotulos) != len(ids_saida):
                raise ErroExperimento("A inferência adversarial perdeu linhas.")
            predicoes = dict(zip(ids_saida, rotulos))

            contagem_por_id = dict(
                zip(selecao.ids_atacados, selecao.contagens_reversas_ataque)
            )
            validade = _avaliar_coerencia(
                pipeline.caracteristicas.desnormalizar(execucao.vetores),
                pipeline.caracteristicas,
                tuple(contagem_por_id[item] for item in ids_saida),
                ids_saida,
            )
            limpo_por_id = dict(
                zip(selecao.ids_atacados, tuple(selecao.vetores_para_ataque))
            )
            custo = custo_perturbacao(
                ids_saida,
                tuple(limpo_por_id[item] for item in ids_saida),
                execucao.vetores,
                tolerancia=max(1e-6, execucao.epsilon * 1e-6),
            )
            avaliacao_concluida = True
        except Exception:
            predicoes = {}
            validade = None
            custo = None

    ids_validos = () if validade is None else tuple(map(str, validade.ids_coerentes))
    metricas = calcular_asr(
        tuple(map(str, selecao.avaliacao.ids)),
        predicoes,
        pareado=True,
        ids_validos=ids_validos,
    )
    status = (
        "available"
        if avaliacao_concluida and predicoes
        else "unavailable"
    )
    return ResultadoAtaqueAvaliado(
        status=status,
        execucao=execucao,
        metricas=metricas,
        custo=custo,
        validade=validade,
    )


def avaliar_ataques(
    pipeline: PipelinePreparado,
    selecao: SelecaoAtaques,
    execucao: AtaquesExecutados,
) -> AvaliacaoAtaques:
    """Avalia ASR, custo e coerência de FGM minimal e PGD."""

    limpa = selecao.dados_limpos
    validade_limpa = _avaliar_coerencia(
        limpa.x,
        pipeline.caracteristicas,
        limpa.pacotes_retorno,
        limpa.ids,
    )
    return AvaliacaoAtaques(
        _avaliar_metodo(pipeline, selecao, execucao.fgm),
        _avaliar_metodo(pipeline, selecao, execucao.pgd),
        validade_limpa,
        dict(execucao.contrato_art),
    )


def montar_avaliacao_payload(
    pipeline: PipelinePreparado,
    metricas: Mapping[str, MetricasPayload],
) -> Mapping[str, ResultadoPayloadAvaliado]:
    """Acrescenta a verificação de coerência à avaliação do notebook."""

    nomes = {nome for nome, _ in CENARIOS_PAYLOAD}
    if set(metricas) != nomes:
        raise ErroExperimento("A avaliação deve conter os três payloads.")
    resultados: dict[str, ResultadoPayloadAvaliado] = {}
    for nome, nivel in CENARIOS_PAYLOAD:
        conjunto = pipeline.dados[nome]
        resultado = metricas[nome]
        if (
            not isinstance(resultado, MetricasPayload)
            or resultado.conjunto != nome
            or resultado.nivel != nivel
            or resultado.total_dos != len(conjunto)
        ):
            raise ErroExperimento(f"Métricas incompatíveis para {nome}.")
        validade = _avaliar_coerencia(
            conjunto.x,
            pipeline.caracteristicas,
            conjunto.pacotes_retorno,
            conjunto.ids,
        )
        resultados[nome] = ResultadoPayloadAvaliado(resultado, validade)
    return resultados
