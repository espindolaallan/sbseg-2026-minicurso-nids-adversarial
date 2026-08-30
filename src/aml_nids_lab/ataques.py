"""Seleção da amostra e ataques no espaço de características.

O fluxo desta etapa é direto:

1. identificar os fluxos DoS detectados na avaliação sem perturbação;
2. selecionar uma amostra determinística para avaliação;
3. executar FGM minimal e PGD nos mesmos vetores;
4. validar forma, domínio e limites das saídas.

Os ataques recebem vetores já normalizados no intervalo ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


IdFluxo = str | int


class ErroAmostragem(ValueError):
    """Indica dados incompatíveis com a seleção da amostra."""


class ErroAtaque(ValueError):
    """Indica uma entrada incompatível com FGM ou PGD."""


class ErroDependenciaAtaque(RuntimeError):
    """Indica que NumPy ou ART não estão disponíveis."""


@dataclass(frozen=True, slots=True)
class AmostraAtaque:
    """Amostra extraída de uma ordenação determinística dos IDs elegíveis."""

    solicitada: int
    efetiva: int
    populacao_elegivel: int
    semente: int
    ids: tuple[IdFluxo, ...]
    ids_sha256: str

@dataclass(frozen=True, slots=True)
class ResultadoMetodoAtaque:
    """Resultado validado de FGM ou PGD."""

    metodo: str
    status: str
    ids_sha256: str
    amostra_limpa_sha256: str
    mascara_sha256: str
    epsilon: float
    passo: float
    segundos: float
    vetores: Any | None
    maior_linf: float | None
    ids_saida: tuple[IdFluxo, ...] = ()

    @property
    def disponivel(self) -> bool:
        return self.status == "available"


@dataclass(frozen=True, slots=True)
class ResultadoAtaques:
    """FGM minimal e PGD executados sob o mesmo contrato de amostra."""

    metodos: tuple[ResultadoMetodoAtaque, ...]

    def por_metodo(self, metodo: str) -> ResultadoMetodoAtaque:
        procurado = metodo.upper()
        for resultado in self.metodos:
            if resultado.metodo == procurado:
                return resultado
        raise KeyError(f"Método de ataque desconhecido: {metodo!r}.")


class _ErroSaida(ValueError):
    pass


def _id_canonico(identificador: IdFluxo) -> str:
    if isinstance(identificador, bool) or not isinstance(identificador, (str, int)):
        raise ErroAmostragem("Cada ID deve ser uma string ou um inteiro.")
    if isinstance(identificador, str) and not identificador:
        raise ErroAmostragem("IDs não podem ser vazios.")
    return json.dumps(identificador, ensure_ascii=False, separators=(",", ":"))


def _ids_validos(ids: Iterable[IdFluxo]) -> tuple[tuple[IdFluxo, ...], tuple[str, ...]]:
    valores = tuple(ids)
    canonicos = tuple(_id_canonico(identificador) for identificador in valores)
    if len(set(canonicos)) != len(canonicos):
        raise ErroAmostragem("Os IDs devem ser únicos.")
    return valores, canonicos


def fluxos_elegiveis(
    ids: Iterable[IdFluxo],
    rotulos: Iterable[int],
    predicoes: Iterable[int],
    *,
    rotulo_ataque: int = 1,
) -> tuple[IdFluxo, ...]:
    """Retorna os fluxos DoS verdadeiros detectados antes do ataque.

    Essa é a população elegível usada no denominador de ASR. Falsos negativos
    da avaliação sem perturbação não podem se tornar sucessos adversariais.
    """

    ids_validos, _ = _ids_validos(ids)
    rotulos_validos = tuple(rotulos)
    predicoes_validas = tuple(predicoes)
    if not (len(ids_validos) == len(rotulos_validos) == len(predicoes_validas)):
        raise ErroAmostragem("IDs, rótulos e predições devem ter o mesmo tamanho.")
    if isinstance(rotulo_ataque, bool) or rotulo_ataque not in (0, 1):
        raise ErroAmostragem("rotulo_ataque deve ser 0 ou 1.")
    for nome, valores in (("rotulos", rotulos_validos), ("predicoes", predicoes_validas)):
        if any(
            isinstance(valor, bool) or not isinstance(valor, int) or valor not in (0, 1)
            for valor in valores
        ):
            raise ErroAmostragem(f"{nome} deve conter inteiros 0 ou 1.")
    return tuple(
        identificador
        for identificador, rotulo, predicao in zip(
            ids_validos, rotulos_validos, predicoes_validas
        )
        if rotulo == rotulo_ataque and predicao == rotulo_ataque
    )


def selecionar_amostra(
    ids_elegiveis: Iterable[IdFluxo],
    quantidade: int,
    *,
    semente: int,
) -> AmostraAtaque:
    """Seleciona IDs por uma ordenação SHA-256 estável.

    Com a mesma semente, uma amostra menor é prefixo de uma maior.
    """

    if isinstance(quantidade, bool) or not isinstance(quantidade, int) or quantidade < 0:
        raise ErroAmostragem("quantidade deve ser um inteiro não negativo.")
    if isinstance(semente, bool) or not isinstance(semente, int) or semente < 0:
        raise ErroAmostragem("semente deve ser um inteiro não negativo.")
    valores, canonicos = _ids_validos(ids_elegiveis)
    ordenaveis = []
    for identificador, id_codificado in zip(valores, canonicos):
        material = (
            f"aml-nids-lab-sample-v1\0{semente}\0{id_codificado}"
        ).encode("utf-8")
        ordenaveis.append(
            (hashlib.sha256(material).digest(), id_codificado, identificador)
        )
    ordenados = tuple(item[2] for item in sorted(ordenaveis))
    selecionados = ordenados[:quantidade]
    return AmostraAtaque(
        solicitada=quantidade,
        efetiva=len(selecionados),
        populacao_elegivel=len(ordenados),
        semente=semente,
        ids=selecionados,
        ids_sha256=_sha256_canonico(list(selecionados)),
    )


def _valor_python(valor: Any) -> Any:
    conversor = getattr(valor, "tolist", None)
    return conversor() if callable(conversor) else valor


def _numero_finito(valor: Any, nome: str) -> float:
    if isinstance(valor, (bool, str, bytes)):
        raise ErroAtaque(f"{nome} deve ser numérico e finito.")
    try:
        convertido = float(valor)
    except (TypeError, ValueError, OverflowError) as erro:
        raise ErroAtaque(f"{nome} deve ser numérico e finito.") from erro
    if not math.isfinite(convertido):
        raise ErroAtaque(f"{nome} deve ser numérico e finito.")
    return convertido


def _matriz(valor: Any, nome: str) -> tuple[tuple[float, ...], ...]:
    materializado = _valor_python(valor)
    if isinstance(materializado, (str, bytes)):
        raise ErroAtaque(f"{nome} deve ser uma matriz bidimensional.")
    try:
        linhas_brutas = tuple(materializado)
    except TypeError as erro:
        raise ErroAtaque(f"{nome} deve ser uma matriz bidimensional.") from erro
    if not linhas_brutas:
        raise ErroAtaque(f"{nome} não pode ter zero linhas.")

    linhas: list[tuple[float, ...]] = []
    largura: int | None = None
    for indice_linha, linha in enumerate(linhas_brutas):
        linha_materializada = _valor_python(linha)
        if isinstance(linha_materializada, (str, bytes)):
            raise ErroAtaque(f"{nome}[{indice_linha}] deve ser uma sequência numérica.")
        try:
            valores = tuple(linha_materializada)
        except TypeError as erro:
            raise ErroAtaque(
                f"{nome}[{indice_linha}] deve ser uma sequência numérica."
            ) from erro
        if largura is None:
            largura = len(valores)
            if largura == 0:
                raise ErroAtaque(f"{nome} não pode ter zero colunas.")
        elif len(valores) != largura:
            raise ErroAtaque(f"{nome} deve ser retangular.")
        linhas.append(
            tuple(
                _numero_finito(item, f"{nome}[{indice_linha}][{indice_coluna}]")
                for indice_coluna, item in enumerate(valores)
            )
        )
    return tuple(linhas)


def _ids_da_amostra(
    ids: Iterable[IdFluxo], quantidade_linhas: int
) -> tuple[IdFluxo, ...]:
    try:
        valores, _ = _ids_validos(ids)
    except ErroAmostragem as erro:
        raise ErroAtaque(str(erro)) from erro
    if len(valores) != quantidade_linhas:
        raise ErroAtaque("ids e vetores devem descrever a mesma amostra.")
    return valores


def _mascara_binaria(mascara: Any, largura: int) -> tuple[float, ...]:
    materializada = _valor_python(mascara)
    if isinstance(materializada, (str, bytes)):
        raise ErroAtaque("mascara deve ser um vetor binário.")
    try:
        valores = tuple(materializada)
    except TypeError as erro:
        raise ErroAtaque("mascara deve ser um vetor binário.") from erro
    if len(valores) != largura:
        raise ErroAtaque(f"mascara deve ter {largura} posições.")
    normalizada = []
    for indice, valor in enumerate(valores):
        numero = _numero_finito(valor, f"mascara[{indice}]")
        if numero not in (0.0, 1.0):
            raise ErroAtaque("mascara deve conter somente zero ou um.")
        normalizada.append(numero)
    if not any(normalizada):
        raise ErroAtaque("mascara deve habilitar ao menos uma característica.")
    return tuple(normalizada)


def _sha256_canonico(valor: Any) -> str:
    conteudo = json.dumps(
        valor,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(conteudo).hexdigest()


def _carregar_dependencias() -> tuple[Any, type[Any], type[Any]]:
    try:
        import numpy as np
        from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    except (ImportError, ModuleNotFoundError) as erro:
        dependencia = getattr(erro, "name", None) or type(erro).__name__
        raise ErroDependenciaAtaque(
            "Não foi possível carregar NumPy/ART para executar FGM minimal e PGD "
            f"(dependência: {dependencia})."
        ) from erro
    return np, FastGradientMethod, ProjectedGradientDescent


def _tempo(inicio: float, relogio: Callable[[], float]) -> float:
    return max(0.0, float(relogio()) - inicio)


def _validar_saidas(
    limpos: tuple[tuple[float, ...], ...],
    adversariais: Any,
    mascara: tuple[float, ...],
    epsilon: float,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    float | None,
]:
    """Valida forma, finitude, clipping, máscara e L-infinito linha a linha."""

    materializados = _valor_python(adversariais)
    if isinstance(materializados, (str, bytes)):
        raise _ErroSaida("O ART não devolveu uma matriz bidimensional.")
    try:
        linhas_geradas = tuple(materializados)
    except TypeError as erro:
        raise _ErroSaida("O ART não devolveu uma matriz bidimensional.") from erro
    if len(linhas_geradas) != len(limpos):
        raise _ErroSaida("O ART devolveu uma matriz com cardinalidade diferente.")

    tolerancia = max(1e-6, epsilon * 1e-6)
    largura = len(limpos[0])
    linhas_validas: list[tuple[float, ...]] = []
    posicoes_validas: list[int] = []
    maior_delta = 0.0

    for indice, (limpo, linha_bruta) in enumerate(zip(limpos, linhas_geradas)):
        try:
            materializada = _valor_python(linha_bruta)
            if isinstance(materializada, (str, bytes)):
                raise _ErroSaida("a saída não é uma sequência numérica")
            try:
                valores = tuple(materializada)
            except TypeError as erro:
                raise _ErroSaida("a saída não é uma sequência numérica") from erro
            if len(valores) != largura:
                raise _ErroSaida(
                    f"a saída possui {len(valores)} características; eram esperadas {largura}"
                )
            gerada = tuple(
                _numero_finito(valor, f"saidas[{indice}][{coluna}]")
                for coluna, valor in enumerate(valores)
            )
            maior_delta_linha = 0.0
            for coluna, (antes, depois) in enumerate(zip(limpo, gerada)):
                if depois < -tolerancia or depois > 1.0 + tolerancia:
                    raise _ErroSaida(
                        f"valor fora de [0, 1] na característica {coluna}"
                    )
                delta = abs(depois - antes)
                maior_delta_linha = max(maior_delta_linha, delta)
                if mascara[coluna] == 0.0 and delta > tolerancia:
                    raise _ErroSaida(
                        f"característica bloqueada alterada na posição {coluna}"
                    )
                if delta > epsilon + tolerancia:
                    raise _ErroSaida(
                        f"limite L_inf excedido na característica {coluna}"
                    )
        except (ErroAtaque, _ErroSaida):
            continue
        linhas_validas.append(gerada)
        posicoes_validas.append(indice)
        maior_delta = max(maior_delta, maior_delta_linha)

    return (
        tuple(linhas_validas),
        tuple(posicoes_validas),
        maior_delta if linhas_validas else None,
    )


def executar_fgm_pgd(
    classificador_art: Any,
    vetores: Any,
    ids: Iterable[IdFluxo],
    *,
    epsilon: float,
    mascara: Any,
    iteracoes_pgd: int = 30,
    relogio: Callable[[], float] = time.perf_counter,
) -> ResultadoAtaques:
    """Executa FGM minimal e PGD sobre os mesmos vetores e IDs.

    FGM usa ``minimal=True``, passo ``epsilon / 30`` e nenhuma inicialização
    aleatória. PGD usa ``iteracoes_pgd`` passos, passo
    ``epsilon / iteracoes_pgd`` e nenhuma inicialização aleatória. A
    indisponibilidade de um método não impede a execução do outro.
    """

    if classificador_art is None:
        raise ErroAtaque("classificador_art não pode ser None.")
    epsilon_valor = _numero_finito(epsilon, "epsilon")
    if epsilon_valor <= 0.0:
        raise ErroAtaque("epsilon deve ser positivo.")
    if (
        isinstance(iteracoes_pgd, bool)
        or not isinstance(iteracoes_pgd, int)
        or iteracoes_pgd <= 0
    ):
        raise ErroAtaque("iteracoes_pgd deve ser um inteiro positivo.")

    forma = getattr(vetores, "shape", None)
    entrada_vazia = (
        isinstance(forma, tuple)
        and len(forma) == 2
        and forma[0] == 0
        and forma[1] > 0
    )
    if entrada_vazia:
        limpos: tuple[tuple[float, ...], ...] = ()
        identificadores = _ids_da_amostra(ids, 0)
        mascara_valores = _mascara_binaria(mascara, forma[1])
    else:
        limpos = _matriz(vetores, "vetores")
        identificadores = _ids_da_amostra(ids, len(limpos))
        mascara_valores = _mascara_binaria(mascara, len(limpos[0]))
        for indice_linha, linha in enumerate(limpos):
            for indice_coluna, valor in enumerate(linha):
                if valor < -1e-6 or valor > 1.0 + 1e-6:
                    raise ErroAtaque(
                        "vetores deve estar em [0, 1] "
                        f"(linha {indice_linha}, coluna {indice_coluna})."
                    )

    np, tipo_fgm, tipo_pgd = _carregar_dependencias()
    ids_sha256 = _sha256_canonico(list(identificadores))
    limpos_sha256 = _sha256_canonico(
        []
        if entrada_vazia
        else {"sample_ids": list(identificadores), "vectors": limpos}
    )
    mascara_sha256 = _sha256_canonico(mascara_valores)
    passo_fgm = epsilon_valor / 30.0
    passo_pgd = epsilon_valor / float(iteracoes_pgd)
    parametros_comuns = {
        "estimator": classificador_art,
        "norm": np.inf,
        "eps": epsilon_valor,
        "targeted": False,
        "num_random_init": 0,
    }
    especificacoes = (
        (
            "FGM",
            tipo_fgm,
            passo_fgm,
            {
                **parametros_comuns,
                "eps_step": passo_fgm,
                "minimal": True,
            },
        ),
        (
            "PGD",
            tipo_pgd,
            passo_pgd,
            {
                **parametros_comuns,
                "eps_step": passo_pgd,
                "max_iter": iteracoes_pgd,
                "verbose": False,
            },
        ),
    )

    def construir_resultado(
        *,
        metodo: str,
        status: str,
        passo: float,
        segundos: float,
        vetores: Any | None = None,
        maior_linf: float | None = None,
        ids_saida: tuple[IdFluxo, ...] = (),
    ) -> ResultadoMetodoAtaque:
        return ResultadoMetodoAtaque(
            metodo=metodo,
            status=status,
            ids_sha256=ids_sha256,
            amostra_limpa_sha256=limpos_sha256,
            mascara_sha256=mascara_sha256,
            epsilon=epsilon_valor,
            passo=passo,
            segundos=segundos,
            vetores=vetores,
            maior_linf=maior_linf,
            ids_saida=ids_saida,
        )

    if entrada_vazia:
        return ResultadoAtaques(
            metodos=tuple(
                construir_resultado(
                    metodo=metodo,
                    status="unavailable",
                    passo=passo,
                    segundos=0.0,
                )
                for metodo, _, passo, _ in especificacoes
            )
        )

    resultados = []
    for metodo, tipo_ataque, passo, parametros in especificacoes:
        inicio = float(relogio())
        try:
            ataque = tipo_ataque(**parametros)
        except Exception:
            resultados.append(
                construir_resultado(
                    metodo=metodo,
                    status="unavailable",
                    passo=passo,
                    segundos=_tempo(inicio, relogio),
                )
            )
            continue

        try:
            limpos_array = np.asarray(limpos, dtype=np.float32)
            mascara_array = np.asarray(mascara_valores, dtype=np.float32)
            adversariais = ataque.generate(
                x=limpos_array.copy(), mask=mascara_array.copy()
            )
        except Exception:
            resultados.append(
                construir_resultado(
                    metodo=metodo,
                    status="unavailable",
                    passo=passo,
                    segundos=_tempo(inicio, relogio),
                )
            )
            continue

        try:
            linhas, posicoes, maior_linf = _validar_saidas(
                limpos, adversariais, mascara_valores, epsilon_valor
            )
        except _ErroSaida:
            resultados.append(
                construir_resultado(
                    metodo=metodo,
                    status="unavailable",
                    passo=passo,
                    segundos=_tempo(inicio, relogio),
                )
            )
            continue

        ids_saida = tuple(identificadores[posicao] for posicao in posicoes)
        if not ids_saida:
            resultados.append(
                construir_resultado(
                    metodo=metodo,
                    status="unavailable",
                    passo=passo,
                    segundos=_tempo(inicio, relogio),
                    ids_saida=ids_saida,
                )
            )
            continue

        resultados.append(
            construir_resultado(
                metodo=metodo,
                status="available",
                passo=passo,
                segundos=_tempo(inicio, relogio),
                vetores=np.asarray(linhas, dtype=np.float32),
                maior_linf=maior_linf,
                ids_saida=ids_saida,
            )
        )

    return ResultadoAtaques(metodos=tuple(resultados))
