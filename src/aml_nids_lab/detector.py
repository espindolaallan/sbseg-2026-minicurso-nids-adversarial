"""MLP, decisão binária e adaptação do detector para o ART."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .caracteristicas import ler_artefato


class ErroDetector(ValueError):
    """Indica um checkpoint, uma entrada ou um limiar incompatível."""


FORMAS_CHECKPOINT: Mapping[str, tuple[int, ...]] = {
    "fc1.weight": (100, 49),
    "fc1.bias": (100,),
    "fc2.weight": (100, 100),
    "fc2.bias": (100,),
    "fc3.weight": (100, 100),
    "fc3.bias": (100,),
    "fc4.weight": (1, 100),
    "fc4.bias": (1,),
}


@dataclass(frozen=True, slots=True)
class ResultadoInferencia:
    """Logits, probabilidades e rótulos alinhados à entrada."""

    logits: Any
    probabilidades: Any
    rotulos: Any

    def __len__(self) -> int:
        return len(self.rotulos)


@dataclass(frozen=True, slots=True)
class DetectorART:
    """Vista de duas classes do detector e o contrato dessa adaptação."""

    estimador: Any
    contrato: Mapping[str, Any]


def _torch() -> Any:
    try:
        import torch
    except ImportError as erro:
        raise ErroDetector("PyTorch não está instalado.") from erro
    versao = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(torch.__version__))
    if versao is None or tuple(map(int, versao.groups())) < (2, 10, 0):
        raise ErroDetector("O carregamento seguro do checkpoint exige PyTorch 2.10 ou superior.")
    return torch


def _criar_mlp(torch: Any) -> Any:
    """Cria a rede 49–100–100–100–1 com saída em logit."""

    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(49, 100)
            self.fc2 = torch.nn.Linear(100, 100)
            self.fc3 = torch.nn.Linear(100, 100)
            self.fc4 = torch.nn.Linear(100, 1)

        def forward(self, valores: Any) -> Any:
            valores = torch.relu(self.fc1(valores))
            valores = torch.relu(self.fc2(valores))
            valores = torch.relu(self.fc3(valores))
            return self.fc4(valores)

    return MLP()


def _validar_state_dict(state_dict: Any, torch: Any) -> None:
    if not isinstance(state_dict, Mapping) or set(state_dict) != set(FORMAS_CHECKPOINT):
        raise ErroDetector("O checkpoint não contém o state_dict esperado.")
    for nome, forma in FORMAS_CHECKPOINT.items():
        valor = state_dict[nome]
        if (
            not torch.is_tensor(valor)
            or tuple(valor.shape) != forma
            or not valor.is_floating_point()
            or not bool(torch.isfinite(valor).all().item())
        ):
            raise ErroDetector(f"Tensor incompatível no checkpoint: {nome}.")


def carregar_detector(raiz: str | Path) -> Any:
    """Confere o SHA-256 e carrega o checkpoint em CPU, no modo weights-only."""

    raiz = Path(raiz).expanduser().resolve()
    artefato = ler_artefato(raiz)
    arquitetura = artefato.get("architecture")
    if not isinstance(arquitetura, Mapping) or (
        arquitetura.get("hidden_units") != [100, 100, 100]
        or arquitetura.get("hidden_activation") != "ReLU"
        or arquitetura.get("output_units") != 1
        or arquitetura.get("output") != "logit"
    ):
        raise ErroDetector("A arquitetura declarada não corresponde à MLP.")
    checkpoint = artefato.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ErroDetector("O artefato não define o checkpoint.")
    nome = checkpoint.get("path")
    if not isinstance(nome, str) or Path(nome).name != nome:
        raise ErroDetector("O caminho do checkpoint deve ser um nome de arquivo.")
    caminho = (raiz / "model" / nome).resolve()
    try:
        resumo = hashlib.sha256(caminho.read_bytes()).hexdigest()
    except OSError as erro:
        raise ErroDetector(f"Não foi possível ler {caminho}: {erro}.") from erro
    if resumo != checkpoint.get("sha256"):
        raise ErroDetector("O checkpoint não corresponde ao artifact.json.")

    torch = _torch()
    try:
        state_dict = torch.load(
            caminho,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    except Exception as erro:
        raise ErroDetector(f"Não foi possível carregar o checkpoint: {erro}.") from erro
    _validar_state_dict(state_dict, torch)
    modelo = _criar_mlp(torch)
    try:
        modelo.load_state_dict(state_dict, strict=True)
    except Exception as erro:
        raise ErroDetector(f"O checkpoint não corresponde à MLP: {erro}.") from erro
    modelo.to(torch.device("cpu"))
    modelo.eval()
    return modelo


def _limiar(valor: Any) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErroDetector("O limiar deve ser numérico.")
    convertido = float(valor)
    if not math.isfinite(convertido) or not 0.0 < convertido < 1.0:
        raise ErroDetector("O limiar deve estar entre zero e um.")
    return convertido


def classificar(probabilidades: Any, limiar: float = 0.3) -> Any:
    """Aplica a regra do detector: ataque quando ``probabilidade > limiar``."""

    import numpy as np

    fronteira = _limiar(limiar)
    try:
        valores = np.asarray(probabilidades, dtype=np.float64)
    except (TypeError, ValueError) as erro:
        raise ErroDetector("As probabilidades devem ser numéricas.") from erro
    if valores.ndim != 1 or not np.isfinite(valores).all():
        raise ErroDetector("As probabilidades devem formar um vetor finito.")
    if ((valores < 0.0) | (valores > 1.0)).any():
        raise ErroDetector("As probabilidades devem estar entre zero e um.")
    resultado = (valores > fronteira).astype(np.int64)
    resultado.setflags(write=False)
    return resultado


def inferir(
    modelo: Any,
    x_normalizado: Any,
    limiar: float = 0.3,
    *,
    tamanho_lote: int = 1024,
) -> ResultadoInferencia:
    """Executa a MLP em CPU e aplica sigmoid e o limiar de decisão."""

    import numpy as np

    if isinstance(tamanho_lote, bool) or not isinstance(tamanho_lote, int) or tamanho_lote <= 0:
        raise ErroDetector("O tamanho do lote deve ser um inteiro positivo.")
    try:
        matriz = np.asarray(x_normalizado, dtype=np.float32)
    except (TypeError, ValueError) as erro:
        raise ErroDetector("A entrada da MLP deve ser numérica.") from erro
    if matriz.ndim != 2 or matriz.shape[1] != 49 or not np.isfinite(matriz).all():
        raise ErroDetector(f"A entrada da MLP deve ter formato (n, 49); recebeu {matriz.shape}.")
    matriz = np.ascontiguousarray(matriz)
    torch = _torch()
    modelo.to(torch.device("cpu"))
    modelo.eval()

    lotes: list[Any] = []
    with torch.no_grad():
        for inicio in range(0, len(matriz), tamanho_lote):
            lote = torch.from_numpy(matriz[inicio : inicio + tamanho_lote])
            saida = modelo(lote)
            if saida.ndim != 2 or tuple(saida.shape) != (len(lote), 1):
                raise ErroDetector("A MLP deve produzir um logit por entrada.")
            lotes.append(saida[:, 0].detach().cpu())
    logits = (
        torch.cat(lotes).numpy().copy()
        if lotes
        else np.empty((0,), dtype=np.float32)
    )
    probabilidades = torch.sigmoid(torch.from_numpy(logits)).numpy().copy()
    rotulos = classificar(probabilidades, limiar)
    logits.setflags(write=False)
    probabilidades.setflags(write=False)
    return ResultadoInferencia(logits, probabilidades, rotulos)


def _fronteira_logit(torch: Any, limiar: float) -> float:
    """Maior logit float32 cuja sigmoid ainda pertence à classe benigna."""

    chave_menos_infinito = 0x007FFFFF
    chave_mais_infinito = 0xFF800000

    def valor(chave: int) -> Any:
        bits = chave ^ 0x80000000 if chave & 0x80000000 else (~chave) & 0xFFFFFFFF
        numero = struct.unpack(">f", struct.pack(">I", bits))[0]
        return torch.tensor(numero, dtype=torch.float32, device="cpu")

    def benigno(logit: Any) -> bool:
        return float(torch.sigmoid(logit).item()) <= limiar

    menor, maior = chave_menos_infinito, chave_mais_infinito
    while maior - menor > 1:
        meio = (menor + maior) // 2
        if benigno(valor(meio)):
            menor = meio
        else:
            maior = meio
    fronteira, sucessor = valor(menor), valor(maior)
    if not benigno(fronteira) or benigno(sucessor):
        raise ErroDetector("Não foi possível representar a fronteira do limiar em float32.")
    return float(fronteira.item())


def adaptar_para_art(modelo: Any, limiar: float = 0.3) -> DetectorART:
    """Expõe ``[fronteira, logit]`` sem alterar a MLP de uma saída."""

    torch = _torch()
    fronteira_decisao = _limiar(limiar)
    fronteira_logit = _fronteira_logit(torch, fronteira_decisao)
    try:
        from art.estimators.classification import PyTorchClassifier
    except ImportError as erro:
        raise ErroDetector("Adversarial Robustness Toolbox não está instalado.") from erro

    class Adaptador(torch.nn.Module):
        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base_model = base

        def forward(self, valores: Any) -> Any:
            logit_ataque = self.base_model(valores)
            if logit_ataque.ndim != 2 or logit_ataque.shape[1] != 1:
                raise ErroDetector("A MLP deve produzir um logit por entrada.")
            logit_benigno = torch.full_like(logit_ataque, fronteira_logit)
            return torch.cat((logit_benigno, logit_ataque), dim=1)

    adaptado = Adaptador(modelo).to(torch.device("cpu"))
    adaptado.eval()
    otimizador = torch.optim.Adam(adaptado.parameters(), lr=0.001)
    try:
        estimador = PyTorchClassifier(
            model=adaptado,
            clip_values=(0.0, 1.0),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=otimizador,
            input_shape=(49,),
            nb_classes=2,
            device_type="cpu",
        )
    except Exception as erro:
        raise ErroDetector(f"Não foi possível construir o estimador ART: {erro}.") from erro
    return DetectorART(
        estimador=estimador,
        contrato={
            "implementation": "art.estimators.classification.PyTorchClassifier",
            "adapter": "two_class_strict_threshold_logits",
            "adapter_logits": [
                "largest_float32_logit_whose_float32_sigmoid_is_not_above_threshold",
                "original_model_logit",
            ],
            "class_order": ["benign", "attack"],
            "threshold": fronteira_decisao,
            "threshold_operator": ">",
            "operational_logit_boundary": fronteira_logit,
            "argmax_tie_class": "benign",
            "loss": "torch.nn.CrossEntropyLoss",
            "optimizer": "torch.optim.Adam",
            "optimizer_learning_rate": 0.001,
            "optimizer_role": "required by ART API; no training performed",
            "input_shape": [49],
            "nb_classes": 2,
            "clip_values": [0.0, 1.0],
            "device_type": "cpu",
        },
    )
