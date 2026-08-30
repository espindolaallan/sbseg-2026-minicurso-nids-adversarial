# Entendendo o Adversário: Da Taxonomia à Prática em Aprendizado de Máquina Adversarial para Detecção de Intrusões em Redes

Material prático do minicurso apresentado no
[SBSeg 2026](https://www.sbseg2026.uff.br/chamadas/minicursos/).

## Mini-laboratório: avaliação adversarial de NIDS

Nesta prática, você avaliará como mudanças nas entradas de um NIDS afetam suas
decisões. Ao final, poderá comparar diferentes formas de ataque, interpretar os
resultados e distinguir as evidências produzidas em cada espaço experimental.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espindolaallan/sbseg-2026-minicurso-nids-adversarial/blob/main/notebooks/01_minilab.ipynb)

No Colab, selecione **Ambiente de execução > Executar tudo**.

## Experimentos e métricas

- **Espaço de características:** FGM minimal e PGD modificam a mesma amostra
  `E_N` de fluxos DoS corretamente detectados pelo NIDS na avaliação sem
  perturbação.
- **Espaço do problema:** conjuntos pré-computados representam manipulações
  nominais de payload de +10%, +50% e +100%, operacionalizadas pela alteração
  de `IP.len` e seguidas de reextração.
- **ASR:** `evasões / |E_N|`. No espaço de características, existe pareamento
  entre as amostras originais e modificadas.
- **Recall:** `TP / (TP + FN)`.
- **FNR:** `FN / (TP + FN)`. Recall e FNR são usados para os conjuntos no
  espaço do problema, sem calcular ASR.
