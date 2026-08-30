# Uso e distribuição

Este repositório reúne código, dados derivados e um modelo treinado para a
prática acadêmica do minicurso SBSeg 2026.

## Código

O código desenvolvido para a prática é distribuído sob a licença MIT descrita
em [`LICENSE`](LICENSE).

## Dados

Os quatro recortes em `data/` derivam do
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) e de
transformações do pipeline D-MOOD. Eles podem ser redistribuídos com este
material para ensino, pesquisa e reprodução da prática. A origem e o
`data/manifest.json` devem ser preservados.

Os fatores +10%, +50% e +100% são configurações nominais da manipulação de
payload.

## Modelo

O checkpoint em `model/` provém do projeto
[D-MOOD](https://github.com/espindolaallan/d-mood). Ele pode ser redistribuído
com este material para ensino, pesquisa e reprodução da prática. Sua origem e
identidade estão registradas em `model/artifact.json`.

Estas permissões não alteram as condições de materiais ou dependências de
terceiros. Para outros usos, consulte as fontes originais.

## Citação

Ao reutilizar os recortes UNSW-NB15-Esp, cite o
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) como
conjunto de origem e o [D-MOOD](https://github.com/espindolaallan/d-mood) como
projeto que produziu os artefatos derivados.

Ao reutilizar o código ou o checkpoint, cite o D-MOOD.

Ao reutilizar a análise de coerência entre características aplicada aos
conjuntos manipulados e reextraídos, cite também:

Espindola, A. S.; Santin, A. O.; Casimiro, A.; Ferreira, P. M.; Viegas, E. K.
*Beyond Attack Success Rate: Representing Consistency in Evading the NIDS
Feature-Space*. Workshop de Cibersegurança em IA (WCIA), 2026.
