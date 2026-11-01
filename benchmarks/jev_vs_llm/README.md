# Jev × LLM: o mesmo trabalho, com acerto, tempo e custo

O argumento do projeto é "decisões baratas por turno com o Jev, LLM só nos momentos-chave".
Aqui o Jev e um LLM barato (Mistral Small 3.2, pela OpenRouter) fazem **o mesmo trabalho**,
com as mesmas perguntas, opções e rubricas (montadas do `questions.py`), sem cache e com 8
requisições em paralelo dos dois lados. Uma execução, 01/11/2026.

```bash
python benchmarks/jev_vs_llm/run.py --modelo mistralai/mistral-small-3.2-24b-instruct
```

**Preço do Jev:** US$ 0,042 por milhão de tokens de entrada, saída grátis, segundo sites de
terceiros ([1](https://flaviocopes.com/jev-pricing/), [2](https://www.layer3labs.io/guides/jev-pricing));
não confirmado na página da TypeSafe. O do Mistral vem do catálogo da OpenRouter.

## A. Satisfação da conversa (297 conversas do JDDC traduzidas, nota real)

| | Spearman | MAE | insatisfeitas achadas (de 76) | alarmes falsos | tempo por conversa | US$ / 1.000 conversas |
|---|---:|---:|---:|---:|---:|---:|
| **Jev** | **0,71** | 0,74 | 55 | 28 | **0,22 s** | **0,05** |
| Mistral Small 3.2 | 0,65 | 0,72 | 54 | 29 | 1,38 s | 0,08 |

ΔSpearman (Jev − Mistral): **+0,06, IC95% [+0,01; +0,11]**. Os dois acham quase as mesmas
insatisfeitas; o Jev ordena um pouco melhor, é 6× mais rápido e ~35% mais barato.

## B. Leitura por turno (759 turnos de 34 conversas)

| | tempo por turno | tempo total | custo |
|---|---:|---:|---:|
| **Jev** | **0,20 s** | **23 s** | **US$ 0,046** |
| Mistral Small 3.2 | 2,46 s | 255 s | US$ 0,074 |

- **Concordância** no cliente: sentimento Spearman 0,74; mesma emoção em 68% dos turnos.
- **Nota pelo sentimento final do cliente** contra a nota real: Jev 0,58, Mistral 0,48
  (34 conversas, amostra pequena).
- **Alertas (probabilidade ≥ 0,7):**

| alerta | Jev | Mistral | os dois |
|---|---:|---:|---:|
| cancelamento | 3 | 28 | 3 |
| pede escalada | 1 | 13 | 1 |
| sarcasmo | 0 | 6 | 0 |

  Não há rótulo por turno, mas lendo os turnos: no JDDC, "cancelar" é quase sempre cancelar um
  pedido, passagem ou cadastro, ou pedir reembolso, não abandonar a empresa (que é o que a
  pergunta pede). O Mistral marca esses casos; o Jev quase nunca. Num painel de alertas, os
  disparos do Mistral seriam, na maioria, falsos.

## Leitura

- **O Jev faz o mesmo trabalho 6 a 12× mais rápido, ~35–40% mais barato, com acerto igual ou
  um pouco melhor**, e com alertas bem mais conservadores.
- A economia aqui é **bem menor que os 96% do paper do Emo-Jev**, porque o comparador é um LLM
  já muito barato. Contra modelos maiores a diferença de custo cresce; a de tempo já é grande.
- Para um painel em tempo real, o tempo pesa mais que o custo: 23 s contra 4 min para ler 34
  conversas.

## Limites

- Uma execução, um LLM, conversas traduzidas do chinês; preço do Jev de terceiros.
- Algumas mensagens das traduções do piloto saíram com trechos de JSON no texto (defeito do
  tradutor); afetam os dois lados igualmente.
- Não medimos LLMs maiores (que tendem a acertar mais e custar mais).
