"""As perguntas feitas ao Jev em cada turno, separadas por falante.

Todas as perguntas de um turno vão numa única requisição. O estado enviado é um JSON
com o turno atual e alguns turnos anteriores como contexto (ver ``turn_state``), porque
sarcasmo e mudança de humor só fazem sentido em relação ao que veio antes.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_typesafe import Choice, Noul, NoulCriteria, Question, Score

from sentiment_classifier.models import Speaker, Turn

CONTEXT_TURNS = 4

_ABOUT = (
    "Avalie SOMENTE a fala em `turno_atual`, de um atendimento ao cliente em português do "
    "Brasil. Use `contexto` apenas para interpretar o tom (ironia, mudança de humor). Se "
    "`turno_atual.voz` existir, a conversa foi uma ligação: são medidas do áudio comparadas "
    "a como AQUELE falante falava no começo da ligação (velocidade, volume, tom de voz), mais "
    "se ele interrompeu o outro e o silêncio antes de falar. Use-as junto com o texto; elas pesam mais quando o texto é "
    "ambíguo, como um 'ótimo, obrigado' que pode ser sincero ou irônico."
)

CUSTOMER_EMOTIONS: dict[str, str] = {
    "frustracao": "Cansado de tentar sem resultado; repete o problema, reclama da demora.",
    "raiva": "Hostil, acusa, ameaça, usa linguagem forte ou caixa alta.",
    "confusao": "Não entende o que está acontecendo ou o que precisa fazer.",
    "ansiedade": "Preocupado com consequências: prazo, dinheiro, perder o serviço.",
    "alivio": "Problema resolvido ou encaminhado; a tensão diminuiu.",
    "satisfacao": "Contente com o atendimento ou com a solução.",
    "neutro": "Fala informativa, sem emoção marcada.",
}

AGENT_POSTURES: dict[str, str] = {
    "empatico": "Reconhece o problema e o sentimento do cliente com palavras próprias.",
    "resolutivo": "Age ou propõe um passo concreto que avança a solução.",
    "burocratico": "Repete procedimento, script ou política sem adaptar ao caso.",
    "defensivo": "Culpa o cliente, outro setor ou o sistema; nega responsabilidade.",
    "neutro": "Pede ou confirma informação, sem postura marcada.",
}

CUSTOMER_QUESTIONS: dict[str, Question] = {
    "sentimento": Score(
        instructions=_ABOUT + " Como o cliente está se sentindo nesta fala?",
        criteria=[
            "Muito negativo: irritado, hostil ou desesperado.",
            "Negativo: insatisfeito ou frustrado, mas contido.",
            "Neutro: informativo, sem carga emocional clara.",
            "Positivo: cordial, colaborativo, com alguma satisfação.",
            "Muito positivo: grato, aliviado ou elogiando de forma sincera.",
        ],
    ),
    "emocao": Choice(
        instructions=_ABOUT + " Qual emoção predomina na fala do cliente?",
        criteria=dict(CUSTOMER_EMOTIONS),
    ),
    "sarcasmo": Noul(
        instructions=_ABOUT
        + " O cliente está sendo sarcástico ou irônico, dizendo algo positivo com sentido "
        "negativo (por exemplo, 'ótimo, mais uma vez' depois de um problema repetido)?",
        criteria=NoulCriteria(
            true="O sentido literal positivo é desmentido pelo contexto.",
            false="Elogio ou agradecimento sincero, ou fala sem ironia.",
        ),
    ),
    "cancelamento": Noul(
        instructions=_ABOUT
        + " O cliente expressa intenção de cancelar, trocar de empresa ou encerrar o "
        "contrato, mesmo que de forma educada ou condicional?",
    ),
    "pede_escalada": Noul(
        instructions=_ABOUT
        + " O cliente pede supervisor, gerente, outro setor, ou ameaça reclamar em órgão "
        "externo (Procon, Reclame Aqui, Justiça, redes sociais)?",
    ),
}

AGENT_QUESTIONS: dict[str, Question] = {
    "sentimento": Score(
        instructions=_ABOUT + " Qual é o tom do atendente nesta fala?",
        criteria=[
            "Hostil, ríspido ou desrespeitoso.",
            "Frio ou impaciente.",
            "Neutro e protocolar.",
            "Cordial.",
            "Empático e claramente empenhado em resolver.",
        ],
    ),
    "postura": Choice(
        instructions=_ABOUT + " Qual postura predomina na fala do atendente?",
        criteria=dict(AGENT_POSTURES),
    ),
    "solucao_concreta": Noul(
        instructions=_ABOUT
        + " O atendente oferece uma solução ou próximo passo concreto, com prazo ou ação "
        "definida (não apenas 'vamos verificar')?",
    ),
}

QUESTIONS: dict[Speaker, dict[str, Question]] = {
    "cliente": CUSTOMER_QUESTIONS,
    "atendente": AGENT_QUESTIONS,
}
LABEL_QUESTION: dict[Speaker, str] = {"cliente": "emocao", "atendente": "postura"}


def turn_state(
    turns: Sequence[Turn], index: int, context: int = CONTEXT_TURNS
) -> dict[str, object]:
    """Estado JSON de um turno: a fala atual mais as ``context`` falas anteriores."""
    current = turns[index]
    previous = turns[max(0, index - context) : index]
    turno_atual: dict[str, object] = {"falante": current.speaker, "texto": current.text}
    if current.voice:
        turno_atual["voz"] = current.voice
    return {
        "contexto": [{"falante": t.speaker, "texto": t.text} for t in previous],
        "turno_atual": turno_atual,
    }
