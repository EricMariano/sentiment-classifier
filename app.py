"""Interface em Streamlit: ``python -m streamlit run app.py``."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from sentiment_classifier.batch import CSV_COLUMNS, load_conversations, run_batch
from sentiment_classifier.classify import (
    JevTurnClassifier,
    OfflineHeuristicClassifier,
    TurnClassifier,
)
from sentiment_classifier.graph import (
    AnalysisState,
    Graph,
    analyze,
    build_graph,
    finalize,
    memory_checkpointer,
    run_config,
    stream_analysis,
)
from sentiment_classifier.parsing import parse_chat
from sentiment_classifier.providers import (
    has_jev,
    load_csat_model,
    make_classifier,
    make_csat_estimator,
    make_summarizer,
)

load_dotenv()
ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
SERIES = {"cliente": "#2a78d6", "atendente": "#eb6834"}
ALERT_ICON = {"cancelamento": "🚪", "escalar": "⬆️", "sarcasmo": "🙃"}
CAUSALITY = (
    "Pontos de virada (linhas tracejadas) marcam onde o sentimento do cliente mudou. A fala "
    "do atendente indicada veio **antes** da mudança: é correlação no tempo, não prova de "
    "que ela causou a mudança. Use como ponto de partida para ouvir ou ler o trecho."
)

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬", layout="wide")
st.title("Sentiment Classifier")
st.caption(
    "Análise de atendimento em PT-BR: o Jev lê cada turno (barato), "
    "o LLM só escreve sobre os momentos que mudaram a conversa."
)

with st.sidebar:
    st.header("Configuração")
    has_llm = bool(os.environ.get("CONVERSA_CHAT_MODEL"))
    offline = st.toggle("Forçar modo offline", value=not has_jev())
    csat_model = load_csat_model()
    st.markdown(
        f"- Jev: {'✅ chave encontrada' if has_jev() else '❌ sem `TYPESAFE_API_KEY`'}\n"
        f"- LLM: {'✅ ' + os.environ['CONVERSA_CHAT_MODEL'] if has_llm else '❌ sem `CONVERSA_CHAT_MODEL`'}\n"
        f"- Satisfação calibrada: {'✅ ' + str(csat_model.n_train) + ' conversas' if csat_model else '— sem modelo treinado'}"
    )
    names = st.text_input(
        "Nomes conhecidos para anonimizar", help="Separados por vírgula, como viriam do CRM."
    )
    review_on = st.toggle(
        "Revisão do supervisor antes de escalar",
        value=True,
        help="Com alerta de escalada ou cancelamento, o grafo pausa (interrupt do LangGraph) "
        "e só termina depois da sua decisão.",
    )
known_names = [n for n in names.split(",") if n.strip()]


def analysis_graph(classifier: TurnClassifier, own_checkpointer: bool) -> Graph:
    """O grafo do texto. Na ligação ele vira subgrafo e usa o checkpointer do grafo de fora."""
    return build_graph(
        classifier,
        make_summarizer(offline),
        make_csat_estimator(offline),
        csat_model,
        review=review_on,
        checkpointer=memory_checkpointer() if own_checkpointer else None,
    )


def run_live(graph: Any, request: Any, key: str, classifier: TurnClassifier) -> None:
    """Roda o grafo mostrando o progresso: a análise do Jev primeiro, depois o resumo do LLM
    sendo escrito. Guarda o resultado (e a pausa para revisão, se houver) na sessão."""
    config = run_config(f"{key}-{uuid.uuid4().hex[:8]}")
    progress = st.empty()
    live = st.container()
    progress.info("Lendo os turnos com o Jev…", icon="⏳")
    box = None
    written = ""
    pending = None
    final: AnalysisState = {}
    for event in stream_analysis(graph, request, config):
        if event.kind == "transcript":
            progress.info("Ligação transcrita. Lendo os turnos com o Jev…", icon="⏳")
        elif event.kind == "analysis" and event.state is not None:
            progress.info("Jev pronto. O LLM está escrevendo sobre os momentos-chave…", icon="✍️")
            with live:
                box = render(event.state, classifier, key, live=True)
        elif event.kind == "token" and box is not None:
            written += event.text
            box.markdown(written + " ▌")
        elif event.kind == "interrupt":
            pending = event.payload
        elif event.kind == "done" and event.state is not None:
            final = event.state
    st.session_state[key] = {
        "graph": graph,
        "config": config,
        "classifier": classifier,
        "state": final,
        "pending": pending,
    }
    st.rerun()


def review_panel(key: str) -> None:
    """O grafo pausou antes de escalar: o supervisor decide e o grafo retoma (Command)."""
    data = st.session_state[key]
    pending, decision = data.get("pending"), data["state"].get("review")
    if pending:
        with st.container(border=True):
            st.markdown(f"**Revisão do supervisor.** {pending['pergunta']}")
            for a in pending["alertas"]:
                st.markdown(
                    f"- {ALERT_ICON[a['tipo']]} **{a['tipo']}** no turno {a['turno']}: {a['motivo']}"
                )
            st.caption(
                f"Satisfação estimada: {pending['satisfacao_estimada']} / 5. O grafo está "
                "pausado (interrupt) e só termina com a sua decisão."
            )
            note = st.text_input("Observação (opcional)", key=f"note-{key}")
            c1, c2, _ = st.columns([1, 1, 3])
            choice = None
            if c1.button("Escalar", type="primary", key=f"yes-{key}"):
                choice = True
            if c2.button("Não escalar", key=f"no-{key}"):
                choice = False
            if choice is not None:
                state = data["graph"].invoke(
                    Command(resume={"escalar": choice, "nota": note}), data["config"]
                )
                data["state"], data["pending"] = finalize(state), None
                st.rerun()
    elif decision is not None:
        text = "Escalado pelo supervisor" if decision.escalate else "Supervisor decidiu não escalar"
        st.success(text + (f": {decision.note}" if decision.note else "."), icon="🧑‍💼")


def curve_chart(state: AnalysisState) -> alt.LayerChart:
    df = pd.DataFrame(
        {
            "turno": r.turn.index,
            "falante": r.turn.speaker,
            "sentimento": round(r.sentiment, 2),
            "rótulo": r.label,
            "texto": r.turn.text[:140],
            "voz": str((r.turn.voice or {}).get("leitura", "")),
        }
        for r in state["readings"]
    )
    color = alt.Color(
        "falante:N",
        scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
        legend=alt.Legend(orient="top", title=None),
    )
    base = alt.Chart(df).encode(
        x=alt.X("turno:Q", axis=alt.Axis(tickMinStep=1, grid=False, title="turno")),
        y=alt.Y(
            "sentimento:Q",
            scale=alt.Scale(domain=[1, 5]),
            axis=alt.Axis(values=[1, 2, 3, 4, 5], title="sentimento (cliente) / tom (atendente)"),
        ),
        color=color,
    )
    tooltip = ["turno", "falante", "sentimento", "rótulo", "texto"]
    if df["voz"].any():
        tooltip.append("voz")
    layers: list[alt.Chart] = [
        base.mark_line(strokeWidth=2, interpolate="monotone"),
        base.mark_circle(size=90, opacity=1, stroke="white", strokeWidth=2).encode(tooltip=tooltip),
    ]
    dyn = state["dynamics"]
    if dyn.turning_points:
        tp = pd.DataFrame(
            {
                "turno": t.customer_after,
                "virada": f"{t.direction} {t.delta:+.1f}",
                "antes": f"fala(s) do atendente antes: {list(t.agent_turns) or 'nenhuma'}",
                "aviso": "correlação no tempo, não prova de causa",
                "y": 6 + 13 * (i % 2),  # alterna a altura do rótulo para viradas vizinhas
            }
            for i, t in enumerate(dyn.turning_points)
        )
        layers = [
            alt.Chart(tp)
            .mark_rule(strokeDash=[4, 4], color="#8a8984")
            .encode(x="turno:Q", tooltip=["virada", "antes", "aviso"]),
            alt.Chart(tp)
            .mark_text(dy=-8, color="#8a8984", fontSize=11)
            .encode(x="turno:Q", y=alt.Y("y:Q", scale=None, axis=None), text="virada"),
            *layers,
        ]
    return alt.layer(*layers).properties(height=320)


def render(state: AnalysisState, classifier: TurnClassifier, key: str, live: bool = False) -> Any:
    """Desenha a análise. Com ``live=True`` (o LLM ainda escrevendo), devolve o espaço onde o
    resumo vai aparecendo."""
    readings, dyn, csat = state["readings"], state["dynamics"], state["csat"]
    if isinstance(classifier, OfflineHeuristicClassifier):
        st.warning(
            "Modo offline: as leituras vêm de uma lista de palavras, não do Jev. "
            "Serve para ver o fluxo; os números não dizem nada sobre a qualidade do Jev.",
            icon="⚠️",
        )
    if state["failures"]:
        st.warning(
            f"{len(state['failures'])} turno(s) ficaram sem leitura depois das tentativas "
            f"(turnos {[f.turn.index for f in state['failures']]}). A análise seguiu sem eles; "
            "a curva liga os pontos vizinhos.",
            icon="🕳️",
        )

    jev_calls = classifier.usage.requests if isinstance(classifier, JevTurnClassifier) else 0
    c1, c2, c3 = st.columns(3)
    c4, c5, _ = st.columns(3)
    c1.metric("Turnos", len(readings) + len(state["failures"]))
    cached = classifier.usage.cached if isinstance(classifier, JevTurnClassifier) else 0
    c2.metric(
        "Chamadas ao Jev",
        jev_calls,
        delta=f"{cached} do cache" if cached else None,
        delta_color="off",
        help="Uma por turno lido. Respostas já vistas vêm do cache, sem custo. 0 no modo offline.",
    )
    c3.metric(
        "Turnos ao LLM",
        len(dyn.key_turns) if make_summarizer(offline) else 0,
        help="O LLM só recebe os momentos-chave. 0 = resumo montado sem LLM.",
    )
    c4.metric(
        "Dados anonimizados",
        sum(state["redaction_counts"].values()),
        help="Trocados por marcadores antes de sair da máquina.",
    )
    csat_help = {
        "jev": "Estimativa do Jev sobre a conversa inteira, sem treino. Em 297 conversas de "
        "atendimento traduzidas para português, ordenou bem as notas (Spearman 0,71) e achou "
        "71% das insatisfeitas, mas comprime a escala.",
        "offline": "Sem Jev: repete o sentimento final do cliente. Não é uma estimativa.",
        "calibrado": "",
    }[csat.source]
    c5.metric(
        "Satisfação estimada",
        f"{csat.value:.1f} / 5",
        delta=(
            f"calibrado: {state['csat_predicted']:.1f}"
            if state["csat_predicted"] is not None
            else None
        ),
        delta_color="off",
        help=csat_help,
    )

    st.subheader("Curva da conversa")
    st.altair_chart(curve_chart(state), use_container_width=True)
    if dyn.turning_points:
        st.caption(CAUSALITY)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Análise")
        slot = None
        if live:
            st.caption(f"O LLM está escrevendo a partir de {len(dyn.key_turns)} turnos-chave…")
            slot = st.empty()
        else:
            summary = state["summary"]
            st.caption(
                f"Escrita pelo LLM a partir de {len(dyn.key_turns)} turnos-chave"
                if summary.source == "llm"
                else "Montada dos sinais, sem LLM"
            )
            if summary.note:
                st.info(summary.note, icon="ℹ️")
            st.markdown(summary.text)
        if dyn.turning_points:
            st.caption('Lembrete: "antes de" não é "por causa de".')
    with right:
        st.subheader("Alertas")
        if not dyn.alerts:
            st.success("Nenhum alerta.", icon="✅")
        for a in dyn.alerts:
            p = f" · p={a.probability:.2f}" if a.probability is not None else ""
            st.error(
                f"**{a.kind}** no turno {a.turn_index}{p}\n\n{a.reason}", icon=ALERT_ICON[a.kind]
            )
        if state["redaction_counts"]:
            st.caption(
                "Anonimizado antes de sair da máquina: "
                + ", ".join(f"{k} ×{v}" for k, v in state["redaction_counts"].items())
            )

    with st.expander("Leitura turno a turno (texto já anonimizado)"):
        rows = []
        for r in readings:
            row: dict[str, object] = {
                "turno": r.turn.index,
                "falante": r.turn.speaker,
                "texto": r.turn.text,
                "sentimento": round(r.sentiment, 2),
                "conf.": round(r.sentiment_confidence, 2),
                "rótulo": r.label,
                **{k: round(v, 2) for k, v in r.flags.items()},
            }
            if r.turn.voice:
                row["voz"] = r.turn.voice.get("leitura", "")
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            key=f"t-{key}-{'live' if live else 'final'}",
        )
    return slot


tab_text, tab_call, tab_batch = st.tabs(["💬 Conversa", "📞 Ligação", "📋 Lote"])

# --- Conversa em texto --------------------------------------------------------------------
with tab_text:
    samples = sorted(p.name for p in SAMPLES.glob("*.txt"))
    choice = st.selectbox("Conversa", ["(colar a minha)", *samples], index=1)
    default = "" if choice.startswith("(") else (SAMPLES / choice).read_text()
    raw = st.text_area(
        "Uma fala por linha, começando com `Cliente:` ou `Atendente:`", default, height=200
    )
    # Só analisa no clique: o Streamlit reexecuta o script a cada interação, e cada análise
    # custa uma requisição ao Jev por turno.
    if st.button("Analisar", type="primary", disabled=not raw.strip()):
        try:
            turns = parse_chat(raw)
        except ValueError as exc:
            st.error(f"Não consegui ler a conversa: {exc}")
        else:
            classifier = make_classifier(offline)
            graph = analysis_graph(classifier, own_checkpointer=True)
            run_live(graph, {"turns": turns, "known_names": known_names}, "text", classifier)
    if "text" in st.session_state:
        review_panel("text")
        render(st.session_state.text["state"], st.session_state.text["classifier"], "text")

# --- Lote ---------------------------------------------------------------------------------
with tab_batch:
    st.markdown(
        "Várias conversas de uma vez: `.txt` (formato `Cliente:`) ou `.jsonl` "
        '(`{"id", "messages": [{"role", "content"}], "nota"}`). Uma conversa com problema vira '
        "uma linha de erro; o lote continua."
    )
    files = st.file_uploader(
        "Arquivos", type=["txt", "jsonl"], accept_multiple_files=True, key="batch_files"
    )
    use_samples = st.checkbox("Usar as conversas de exemplo", value=not files)
    with_llm = st.checkbox("Resumo com LLM nas conversas com virada ou alerta", value=False)
    if st.button("Analisar lote", type="primary", disabled=not (files or use_samples)):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [SAMPLES] if use_samples else []
            for f in files or []:
                p = Path(tmp) / f.name
                p.write_bytes(f.getvalue())
                paths.append(p)
            items = list(load_conversations(paths))
        bar = st.progress(0.0, text=f"0 de {len(items)}")
        st.session_state.batch = run_batch(
            items,
            make_classifier(offline),
            make_summarizer(offline) if with_llm else None,
            make_csat_estimator(offline),
            csat_model,
            lambda n, total, row: bar.progress(n / total, text=f"{n} de {total}: {row.id}"),
        )

    if "batch" in st.session_state:
        rows = st.session_state.batch
        c1, c2, c3 = st.columns(3)
        c1.metric("Conversas", len(rows))
        c2.metric("Pedem atenção", sum(1 for r in rows if r.alertas or r.erro))
        c3.metric("Com erro", sum(1 for r in rows if r.erro))
        table = pd.DataFrame([asdict(r) for r in rows])[CSV_COLUMNS]
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button(
            "Baixar CSV", table.to_csv(index=False).encode(), "relatorio.csv", "text/csv"
        )
        for r in rows:
            if r.resumo:
                with st.expander(f"Resumo: {r.id}"):
                    st.markdown(r.resumo)

# --- Ligação (por último: depende do extra de áudio) --------------------------------------
with tab_call:
    try:
        from sentiment_classifier.audio.pipeline import build_call_graph
        from sentiment_classifier.audio.transcribe import (
            LocalStereoTranscriber,
            OpenAIDiarizedTranscriber,
        )
    except ImportError:
        st.info('Instale o extra de áudio: `pip install -e ".[audio]"`')
        st.stop()

    @st.cache_resource(show_spinner="Carregando o Whisper…")
    def local_transcriber(channel: str) -> LocalStereoTranscriber:
        return LocalStereoTranscriber(agent_channel=channel)  # type: ignore[arg-type]

    audio_samples = sorted(p.name for p in (SAMPLES / "audio").glob("*.flac"))
    source = st.radio("Áudio", ["exemplo", "enviar arquivo"], horizontal=True)
    audio_path: Path | None = None
    if source == "exemplo":
        picked = st.selectbox("Ligação de exemplo (sintética, mesmo texto)", audio_samples)
        audio_path = SAMPLES / "audio" / picked
        st.caption(
            "As duas ligações têm o mesmo texto. Na irritada, o cliente fala mais alto, mais "
            "rápido e atropela o atendente. Voz sintética: é demonstração, não evidência."
        )
    else:
        upload = st.file_uploader("Gravação", type=["wav", "flac", "ogg", "mp3"])
        if upload:
            with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as tmp:
                tmp.write(upload.getvalue())
            audio_path = Path(tmp.name)
    if audio_path:
        st.audio(str(audio_path))

    col_a, col_b = st.columns(2)
    mono = col_a.toggle(
        "Áudio mono (separar falantes na OpenAI)",
        help="Sem um canal por pessoa, a separação de falantes é feita pelo "
        "gpt-4o-transcribe-diarize da OpenAI.",
    )
    channel = col_b.selectbox(
        "Canal do atendente", ["esquerdo", "direito"], disabled=mono, help="Gravação estéreo."
    )
    if mono:
        st.warning(
            "No modo mono, **o áudio original vai para a OpenAI antes da anonimização**, com "
            "qualquer CPF, nome ou endereço falado. No estéreo, tudo fica na máquina.",
            icon="🔒",
        )
    compare = st.checkbox(
        "Comparar com a leitura só pelo texto",
        help="Roda o Jev duas vezes: com e sem os sinais de voz. Dobra as chamadas.",
    )

    if st.button("Transcrever e analisar", type="primary", disabled=audio_path is None):
        assert audio_path is not None
        transcriber = OpenAIDiarizedTranscriber() if mono else local_transcriber(channel)
        role_client = None
        if mono and has_jev() and not offline:
            from langchain_typesafe import TypeSafeClassifier

            role_client = TypeSafeClassifier()
        classifier = make_classifier(offline)
        # A análise do texto entra como subgrafo do grafo da ligação.
        graph = build_call_graph(
            transcriber,
            analysis_graph(classifier, own_checkpointer=False),
            role_client,
            memory_checkpointer(),
        )
        if compare:
            st.session_state.pop("call_text_only", None)
            st.session_state.compare_next = True
        try:
            run_live(
                graph, {"path": str(audio_path), "known_names": known_names}, "call", classifier
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    if "call" in st.session_state and st.session_state.pop("compare_next", False):
        # Só as leituras do Jev, sem voz: a comparação não precisa de resumo nem de nota.
        with st.spinner("Lendo de novo, só pelo texto…"):
            call_turns = st.session_state.call["state"]["call"].turns
            st.session_state.call_text_only = analyze(
                [replace(t, voice=None) for t in call_turns],
                make_classifier(offline),
                known_names=known_names,
            )

    if "call" in st.session_state:
        data = st.session_state.call
        state, classifier = data["state"], data["classifier"]
        call, text_only = state["call"], st.session_state.get("call_text_only")
        review_panel("call")
        if call.agent_confidence is not None:
            st.caption(f"Atendente identificado pelo Jev (confiança {call.agent_confidence:.2f}).")
        with st.expander(f"Transcrição e voz ({call.duration:.0f}s, {len(call.turns)} turnos)"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "início (s)": round(c.start, 1),
                        "falante": t.speaker,
                        "texto": t.text,
                        "voz": (t.voice or {}).get("leitura", ""),
                        "volume vs início (dB)": (t.voice or {}).get("volume_vs_inicio_db"),
                        "velocidade vs início (%)": (t.voice or {}).get("velocidade_vs_inicio_pct"),
                        "interrompeu (s)": (t.voice or {}).get("interrompeu_o_outro_s"),
                    }
                    for t, c in zip(call.turns, call.call, strict=True)
                ),
                hide_index=True,
                use_container_width=True,
            )
        if text_only is not None:
            by_turn = {r.turn.index: r for r in text_only["readings"]}
            st.subheader("Com voz × só texto")
            st.dataframe(
                pd.DataFrame(
                    {
                        "turno": r.turn.index,
                        "texto": r.turn.text,
                        "voz": (r.turn.voice or {}).get("leitura", ""),
                        "sentimento com voz": round(r.sentiment, 2),
                        "sentimento só texto": (
                            round(by_turn[r.turn.index].sentiment, 2)
                            if r.turn.index in by_turn
                            else None
                        ),
                        "sarcasmo com voz": round(r.flags.get("sarcasmo", 0), 2),
                        "sarcasmo só texto": (
                            round(by_turn[r.turn.index].flags.get("sarcasmo", 0), 2)
                            if r.turn.index in by_turn
                            else None
                        ),
                    }
                    for r in state["readings"]
                    if r.turn.speaker == "cliente"
                ),
                hide_index=True,
                use_container_width=True,
            )
        render(state, classifier, key="call")
