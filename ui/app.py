import os
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
CATEGORIAS = ["Mecânica", "Elétrica", "Automação"]
PRIORIDADES = ["Baixa", "Média", "Alta"]
MODOS = {"Híbrido": "hybrid", "Somente ML": "ml", "Somente Gemini": "llm"}
HISTORY_PAGE_SIZE = 200

st.set_page_config(page_title="Classificador de Chamados", page_icon="🛠️", layout="wide")


def call_api(method: str, path: str, **kwargs: Any) -> Any | None:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=90, **kwargs)
    except httpx.HTTPError:
        st.error(f"API offline ou inacessível em {API_URL}. Inicie com `uvicorn app.main:app --reload`.")
        st.stop()
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.reason_phrase
        st.error(f"Erro {response.status_code}: {detail}")
        return None
    return response.json()


def render_sidebar() -> None:
    st.sidebar.title("Status")
    health = call_api("GET", "/health")
    if not health:
        return
    st.sidebar.write(f"API: **{health['status']}**")
    for target, loaded in health["modelos_carregados"].items():
        st.sidebar.write(f"Modelo {target}: {'✅' if loaded else '❌ treine o modelo'}")
    gemini = health["gemini"]
    if gemini["disponivel"]:
        st.sidebar.write(f"Gemini: ✅ `{gemini['modelo']}`")
    else:
        st.sidebar.write("Gemini: ❌ não configurado (usando apenas ML)")
    if gemini.get("ultimo_erro"):
        st.sidebar.warning(f"Último erro do Gemini: {gemini['ultimo_erro']}")
    st.sidebar.caption(f"Endpoint: {API_URL}")


def render_flags(flags: dict[str, Any]) -> None:
    if flags["regra_aplicada"]:
        st.warning(f"Regra de segurança elevou a prioridade para Alta (termos: {', '.join(flags['palavras_criticas'])}).")
    if flags["revisao_humana"]:
        st.error("Revisão humana recomendada.")
    if flags["fallback"]:
        st.info(f"Fallback para ML (motivo: {flags['motivo_fallback']}).")
    if flags["escalado_para_llm"]:
        st.caption("Chamado escalado ao Gemini por baixa confiança do ML.")


def render_result(result: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("Categoria", result["categoria"])
    columns[1].metric("Prioridade", result["prioridade"])
    columns[2].metric("Confiança", f"{result['confianca']:.0%}")
    columns[3].metric("Motor", result["motor_usado"])
    render_flags(result["flags"])
    st.write(f"**Ação sugerida:** {result['acao_sugerida']}")
    explanation = result["explicacao"]
    if explanation["termos_ml"]:
        terms = pd.DataFrame(explanation["termos_ml"]).rename(columns={"termo": "Termo", "peso": "Contribuição"})
        st.write("**Termos que mais contribuíram (ML)**")
        st.dataframe(terms, hide_index=True, use_container_width=True)
    if explanation["justificativa_llm"]:
        st.write(f"**Justificativa (Gemini):** {explanation['justificativa_llm']}")
        st.write(f"**Causa provável:** {explanation['causa_provavel']}")
        st.write(f"**Equipamento:** {explanation['equipamento_detectado']}")
    caption = f"Latência total: {result['latencia_ms']:.0f} ms"
    if result["llm_latencia_ms"] is not None:
        caption += f" · Gemini: {result['llm_latencia_ms']:.0f} ms · tokens {result['tokens_entrada']}/{result['tokens_saida']} · custo estimado US$ {result['custo_estimado']:.5f}"
    st.caption(caption)


def render_comparison(text: str) -> None:
    left, right = st.columns(2)
    for column, label, mode in ((left, "ML", "ml"), (right, "Gemini", "llm")):
        with column:
            st.subheader(label)
            result = call_api("POST", "/classificar", json={"texto": text, "modo": mode, "persistir": False})
            if result:
                render_result(result)


def render_correction(result: dict[str, Any]) -> None:
    st.divider()
    st.write("**Confirmar ou corrigir**")
    columns = st.columns(3)
    category = columns[0].selectbox("Categoria correta", CATEGORIAS, index=CATEGORIAS.index(result["categoria"]))
    priority = columns[1].selectbox("Prioridade correta", PRIORIDADES, index=PRIORIDADES.index(result["prioridade"]))
    if columns[2].button("Salvar correção"):
        saved = call_api("PATCH", f"/chamados/{result['id']}/correcao", json={"categoria": category, "prioridade": priority})
        if saved:
            st.success(f"Registrado como {saved['status']}.")
    if columns[2].button("Confirmar classificação"):
        if call_api("POST", f"/chamados/{result['id']}/confirmar"):
            st.success("Classificação confirmada.")


def tab_classify() -> None:
    text = st.text_area("Descrição do chamado", height=140, placeholder="Ex.: Motor da esteira desarmando com cheiro de queimado")
    mode_label = st.radio("Modo", list(MODOS), horizontal=True)
    buttons = st.columns(2)
    if buttons[0].button("Classificar", type="primary") and text.strip():
        st.session_state["last_result"] = call_api("POST", "/classificar", json={"texto": text, "modo": MODOS[mode_label]})
        st.session_state.pop("comparison_text", None)
    if buttons[1].button("Comparar ML vs Gemini") and text.strip():
        st.session_state["comparison_text"] = text
    if st.session_state.get("comparison_text"):
        render_comparison(st.session_state["comparison_text"])
    result = st.session_state.get("last_result")
    if result:
        render_result(result)
        render_correction(result)


def fetch_history(filters: dict[str, Any]) -> pd.DataFrame:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        body = call_api("GET", "/chamados", params={**filters, "pagina": page, "tamanho": HISTORY_PAGE_SIZE})
        if not body:
            break
        items.extend(body["itens"])
        if len(items) >= body["total"] or not body["itens"]:
            break
        page += 1
    return pd.DataFrame(items)


def tab_history() -> None:
    columns = st.columns(5)
    category = columns[0].selectbox("Categoria", ["Todas", *CATEGORIAS])
    priority = columns[1].selectbox("Prioridade", ["Todas", *PRIORIDADES])
    status = columns[2].selectbox("Status", ["Todos", "pendente", "confirmado", "corrigido"])
    start = columns[3].date_input("De", value=None)
    end = columns[4].date_input("Até", value=None)
    filters = {
        "categoria": None if category == "Todas" else category,
        "prioridade": None if priority == "Todas" else priority,
        "status": None if status == "Todos" else status,
        "data_inicio": start.isoformat() if start else None,
        "data_fim": end.isoformat() if end else None,
    }
    frame = fetch_history({key: value for key, value in filters.items() if value})
    if frame.empty:
        st.info("Nenhum chamado encontrado.")
        return
    visible = frame.drop(columns=["termos_ml", "palavras_criticas"])
    st.dataframe(visible, use_container_width=True, hide_index=True)
    st.download_button("Exportar CSV", visible.to_csv(index=False).encode("utf-8"), "chamados.csv", "text/csv")


def engine_summary(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, data in report["motores"].items():
        rows.append(
            {
                "Motor": name,
                "Accuracy (categoria)": data["categoria"]["accuracy"],
                "F1-macro (categoria)": data["categoria"]["f1_macro"],
                "Accuracy (prioridade)": data["prioridade"]["accuracy"],
                "F1-macro (prioridade)": data["prioridade"]["f1_macro"],
                "Latência média (ms)": data["latencia_media_ms"],
                "Latência p95 (ms)": data["latencia_p95_ms"],
                "Custo por 1000 (US$)": data["custo_por_1000_usd"],
                "% escalado ao LLM": data.get("pct_escalado_para_llm"),
            }
        )
    return pd.DataFrame(rows)


def tab_benchmark() -> None:
    try:
        response = httpx.get(f"{API_URL}/benchmark", timeout=30)
    except httpx.HTTPError:
        st.error("API offline.")
        return
    if response.status_code == 404:
        st.info("Benchmark ainda não executado. Rode `python -m ml.benchmark` (use `--amostra 30` para limitar custo do Gemini).")
        return
    report = response.json()
    st.caption(f"Gerado em {report['gerado_em']} · {report['n_amostras']} chamados do teste congelado · Gemini incluído: {report['gemini_incluido']}")
    summary = engine_summary(report)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    quality = summary.melt(id_vars="Motor", value_vars=["F1-macro (categoria)", "F1-macro (prioridade)"], var_name="Alvo", value_name="F1")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(quality, x="Motor", y="F1", color="Alvo", barmode="group", title="F1-macro por motor"), use_container_width=True)
    right.plotly_chart(px.bar(summary, x="Motor", y="Latência média (ms)", title="Latência média"), use_container_width=True)
    st.plotly_chart(px.bar(summary, x="Motor", y="Custo por 1000 (US$)", title="Custo estimado por 1.000 chamados"), use_container_width=True)
    engine = st.selectbox("Matriz de confusão do motor", list(report["motores"]))
    columns = st.columns(2)
    for column, target in zip(columns, ("categoria", "prioridade")):
        matrix = report["motores"][engine][target]["matriz_confusao"]
        figure = px.imshow(matrix["matriz"], x=matrix["labels"], y=matrix["labels"], text_auto=True, labels={"x": "Previsto", "y": "Real"}, title=target.capitalize())
        column.plotly_chart(figure, use_container_width=True)
    analysis = report.get("discordancias")
    if analysis:
        st.subheader(f"Discordâncias ML vs Gemini ({analysis['total']})")
        st.write(f"Quem acertou: {analysis['quem_acertou']}")
        cases = [
            {
                "Texto": case["texto"],
                "Verdadeiro": f"{case['verdadeiro']['categoria']} / {case['verdadeiro']['prioridade']}",
                "ML": f"{case['ml']['categoria']} / {case['ml']['prioridade']}",
                "Gemini": f"{case['llm']['categoria']} / {case['llm']['prioridade']}",
                "Acertou": case["quem_acertou"],
            }
            for case in analysis["casos"]
        ]
        st.dataframe(pd.DataFrame(cases), use_container_width=True, hide_index=True)


def tab_dashboard() -> None:
    body = call_api("GET", "/metricas")
    if not body:
        return
    usage = body["uso"]
    columns = st.columns(4)
    columns[0].metric("Chamados", usage["total_chamados"])
    rate = usage["taxa_correcao"]
    columns[1].metric("Taxa de correção", f"{rate:.0%}" if rate is not None else "—")
    columns[2].metric("Revisões recomendadas", usage["revisoes_humanas_recomendadas"])
    columns[3].metric("Fallbacks", usage["fallbacks"])
    if not usage["total_chamados"]:
        st.info("Classifique alguns chamados para ver os gráficos.")
        return
    left, right = st.columns(2)
    category = pd.Series(usage["por_categoria"], name="Chamados").rename_axis("Categoria").reset_index()
    priority = pd.Series(usage["por_prioridade"], name="Chamados").rename_axis("Prioridade").reset_index()
    left.plotly_chart(px.bar(category, x="Categoria", y="Chamados", title="Volume por categoria"), use_container_width=True)
    right.plotly_chart(px.bar(priority, x="Prioridade", y="Chamados", title="Volume por prioridade"), use_container_width=True)
    series = pd.DataFrame(usage["serie_diaria"])
    st.plotly_chart(px.line(series, x="dia", y="total", markers=True, title="Evolução diária"), use_container_width=True)
    gemini = usage["gemini"]
    st.subheader("Gemini")
    llm_columns = st.columns(4)
    llm_columns[0].metric("Chamadas", gemini["chamadas"])
    llm_columns[1].metric("Tokens (entrada/saída)", f"{gemini['tokens_entrada']}/{gemini['tokens_saida']}")
    llm_columns[2].metric("Custo estimado (US$)", f"{gemini['custo_total_estimado']:.4f}")
    latency = gemini["latencia_media_ms"]
    llm_columns[3].metric("Latência média (ms)", f"{latency:.0f}" if latency else "—")


def metrics_table(metrics: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    for target in ("categoria", "prioridade"):
        holdout = metrics[target]["holdout"] if metrics else None
        rows.append(
            {
                "Alvo": target,
                "Modelo": metrics[target]["melhor_modelo"] if metrics else "—",
                "Accuracy": holdout["accuracy"] if holdout else None,
                "F1-macro": holdout["f1_macro"] if holdout else None,
            }
        )
    return pd.DataFrame(rows)


def tab_retrain() -> None:
    body = call_api("GET", "/metricas")
    if body:
        status = body["uso"]["por_status"]
        st.write(f"Corrigidos: **{status.get('corrigido', 0)}** · Confirmados: **{status.get('confirmado', 0)}**")
    st.caption("O retreino incorpora correções e confirmações humanas e avalia no mesmo teste congelado.")
    if st.button("Retreinar modelo de ML", type="primary"):
        with st.spinner("Treinando..."):
            outcome = call_api("POST", "/retreinar")
        if outcome:
            left, right = st.columns(2)
            left.subheader("Antes")
            left.dataframe(metrics_table(outcome["antes"]), hide_index=True, use_container_width=True)
            right.subheader("Depois")
            right.dataframe(metrics_table(outcome["depois"]), hide_index=True, use_container_width=True)
            st.write(f"Exemplos incorporados: {outcome['correcoes_incorporadas']}")


render_sidebar()
st.title("Classificador Inteligente de Chamados de Manutenção")
classify, history, benchmark, dashboard, retrain = st.tabs(["Classificar", "Histórico", "Benchmark", "Dashboard", "Retreinar"])
with classify:
    tab_classify()
with history:
    tab_history()
with benchmark:
    tab_benchmark()
with dashboard:
    tab_dashboard()
with retrain:
    tab_retrain()
