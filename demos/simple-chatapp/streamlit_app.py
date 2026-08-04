import os


# Reduce macOS runtime instability from tokenizer worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
# If users explicitly force MPS, allow safe CPU fallback for unsupported ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Imports must follow the environment setup above to configure ML runtimes before initialization.
import streamlit as st  # type: ignore  # noqa: E402
from demo_utils import (  # noqa: E402
    ATTACK_PRESETS,
    VERDICT_COLORS,
    build_explanations,
    layer_statuses,
    run_unprotected_baseline,
    summarize_pipeline,
)

from core.orchestrator import PIPipeline  # noqa: E402
from core.settings import Settings  # noqa: E402


st.set_page_config(
    page_title="Barrikade Open Day Demo",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

            :root {
                --bg: #0d0f12;
                --surface: #13161b;
                --border: #1e2330;
                --muted: #3a3f52;
                --text: #c8cdd8;
                --dim: #5a6075;
                --green: #22d47a;
                --red: #ff4f4f;
                --amber: #f5a623;
                --blue: #4a9fff;
            }

            .stApp {
                background: var(--bg);
                color: var(--text);
                font-family: 'IBM Plex Mono', monospace;
            }

            .stApp::before {
                content: '';
                position: fixed;
                inset: 0;
                background-image:
                    linear-gradient(var(--border) 1px, transparent 1px),
                    linear-gradient(90deg, var(--border) 1px, transparent 1px);
                background-size: 40px 40px;
                opacity: 0.35;
                pointer-events: none;
                z-index: 0;
            }

            [data-testid="stMainBlockContainer"] {
                max-width: 780px;
                z-index: 1;
            }

            [data-testid="stMainBlockContainer"],
            [data-testid="stMainBlockContainer"] p,
            [data-testid="stMainBlockContainer"] li,
            [data-testid="stMainBlockContainer"] label,
            [data-testid="stMainBlockContainer"] h1,
            [data-testid="stMainBlockContainer"] h2,
            [data-testid="stMainBlockContainer"] h3 {
                color: var(--text) !important;
            }

            .gg-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                margin-bottom: 24px;
            }

            .headline {
                font-size: 1.7rem;
                font-family: 'IBM Plex Mono', monospace;
                font-weight: 600;
                letter-spacing: -0.2px;
                margin-bottom: 0.2rem;
                color: #ffffff;
            }

            .headline span {
                color: var(--green);
            }

            .subhead {
                color: var(--dim);
                margin-bottom: 0;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 11px;
                letter-spacing: 0.05em;
            }

            .status-pill {
                display: flex;
                align-items: center;
                gap: 7px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 6px 14px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 11px;
                color: var(--dim);
            }

            .status-dot {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: var(--green);
                box-shadow: 0 0 6px var(--green);
            }

            .input-panel,
            .result-panel {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 14px;
            }

            .panel-label {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 10px;
                letter-spacing: 0.12em;
                color: var(--dim);
                text-transform: uppercase;
                margin-bottom: 10px;
            }

            .trace-title {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 11px;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                color: var(--dim);
                margin: 0.45rem 0 0.65rem 0;
            }

            .trace-callout {
                background: #0f1217;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.55rem 0.75rem;
                margin: 0.30rem 0 0.45rem 0;
                color: var(--text);
                font-family: 'IBM Plex Mono', monospace;
                line-height: 1.45;
            }

            .verdict-card {
                padding: 0.95rem 1rem;
                border-radius: 8px;
                border: 1px solid var(--border);
                background: #0f1217;
                color: var(--text);
                margin-bottom: 0.75rem;
            }

            .verdict-kicker {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 10px;
                letter-spacing: 0.12em;
                color: var(--dim);
                text-transform: uppercase;
                margin: 0 0 0.35rem 0;
            }

            .verdict-title {
                margin: 0;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 1.05rem;
                line-height: 1.3;
                letter-spacing: 0.01em;
            }

            .verdict-meta {
                margin-top: 0.6rem;
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
            }

            .verdict-chip {
                border: 1px solid var(--border);
                border-radius: 999px;
                padding: 0.18rem 0.55rem;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 10px;
                line-height: 1.3;
                color: var(--text);
                background: var(--surface);
            }

            [data-testid="stMetricLabel"] *,
            [data-testid="stMetricValue"] {
                color: var(--text) !important;
            }

            .stTextArea textarea,
            .stTextInput input,
            [data-testid="stCodeBlock"] pre {
                color: var(--text) !important;
                background-color: var(--bg) !important;
                border: 1px solid var(--border) !important;
                border-radius: 7px !important;
                font-family: 'IBM Plex Mono', monospace !important;
            }

            .stButton button[kind="secondary"] {
                color: var(--dim) !important;
                background: transparent !important;
                border: 1px solid var(--muted) !important;
                border-radius: 4px !important;
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 11px !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
                line-height: 1.2 !important;
                min-height: 2.75rem !important;
                height: auto !important;
                text-align: left !important;
            }

            .stButton button[kind="primary"] {
                color: #000 !important;
                background: var(--green) !important;
                border: 1px solid var(--green) !important;
                border-radius: 7px !important;
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 12px !important;
                font-weight: 600 !important;
                letter-spacing: 0.05em !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
                line-height: 1.2 !important;
                min-height: 2.75rem !important;
                height: auto !important;
                text-align: left !important;
            }

            .stButton button[kind="primary"] *,
            .stButton button[kind="primary"] span,
            .stButton button[kind="primary"] div,
            .stButton button[kind="primary"] p {
                color: #000 !important;
                -webkit-text-fill-color: #000 !important;
            }

            [data-baseweb="tab-list"] button {
                color: var(--dim) !important;
                background: var(--surface) !important;
                border: 1px solid var(--border) !important;
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 11px !important;
            }

            [data-baseweb="tab-list"] button[aria-selected="true"] {
                color: var(--text) !important;
                background: #1b1f28 !important;
                border: 1px solid #2f3a52 !important;
                border-radius: 6px;
            }

            [data-testid="stAlert"] {
                border-radius: 8px;
                background: var(--surface) !important;
                border: 1px solid var(--border) !important;
            }

            [data-testid="stAlert"] * {
                color: var(--text) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=True)
def get_pipeline() -> PIPipeline:
    return PIPipeline()


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings()


def render_summary(summary: dict) -> None:
    verdict = summary["final_verdict"]
    decision_layer = summary["decision_layer"]
    confidence = float(summary["confidence_score"])
    total_ms = float(summary["total_processing_time_ms"])
    color_map = {
        "allow": "#22d47a",
        "flag": "#f5a623",
        "block": "#ff4f4f",
    }
    color = color_map.get(verdict, VERDICT_COLORS.get(verdict, "#dddddd"))

    st.markdown(
        (
            "<div class='verdict-card'>"
            "<div class='verdict-kicker'>// pipeline summary</div>"
            f"<h2 class='verdict-title' style='color:{color};'>Final verdict: {verdict.upper()}</h2>"
            "<div class='verdict-meta'>"
            f"<span class='verdict-chip'>Layer {decision_layer}</span>"
            f"<span class='verdict-chip'>Confidence {confidence:.2f}</span>"
            f"<span class='verdict-chip'>Total {total_ms:.1f} ms</span>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_layer_block(layer_name: str, summary: dict, status_map: dict) -> None:
    data = summary["layers"].get(layer_name)
    timing = summary["timings"].get(layer_name)
    status = status_map.get(layer_name, "skipped")

    st.markdown(f"### Layer {layer_name} ({status})")
    if timing is not None:
        st.caption(f"Processing time: {float(timing):.2f} ms")

    if data is None:
        st.info("Layer not executed due to early verdict.")
        return

    if layer_name == "A":
        flags = data.get("flags") or []
        col1, col2, col3 = st.columns(3)
        col1.metric("Suspicious", str(bool(data.get("suspicious"))))
        col2.metric("Flag count", len(flags))
        col3.metric("Confidence", f"{float(data.get('confidence_score', 0.0)):.2f}")
        if flags:
            st.write("Flags:", ", ".join(str(flag) for flag in flags))
        st.write("Decoded text preview:")
        st.code(str(data.get("processed_text", ""))[:600])
        return

    if layer_name == "B":
        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", str(data.get("verdict", "n/a")).upper())
        col2.metric("Attack similarity", f"{float(data.get('attack_similarity', 0.0)):.2f}")
        col3.metric("Benign similarity", f"{float(data.get('benign_similarity', 0.0)):.2f}")
        matches = data.get("matches") or []
        if matches:
            st.write("Signature matches")
            st.dataframe(matches, use_container_width=True)
        else:
            st.info("No signature matches.")
        return

    if layer_name in {"C", "D"}:
        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", str(data.get("verdict", "n/a")).upper())
        col2.metric("Risk probability", f"{float(data.get('probability_score', 0.0)):.2f}")
        col3.metric("Confidence", f"{float(data.get('confidence_score', 0.0)):.2f}")
        return

    if layer_name == "E":
        col1, col2 = st.columns(2)
        col1.metric("Decision", str(data.get("decision", "n/a")).upper())
        col2.metric("Model", str(data.get("model", "n/a")))
        st.write("Rationale")
        st.info(str(data.get("rationale", "No rationale returned.")))
        with st.expander("Raw Layer E response"):
            st.code(str(data.get("raw_response", ""))[:1200])


def run_protected(prompt: str) -> dict:
    pipeline = get_pipeline()
    return summarize_pipeline(pipeline.detect(prompt))


def run_baseline(prompt: str) -> dict:
    settings = get_settings()
    baseline = run_unprotected_baseline(prompt, settings)
    return {
        "output": baseline.output,
        "model": baseline.model,
        "latency_ms": baseline.latency_ms,
    }


def main() -> None:
    inject_styles()

    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = ""

    if "pending_prompt" in st.session_state:
        st.session_state["prompt_input"] = st.session_state.pop("pending_prompt")

    st.markdown(
        """
        <div class='gg-header'>
            <div>
                <div class='headline'>Hack the <span>AI</span></div>
                <div class='subhead'>// prompt injection shield - powered by barrikade</div>
            </div>
            <div class='status-pill'><span class='status-dot'></span>agent online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='input-panel'><div class='panel-label'>// goal input</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='panel-label' style='margin-top:10px;'>// try</div>", unsafe_allow_html=True
    )

    preset_cols = st.columns(5)
    for idx, (name, text) in enumerate(ATTACK_PRESETS.items()):
        if preset_cols[idx % 5].button(name, use_container_width=True):
            st.session_state["pending_prompt"] = text
            st.rerun()

    prompt = st.text_area(
        "Prompt input",
        key="prompt_input",
        height=220,
        placeholder="Enter a prompt to test Barrikade defense...",
    )

    actions = st.columns([1.4, 1.4, 1.4, 5.8])
    analyze_clicked = actions[0].button("RUN >", type="primary", use_container_width=True)
    base_clicked = actions[1].button("BASE", use_container_width=True)
    clear_clicked = actions[2].button("CLEAR", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if clear_clicked:
        st.session_state["prompt_input"] = ""
        st.session_state.pop("result", None)
        st.rerun()

    if analyze_clicked:
        if not prompt.strip():
            st.warning("Please enter a prompt before analysis.")
        else:
            with st.spinner("Running Barrikade pipeline..."):
                try:
                    protected = run_protected(prompt)
                    st.session_state["result"] = {
                        "prompt": prompt,
                        "protected": protected,
                        "baseline": None,
                    }
                except Exception as exc:
                    st.error(f"Pipeline execution failed: {exc}")

    if base_clicked:
        result = st.session_state.get("result")
        if result is None:
            st.warning("Run Analyze first, then Run Base.")
        else:
            with st.spinner("Running base LLM without Barrikade..."):
                try:
                    st.session_state["result"]["baseline"] = run_baseline(result["prompt"])
                except Exception as exc:
                    st.error(f"Baseline execution failed: {exc}")

    result = st.session_state.get("result")
    if result is None:
        st.info("Run to see the pipeline verdict and layer breakdown.")
        return

    protected = result["protected"]
    baseline = result["baseline"]

    st.markdown(
        "<div class='result-panel'><div class='panel-label'>// execution trace</div>",
        unsafe_allow_html=True,
    )
    render_summary(protected)

    st.markdown("<div class='trace-title'>// how it was detected</div>", unsafe_allow_html=True)
    for bullet in build_explanations(protected):
        st.markdown(f"<div class='trace-callout'>- {bullet}</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='trace-title'>// optional comparison: without security</div>",
        unsafe_allow_html=True,
    )
    if baseline is None:
        st.info("Baseline has not been run yet. Click Run Base to compare.")
    else:
        st.caption(f"Model: {baseline['model']} | Latency: {float(baseline['latency_ms']):.1f} ms")
        st.code(baseline["output"] or "[No response returned]")

    st.markdown("</div>", unsafe_allow_html=True)

    status_map = layer_statuses(protected)
    tabs = st.tabs(["Layer A", "Layer B", "Layer C", "Layer D", "Layer E", "Timing", "Raw JSON"])

    with tabs[0]:
        render_layer_block("A", protected, status_map)
    with tabs[1]:
        render_layer_block("B", protected, status_map)
    with tabs[2]:
        render_layer_block("C", protected, status_map)
    with tabs[3]:
        render_layer_block("D", protected, status_map)
    with tabs[4]:
        render_layer_block("E", protected, status_map)
    with tabs[5]:
        timings = protected["timings"]
        for layer_name in ["A", "B", "C", "D", "E"]:
            value = timings.get(layer_name)
            if value is None:
                st.write(f"Layer {layer_name}: skipped")
            else:
                st.write(f"Layer {layer_name}: {float(value):.2f} ms")
    with tabs[6]:
        st.json(
            {
                "prompt": result["prompt"],
                "protected": protected,
                "baseline": baseline,
            }
        )


if __name__ == "__main__":
    main()
