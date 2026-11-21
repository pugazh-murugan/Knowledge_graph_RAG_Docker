import streamlit as st

import api

status = api.get_status()

st.caption(
    "Reads a document from the backend's data folder, extracts entities and relationships "
    "with OpenAI, and replaces the saved graph. This uses your OpenAI quota."
)

with st.form("build_form"):
    pdf = st.text_input(
        "File name in the data folder",
        placeholder="Leave empty to use the default document",
        help="A .pdf or .txt file name, not a path.",
    )
    limit = st.number_input(
        "Limit chunks (0 = all)", min_value=0, max_value=1000, value=0,
        help="Process only the first N chunks. Handy for a quick, cheap test.",
    )
    submitted = st.form_submit_button(
        "Build graph", icon=":material/construction:", disabled=status["building"]
    )

if submitted:
    try:
        api.build(pdf.strip() or None, int(limit) or None)
    except api.ApiError as exc:
        st.error(str(exc), icon=":material/error:")
    else:
        st.rerun()


# Poll every 3 seconds only while a build is running
@st.fragment(run_every="3s" if status["building"] else None)
def build_status() -> None:
    current = api.get_status()
    if current["building"]:
        st.info("Build running. This can take a few minutes.", icon=":material/hourglass_top:")
    elif current["last_error"]:
        st.error(current["last_error"], title="Last build failed", icon=":material/error:")
    elif current["loaded"]:
        st.success(
            f"Graph ready: {current['nodes']} nodes, {current['edges']} edges.",
            icon=":material/check_circle:",
        )
    else:
        st.warning("No graph yet.", icon=":material/warning:")

    if status["building"] and not current["building"]:
        api.clear_cache()  # new graph: drop cached reads and refresh the whole page
        st.rerun()


build_status()
