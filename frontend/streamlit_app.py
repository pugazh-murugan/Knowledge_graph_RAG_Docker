import streamlit as st

import api

st.set_page_config(page_title="Knowledge graph RAG", page_icon=":material/hub:", layout="wide")

# Shared state (initialised once, used by the pages)
if "messages" not in st.session_state:
    st.session_state.messages = []

pages = [
    st.Page(
        "app_pages/chat.py", title="Chat", icon=":material/chat:", default=True,
        help="Ask questions about your documents. Answers come only from facts in the knowledge graph.",
    ),
    st.Page(
        "app_pages/explorer.py", title="Explorer", icon=":material/account_tree:",
        help="Browse the knowledge graph, look up one entity, or find the shortest path between two.",
    ),
    st.Page(
        "app_pages/build.py", title="Build", icon=":material/construction:",
        help="Build or rebuild the knowledge graph from a document in the data folder.",
    ),
]
page = st.navigation(pages, position="top")

# Backend connection status (sidebar, shown on every page)
try:
    status = api.get_status()
except api.ApiError as exc:
    st.error(str(exc), title="Backend unavailable", icon=":material/cloud_off:")
    st.stop()

with st.sidebar:
    if status["loaded"]:
        st.caption(f":material/check_circle: Graph loaded: {status['nodes']} nodes, {status['edges']} edges")
    else:
        st.caption(":material/warning: No graph yet. Open the Build page.")

st.title(page.title, icon=page.icon)
page.run()
