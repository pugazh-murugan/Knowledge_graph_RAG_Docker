import streamlit as st

import api

SUGGESTIONS = [
    "What does Redis do in this architecture?",
    "Which databases are used?",
    "How does the API Gateway fit in?",
]

with st.sidebar:
    depth = st.slider("Graph depth", 1, 3, 1, help="How many hops of connections to include around matched entities.")
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()


def show_details(result: dict) -> None:
    with st.expander("Retrieved context"):
        st.markdown("**Linked by name:** " + (", ".join(result["linked"]) or "none"))
        st.markdown("**Found by similarity:** " + (", ".join(result["semantic"]) or "none"))
        st.markdown("**Facts used**")
        st.code("\n".join(result["facts"]), language=None, wrap_lines=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("details"):
            show_details(msg["details"])

suggestion = None
if not st.session_state.messages:
    suggestion = st.pills("Try asking", SUGGESTIONS, label_visibility="collapsed", key="suggestion")

prompt = st.chat_input("Ask a question about the documents", submit_mode="disable") or suggestion

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the graph..."):
                result = api.ask(prompt, depth)
        except api.ApiError as exc:
            st.session_state.messages.pop()  # drop the unanswered question
            st.error(str(exc), icon=":material/error:")
        else:
            st.write(result["answer"])
            show_details(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "details": result}
            )
