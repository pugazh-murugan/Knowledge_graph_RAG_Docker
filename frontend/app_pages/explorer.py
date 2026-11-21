import pandas as pd
import streamlit as st

import api

try:
    stats = api.get_stats()
except api.ApiError as exc:
    st.info(str(exc), title="No graph to explore", icon=":material/info:")
    st.stop()

with st.container(horizontal=True):
    st.metric("Nodes", stats["nodes"], border=True)
    st.metric("Edges", stats["edges"], border=True)
    st.metric("Isolated nodes", stats["isolated_nodes"], border=True)

view = st.segmented_control("View", ["Graph", "Entity", "Path"], default="Graph", label_visibility="collapsed")


# ----------------------------------------------------------------------
if view == "Graph":
    graph = api.get_graph()
    types = sorted({n["type"] for n in graph["nodes"]})

    with st.container(horizontal=True, vertical_alignment="bottom"):
        selected = st.multiselect("Entity types", types, default=types)
        hide_isolated = st.toggle("Hide isolated nodes", value=True)

    nodes = [
        n for n in graph["nodes"]
        if n["type"] in selected and not (hide_isolated and n["degree"] == 0)
    ]
    keep = {n["id"] for n in nodes}
    links = [e for e in graph["edges"] if e["source"] in keep and e["target"] in keep]
    category_index = {t: i for i, t in enumerate(types)}

    option = {
        "tooltip": {"formatter": "{b}: {c}"},
        "legend": {"data": types},
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "categories": [{"name": t} for t in types],
            "data": [
                {
                    "id": n["id"],
                    "name": n["name"],
                    "value": n["description"],
                    "category": category_index[n["type"]],
                    "symbolSize": 8 + 4 * min(n["degree"], 8),
                    "label": {"show": n["degree"] >= 4},
                }
                for n in nodes
            ],
            "links": [
                {"source": e["source"], "target": e["target"], "value": e["relation"]}
                for e in links
            ],
            "edgeSymbol": ["none", "arrow"],
            "edgeSymbolSize": 6,
            "lineStyle": {"curveness": 0.1, "opacity": 0.6},
            "force": {"repulsion": 180, "edgeLength": 70},
            "emphasis": {"focus": "adjacency"},
        }],
    }
    st.echarts_chart(
        option,
        height=620,
        key="kg_graph",
        alt=f"Knowledge graph with {len(nodes)} entities and {len(links)} relationships, coloured by entity type",
    )
    st.caption("Drag nodes, scroll to zoom, hover for descriptions. Node size shows how connected an entity is.")

    st.subheader("Most connected entities")
    st.dataframe(
        pd.DataFrame(stats["top_connected"]).rename(
            columns={"name": "Entity", "type": "Type", "connections": "Connections"}
        ),
        hide_index=True,
        alt="Top five most connected entities in the graph",
    )


# ----------------------------------------------------------------------
elif view == "Entity":
    names = [e["name"] for e in api.get_entities()]
    name = st.selectbox("Entity", names, index=None, placeholder="Choose an entity")
    if name:
        try:
            entity = api.get_entity(name)
        except api.ApiError as exc:
            st.error(str(exc))
        else:
            st.subheader(entity["name"])
            st.badge(entity["type"])
            st.write(entity["description"] or "No description.")

            out_col, in_col = st.columns(2)
            with out_col:
                st.markdown(f"**Outgoing** ({len(entity['outgoing'])})")
                if entity["outgoing"]:
                    st.dataframe(
                        pd.DataFrame(entity["outgoing"]).rename(columns={"relation": "Relation", "target": "Target"}),
                        hide_index=True, alt=f"Relationships going out from {entity['name']}",
                    )
                else:
                    st.caption("None")
            with in_col:
                st.markdown(f"**Incoming** ({len(entity['incoming'])})")
                if entity["incoming"]:
                    st.dataframe(
                        pd.DataFrame(entity["incoming"]).rename(columns={"source": "Source", "relation": "Relation"}),
                        hide_index=True, alt=f"Relationships coming into {entity['name']}",
                    )
                else:
                    st.caption("None")


# ----------------------------------------------------------------------
else:
    names = [e["name"] for e in api.get_entities()]
    left, right = st.columns(2)
    source = left.selectbox("From", names, index=None, placeholder="Choose an entity")
    target = right.selectbox("To", names, index=None, placeholder="Choose an entity")
    if source and target:
        try:
            result = api.get_path(source, target)
        except api.ApiError as exc:
            if exc.status_code == 404:
                st.info(f"No connection found between {source} and {target}.", icon=":material/link_off:")
            else:
                st.error(str(exc))
        else:
            st.success(f"Connected in {result['length']} step(s)", icon=":material/link:")
            st.markdown(" → ".join(f"**{n}**" for n in result["path"]))
