"""Interactive HTML rendering of the full network (06.04 section 4)."""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx

log = logging.getLogger(__name__)

CATEGORY_COLORS = {"Variant": "#00b0f0", "Treatment": "#32cd32", "Cancer": "#ff4c4c"}


def write_network_html(G: nx.Graph, path: Path, min_component_size: int = 50) -> None:
    import plotly.graph_objs as go

    components = [c for c in nx.connected_components(G) if len(c) >= min_component_size]
    H = G.subgraph(set().union(*components)).copy() if components else G.copy()
    log.info("Computing layout for %d nodes, %d edges ...", H.number_of_nodes(), H.number_of_edges())
    pos = nx.spring_layout(H, seed=42, k=0.15, iterations=50)

    degrees = dict(H.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_x, node_y, text, colors, sizes = [], [], [], [], []
    for node in H.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        colors.append(CATEGORY_COLORS.get(H.nodes[node].get("category"), "#888888"))
        text.append(f"{node}<br>Degree: {degrees[node]}")
        sizes.append(5 + degrees[node] / max_degree * 20)

    edge_x, edge_y = [], []
    for u, v in H.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])

    legend = [
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=color), name=category)
        for category, color in CATEGORY_COLORS.items()
    ] + [go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0.01, color="rgba(0,0,0,0)"),
                    name="Size = Node degree")]

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", showlegend=False,
                       line=dict(width=0.2, color="rgba(200,200,200,0.15)")),
            go.Scatter(x=node_x, y=node_y, mode="markers", text=text, hoverinfo="text", showlegend=False, name="",
                       marker=dict(size=sizes, color=colors, opacity=0.85, line=dict(width=0.3, color="white"))),
            *legend,
        ],
        layout=go.Layout(
            title=dict(text="Variantscape: Full network graph of molecular variants, treatments and cancer types",
                       font=dict(size=20, color="white"), x=0.5),
            showlegend=True,
            legend=dict(font=dict(color="white"), title=dict(text="Legend", font=dict(size=14, color="white")),
                        bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="black",
            paper_bgcolor="black",
        ),
    )
    fig.write_html(str(path))
