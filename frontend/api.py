"""Thin client for the FastAPI backend."""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001").rstrip("/")


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _call(method: str, path: str, timeout: float = 30, **kwargs):
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the backend at {BACKEND_URL}.") from exc
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(str(detail), resp.status_code)
    return resp.json()


# --- cheap, cacheable reads ---------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_graph() -> dict:
    return _call("GET", "/graph", timeout=60)


@st.cache_data(ttl=60, show_spinner=False)
def get_stats() -> dict:
    return _call("GET", "/graph/stats")


@st.cache_data(ttl=60, show_spinner=False)
def get_entities() -> list[dict]:
    return _call("GET", "/entities")


def clear_cache() -> None:
    st.cache_data.clear()


# --- uncached calls ------------------------------------------------------
def get_status() -> dict:
    return _call("GET", "/graph/status", timeout=10)


def get_entity(name: str) -> dict:
    return _call("GET", "/entity", params={"name": name})


def get_path(source: str, target: str) -> dict:
    return _call("GET", "/path", params={"source": source, "target": target})


def ask(question: str, depth: int) -> dict:
    return _call("POST", "/ask", timeout=120, json={"question": question, "depth": depth})


def build(pdf: str | None, limit: int | None) -> dict:
    return _call("POST", "/graph/build", json={"pdf": pdf, "limit": limit})
