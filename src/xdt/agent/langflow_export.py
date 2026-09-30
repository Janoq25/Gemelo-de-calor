"""Exportador y generador de flujos para LangFlow (RAG Asistente Documental)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_rag_flow_definition() -> dict[str, Any]:
    """Genera la estructura de grafo JSON estándar compatible con LangFlow v1.0+."""
    prompt_str = (
        "Eres el Asistente Experto del Gemelo Digital de Calor (xai-dt-heat).\n"
        "Responde las dudas del investigador basándote en el contexto.\n\n"
        "Reglas epidemiológicas y metodológicas:\n"
        "1. Nunca afirmes causalidad determinista a partir de los escenarios.\n"
        "2. Fusión en Dos Escalas (A: temporal estado-día, B: espacial transversal).\n"
        "3. Cita las secciones del plan cuando sea relevante.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}\nRespuesta:"
    )

    return {
        "name": "XAI-DT-Heat Knowledge Assistant",
        "description": "Asistente RAG para consultas sobre metodología y datos del Gemelo Digital.",
        "data": {
            "nodes": [
                {
                    "id": "file_loader_1",
                    "type": "File",
                    "data": {
                        "type": "File",
                        "node": {
                            "template": {
                                "path": {
                                    "value": "plan_gemelo_digital_calor.md",
                                    "type": "str",
                                    "display_name": "Ruta del Documento Maestro",
                                }
                            },
                            "description": "Carga el plan de investigación y especificación.",
                        },
                    },
                    "position": {"x": 100, "y": 100},
                },
                {
                    "id": "text_splitter_1",
                    "type": "RecursiveCharacterTextSplitter",
                    "data": {
                        "type": "RecursiveCharacterTextSplitter",
                        "node": {
                            "template": {
                                "chunk_size": {"value": 1000, "type": "int"},
                                "chunk_overlap": {"value": 150, "type": "int"},
                                "separators": {
                                    "value": ["\n## ", "\n### ", "\n\n", "\n", " "],
                                    "type": "list",
                                },
                            },
                            "description": "Divide respetando la jerarquía de secciones.",
                        },
                    },
                    "position": {"x": 350, "y": 100},
                },
                {
                    "id": "embeddings_1",
                    "type": "OpenAIEmbeddings",
                    "data": {
                        "type": "OpenAIEmbeddings",
                        "node": {
                            "template": {
                                "model": {"value": "text-embedding-3-small", "type": "str"},
                            },
                            "description": "Genera embeddings semánticos para búsqueda vectorial.",
                        },
                    },
                    "position": {"x": 350, "y": 300},
                },
                {
                    "id": "vector_store_1",
                    "type": "Chroma",
                    "data": {
                        "type": "Chroma",
                        "node": {
                            "template": {
                                "collection_name": {"value": "xai_dt_heat_docs", "type": "str"},
                            },
                            "description": "Base vectorial en memoria / Chroma.",
                        },
                    },
                    "position": {"x": 650, "y": 200},
                },
                {
                    "id": "prompt_template_1",
                    "type": "Prompt",
                    "data": {
                        "type": "Prompt",
                        "node": {
                            "template": {
                                "template": {
                                    "value": prompt_str,
                                    "type": "str",
                                },
                            },
                            "description": "Plantilla del sistema con directrices epidemiológicas.",
                        },
                    },
                    "position": {"x": 900, "y": 100},
                },
                {
                    "id": "llm_1",
                    "type": "ChatOpenAI",
                    "data": {
                        "type": "ChatOpenAI",
                        "node": {
                            "template": {
                                "model_name": {"value": "gpt-4o-mini", "type": "str"},
                                "temperature": {"value": 0.1, "type": "float"},
                            },
                            "description": "Modelo de lenguaje para síntesis precisa.",
                        },
                    },
                    "position": {"x": 900, "y": 300},
                },
                {
                    "id": "chat_output_1",
                    "type": "ChatOutput",
                    "data": {
                        "type": "ChatOutput",
                        "node": {
                            "description": "Canal de salida y chat para el usuario o consola.",
                        },
                    },
                    "position": {"x": 1200, "y": 200},
                },
            ],
            "edges": [
                {"source": "file_loader_1", "target": "text_splitter_1"},
                {"source": "text_splitter_1", "target": "vector_store_1"},
                {"source": "embeddings_1", "target": "vector_store_1"},
                {"source": "vector_store_1", "target": "prompt_template_1"},
                {"source": "prompt_template_1", "target": "llm_1"},
                {"source": "llm_1", "target": "chat_output_1"},
            ],
        },
    }


def export_langflow_json(output_path: str | Path = "langflow/gemelo_rag_flow.json") -> Path:
    """Exporta el archivo de flujo de LangFlow a disco."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flow_data = get_rag_flow_definition()
    out.write_text(json.dumps(flow_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
