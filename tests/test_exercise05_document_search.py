import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("GROQ_API_KEY", "test-key")
EXERCISE_05_DIR = Path(__file__).resolve().parents[1] / "Exercise_05"
sys.path.insert(0, str(EXERCISE_05_DIR))

agent = importlib.import_module("agent")


class Exercise05DocumentSearchTests(unittest.TestCase):
    def test_document_directory_is_resolved_from_repository(self) -> None:
        expected = Path(__file__).resolve().parents[1] / "Exercise_04" / "documents"
        self.assertEqual(agent.PROJECT_DOCS_DIR, expected)

    def test_search_returns_relevant_excerpt_and_filename(self) -> None:
        index = Mock()
        index.search.return_value = (
            np.array([[0.91]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        embedding_model = Mock()
        embedding_model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        chunks = [{
            "document_name": "python_intro.md",
            "chunk_index": 0,
            "content": "Python is a high-level interpreted programming language.",
        }]

        with patch.object(
            agent,
            "_get_project_document_index",
            return_value=(index, chunks, embedding_model, []),
        ):
            result = agent.search_project_documents("¿Qué es Python?")

        self.assertEqual(result["results_found"], 1)
        self.assertEqual(result["results"][0]["document"], "python_intro.md")
        self.assertIn("high-level interpreted", result["results"][0]["excerpt"])
        embedding_model.encode.assert_called_once_with(
            ["¿Qué es Python?"],
            convert_to_numpy=True,
        )

    def test_empty_query_does_not_load_documents(self) -> None:
        with patch.object(agent, "_get_project_document_index") as get_index:
            result = agent.search_project_documents("  ")

        get_index.assert_not_called()
        self.assertEqual(result["results_found"], 0)

    def test_search_tool_is_registered_for_the_agent(self) -> None:
        self.assertIs(
            agent.TOOL_FUNCTIONS["search_project_documents"],
            agent.search_project_documents,
        )
        schema = next(
            item["function"]
            for item in agent.TOOL_SCHEMAS
            if item["function"]["name"] == "search_project_documents"
        )
        self.assertEqual(schema["parameters"]["required"], ["query"])


if __name__ == "__main__":
    unittest.main()
