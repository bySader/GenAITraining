import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("GROQ_API_KEY", "test-key")
EXERCISE_04_DIR = Path(__file__).resolve().parents[1] / "Exercise_04"
sys.path.insert(0, str(EXERCISE_04_DIR))

rag_engine = importlib.import_module("rag_engine")


class Exercise04RagTests(unittest.TestCase):
    def test_scope_guard_accepts_spanish_topics_with_or_without_accents(self) -> None:
        self.assertTrue(rag_engine.is_in_scope("¿Cómo funcionan los decoradores de Python?"))
        self.assertTrue(rag_engine.is_in_scope("Que es el aprendizaje automatico?"))
        self.assertTrue(rag_engine.is_in_scope("¿Cómo diseño una base de datos relacional?"))

    def test_scope_guard_does_not_match_keywords_inside_unrelated_words(self) -> None:
        self.assertFalse(rag_engine.is_in_scope("¿Cuál es la capital de Francia?"))

    def test_spanish_in_scope_question_reaches_retrieval(self) -> None:
        retrieved_chunk = {
            "document_name": "python_decorators.txt",
            "topic": "Programming",
            "author": "Carlos Rivera",
            "content": "A decorator extends a function without changing its source.",
        }
        mocked_client = Mock()
        mocked_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Un decorador extiende una función."))]
        )

        with (
            patch.object(
                rag_engine,
                "retrieve_and_rerank",
                return_value=[retrieved_chunk],
            ) as retrieve,
            patch.object(rag_engine, "client", mocked_client),
        ):
            result = rag_engine.ask_rag(
                "¿Qué hacen los decoradores de Python?",
                Mock(),
                [retrieved_chunk],
                Mock(),
            )

        retrieve.assert_called_once()
        self.assertTrue(result["in_scope"])
        self.assertEqual(result["sources"], ["python_decorators.txt"])

    def test_old_index_without_model_marker_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.faiss"
            metadata_path = Path(temporary_directory) / "metadata.pkl"
            model_path = Path(temporary_directory) / "embedding_model.txt"
            index_path.touch()
            metadata_path.touch()
            model = Mock()
            rebuilt_index = Mock()
            rebuilt_chunks = [{"content": "rebuilt"}]

            with (
                patch.object(rag_engine, "INDEX_PATH", index_path),
                patch.object(rag_engine, "METADATA_PATH", metadata_path),
                patch.object(rag_engine, "EMBEDDING_MODEL_PATH", model_path),
                patch.object(rag_engine, "SentenceTransformer", return_value=model),
                patch.object(
                    rag_engine,
                    "build_index",
                    return_value=(rebuilt_index, rebuilt_chunks),
                ) as build_index,
                patch.object(rag_engine, "save_index") as save_index,
            ):
                index, chunks, loaded_model = rag_engine.get_or_build_index()

            build_index.assert_called_once()
            save_index.assert_called_once_with(rebuilt_index, rebuilt_chunks)
            self.assertEqual(model_path.read_text(encoding="utf-8"), rag_engine.EMBED_MODEL)
            self.assertIs(index, rebuilt_index)
            self.assertEqual(chunks, rebuilt_chunks)
            self.assertIs(loaded_model, model)

    def test_main_receives_question_before_loading_index(self) -> None:
        events = []
        index = Mock()
        chunks = [{"document_name": "python_intro.md"}]
        model = Mock()
        questions = iter(["What is Python?", "quit"])

        def read_input(prompt: str) -> str:
            events.append("input")
            return next(questions)

        def load_index():
            events.append("load")
            return index, chunks, model

        def answer_question(question, loaded_index, loaded_chunks, loaded_model, verbose=False):
            events.append("answer")
            self.assertEqual(question, "What is Python?")
            self.assertIs(loaded_index, index)
            self.assertIs(loaded_chunks, chunks)
            self.assertIs(loaded_model, model)
            return {
                "answer": "Python is a programming language.",
                "sources": ["python_intro.md"],
                "in_scope": True,
            }

        with (
            patch("builtins.input", side_effect=read_input),
            patch.object(rag_engine, "get_or_build_index", side_effect=load_index),
            patch.object(rag_engine, "ask_rag", side_effect=answer_question),
        ):
            rag_engine.main()

        self.assertEqual(events, ["input", "load", "answer", "input"])

    def test_build_index_fails_explicitly_when_no_document_can_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Mock()

            with self.assertRaisesRegex(RuntimeError, "No se pudo cargar contenido"):
                rag_engine.build_index(Path(temporary_directory), model)

            model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
