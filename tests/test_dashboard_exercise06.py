import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import TextIO
from unittest.mock import Mock, patch

import app as dashboard


class DashboardExercise06Tests(unittest.TestCase):
    def setUp(self) -> None:
        dashboard.app.config.update(TESTING=True)
        self.client = dashboard.app.test_client()

    def test_dashboard_renders_exercises_and_search_controls(self) -> None:
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="exerciseSearch"', html)
        self.assertIn('id="exerciseList"', html)
        self.assertIn('data-id="exercise_06"', html)
        self.assertIn("addEventListener('input', renderExerciseList)", html)
        self.assertIn("selectExercise(button.dataset.id)", html)
        self.assertIn("restoreConversationHistory(exercise.id)", html)
        self.assertIn(
            r"legacyOutput.match(/(?:^|\n)SQL:\s*([\s\S]*?)(?=\nAnswer:\s*\n)/)",
            html,
        )
        self.assertIn(
            r"legacyOutput.match(/\nAnswer:\s*\n([\s\S]*?)(?=\n-{5,}\s*(?:\n|$)|\nEstado:|\n\nMenu:|$)/)",
            html,
        )
        self.assertIn("/api/conversations/", html)

    def test_exercise_api_includes_exercise06(self) -> None:
        response = self.client.get("/api/exercises")

        self.assertEqual(response.status_code, 200)
        exercises = response.get_json()["exercises"]
        exercise = next(item for item in exercises if item["id"] == "exercise_06")
        self.assertEqual(exercise["main_script"], "workflow.py")

    def test_conversation_history_can_be_saved_loaded_and_cleared(self) -> None:
        owner_id = "11111111-1111-4111-8111-111111111111"
        other_owner_id = "22222222-2222-4222-8222-222222222222"
        messages = [
            {
                "id": "message-1",
                "role": "user",
                "content": "¿Cuántos tickets están abiertos?",
                "createdAt": "2026-09-29T12:00:00.000Z",
                "scriptName": "",
                "sqlQuery": "",
                "status": "completed",
            },
            {
                "id": "message-2",
                "role": "assistant",
                "content": "Hay tres tickets abiertos.",
                "createdAt": "2026-09-29T12:00:02.000Z",
                "scriptName": "workflow.py",
                "sqlQuery": "SELECT COUNT(*) FROM tickets WHERE status = 'Open';",
                "status": "completed",
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(
                dashboard,
                "CONVERSATION_DB_PATH",
                Path(temporary_directory) / "instance" / "history.sqlite3",
            ):
                saved = self.client.put(
                    f"/api/conversations/{owner_id}/exercise_06",
                    json={"messages": messages},
                )
                loaded = self.client.get(
                    f"/api/conversations/{owner_id}/exercise_06"
                )
                isolated = self.client.get(
                    f"/api/conversations/{other_owner_id}/exercise_06"
                )
                cleared = self.client.delete(
                    f"/api/conversations/{owner_id}/exercise_06"
                )
                after_clear = self.client.get(
                    f"/api/conversations/{owner_id}/exercise_06"
                )

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(loaded.get_json()["messages"], messages)
        self.assertEqual(isolated.get_json()["messages"], [])
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(after_clear.get_json()["messages"], [])

    def test_conversation_history_rejects_invalid_owner_and_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(
                dashboard,
                "CONVERSATION_DB_PATH",
                Path(temporary_directory) / "history.sqlite3",
            ):
                invalid_owner = self.client.get(
                    "/api/conversations/not-a-uuid/exercise_06"
                )
                invalid_message = self.client.put(
                    "/api/conversations/11111111-1111-4111-8111-111111111111/exercise_06",
                    json={"messages": [{"id": "bad", "role": [], "content": "x"}]},
                )

        self.assertEqual(invalid_owner.status_code, 400)
        self.assertEqual(invalid_message.status_code, 400)

    def test_workflow_rejects_empty_question(self) -> None:
        response = self.client.post(
            "/api/run",
            json={"exerciseId": "exercise_06", "scriptName": "workflow.py", "stdinText": "  "},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("pregunta", response.get_json()["error"].lower())

    def test_workflow_reports_prerequisite_error_without_starting_process(self) -> None:
        with (
            patch.object(dashboard, "exercise06_prerequisite_error", return_value="Falta configuración."),
            patch.object(dashboard.subprocess, "Popen") as popen,
        ):
            response = self.client.post(
                "/api/run",
                json={
                    "exerciseId": "exercise_06",
                    "scriptName": "workflow.py",
                    "stdinText": "¿Cuántos tickets están abiertos?",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "Falta configuración.")
        popen.assert_not_called()

    def test_workflow_reports_missing_api_key(self) -> None:
        with (
            patch.object(dashboard, "GLOBAL_GROQ_API_KEY", ""),
            patch.dict(os.environ, {}, clear=True),
            patch.object(dashboard.subprocess, "Popen") as popen,
        ):
            response = self.client.post(
                "/api/run",
                json={
                    "exerciseId": "exercise_06",
                    "scriptName": "workflow.py",
                    "stdinText": "¿Cuántos tickets existen?",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("GROQ_API_KEY", response.get_json()["error"])
        popen.assert_not_called()

    def test_database_prerequisite_check_is_read_only_and_accepts_expected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            exercise_directory = Path(temporary_directory) / "Exercise_06"
            exercise_directory.mkdir()
            database_path = exercise_directory / "company.db"
            connection = sqlite3.connect(database_path)
            try:
                for table, columns in dashboard.EXERCISE_06_REQUIRED_COLUMNS.items():
                    column_definitions = ", ".join(f'"{column}" TEXT' for column in columns)
                    connection.execute(f'CREATE TABLE "{table}" ({column_definitions})')
            finally:
                connection.close()

            with (
                patch.object(dashboard, "ROOT", Path(temporary_directory)),
                patch.object(dashboard, "GLOBAL_GROQ_API_KEY", "configured"),
                patch.object(dashboard.importlib.util, "find_spec", return_value=object()),
            ):
                error = dashboard.exercise06_prerequisite_error()

            self.assertIsNone(error)
            connection = sqlite3.connect(database_path)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()
            self.assertEqual(tables, set(dashboard.EXERCISE_06_REQUIRED_COLUMNS))

    def test_process_start_failure_returns_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(dashboard, "LOG_DIR", Path(temporary_directory)),
                patch.object(dashboard, "exercise06_prerequisite_error", return_value=None),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
                patch.object(
                    dashboard.subprocess,
                    "Popen",
                    side_effect=FileNotFoundError("interpreter unavailable"),
                ),
            ):
                response = self.client.post(
                    "/api/run",
                    json={
                        "exerciseId": "exercise_06",
                        "scriptName": "workflow.py",
                        "stdinText": "¿Cuántos tickets existen?",
                    },
                )

        self.assertEqual(response.status_code, 500)
        self.assertIn("No se pudo iniciar", response.get_json()["error"])

    def test_dashboard_process_returns_structured_workflow_output(self) -> None:
        structured_output = json.dumps({
            "sql_query": "SELECT COUNT(*) FROM tickets;",
            "final_response": "Hay 12 tickets.",
            "error": "",
        })
        fake_process = Mock()
        fake_process.pid = 43210
        fake_process.stdin = Mock()
        fake_process.poll.return_value = 0
        fake_process.returncode = 0

        def start_process(*, stdout: TextIO, **_: object) -> Mock:
            stdout.write(structured_output)
            return fake_process

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(dashboard, "LOG_DIR", Path(temporary_directory)),
                patch.object(dashboard, "PROCESSES", {}),
                patch.object(dashboard, "exercise06_prerequisite_error", return_value=None),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
                patch.object(dashboard.subprocess, "Popen", side_effect=start_process) as popen,
            ):
                started = self.client.post(
                    "/api/run",
                    json={
                        "exerciseId": "exercise_06",
                        "scriptName": "workflow.py",
                        "stdinText": "¿Cuántos tickets existen?",
                    },
                )
                self.assertEqual(started.status_code, 200)
                fake_process.stdin.write.assert_called_once_with("¿Cuántos tickets existen?")
                command = popen.call_args.args[0]
                self.assertEqual(command[-2:], ["workflow.py", "--dashboard"])

                completed = self.client.get("/api/process/43210")

        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.get_json()["status"], "completed")
        self.assertEqual(json.loads(completed.get_json()["output"]), {
            "sql_query": "SELECT COUNT(*) FROM tickets;",
            "final_response": "Hay 12 tickets.",
            "error": "",
        })

    def test_missing_process_returns_not_found(self) -> None:
        with patch.object(dashboard, "PROCESSES", {}):
            response = self.client.get("/api/process/12345")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
