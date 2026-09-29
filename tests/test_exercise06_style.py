import unittest
from unittest.mock import Mock, patch

from Exercise_06 import workflow


class Exercise06AssistantStyleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state: workflow.AgentState = {
            "question": "¿Cuántos tickets están abiertos?",
            "route": "",
            "sql_query": "SELECT ticket_id FROM tickets WHERE status = 'Open';",
            "query_result": '[{"ticket_id": "T-1"}]',
            "final_response": "",
            "error": "",
        }

    def test_formatter_uses_shared_style_and_keeps_grounding_rules(self) -> None:
        model = Mock(
            invoke=Mock(return_value=Mock(content="Hay un ticket abierto."))
        )
        with patch.object(workflow, "llm", model):
            result = workflow.formatter_node(self.state)

        system_prompt = model.invoke.call_args.args[0][0].content
        self.assertIn("cercano, claro y respetuoso", system_prompt)
        self.assertIn("Base your answer ONLY on the data provided", system_prompt)
        self.assertEqual(result["final_response"], "Hay un ticket abierto.")

    def test_direct_answer_uses_shared_style(self) -> None:
        model = Mock(
            invoke=Mock(return_value=Mock(content="LangGraph conecta pasos de un flujo."))
        )
        with patch.object(workflow, "llm", model):
            result = workflow.direct_answer_node({**self.state, "question": "¿Qué es LangGraph?"})

        system_prompt = model.invoke.call_args.args[0][0].content
        self.assertIn("cercano, claro y respetuoso", system_prompt)
        self.assertEqual(result["final_response"], "LangGraph conecta pasos de un flujo.")

    def test_rag_uses_shared_style_and_returns_source(self) -> None:
        model = Mock(
            invoke=Mock(return_value=Mock(content="Configura la VPN desde el portal de TI."))
        )
        with patch.object(workflow, "llm", model):
            result = workflow.rag_node({
                **self.state,
                "question": "¿Cómo configuro la VPN? Muéstrame el setup guide.",
            })

        system_prompt = model.invoke.call_args.args[0][0].content
        self.assertIn("cercano, claro y respetuoso", system_prompt)
        self.assertIn("using ONLY the documentation excerpts", system_prompt)
        self.assertIn("KB-002", result["final_response"])

    def test_empty_and_error_results_use_warmer_messages(self) -> None:
        empty_result = workflow.formatter_node({**self.state, "query_result": "[]"})
        failed_result = workflow.formatter_node({**self.state, "error": "database unavailable"})
        no_article = workflow.rag_node({**self.state, "question": "pregunta sin coincidencias"})

        self.assertIn("¿Quieres probar con otros términos?", empty_result["final_response"])
        self.assertIn("Puedes reformular la pregunta", failed_result["final_response"])
        self.assertIn("¿Quieres probar con otros términos o temas?", no_article["final_response"])


if __name__ == "__main__":
    unittest.main()
