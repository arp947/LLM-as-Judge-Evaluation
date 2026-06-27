import json
import re
import traceback
from groq import Groq
from config import judge_settings
from schema import StructuredJudgeVerdict

class LLMJudgeEngine:
    def __init__(self):
        self.client = Groq(api_key=judge_settings.GROQ_API_KEY)
        self.call_count = 0
        self.total_tokens_consumed = 0

    # All key variants the model has been observed to use for the winner field
    _WINNER_KEYS = ["overall_winner", "recommended_output", "winner", "recommended_model"]

    def _extract_winner(self, data: dict) -> str:
        """Try all known winner key variants; normalize free-text values to A/B/TIE."""
        for key in self._WINNER_KEYS:
            val = data.get(key)
            if not val or not isinstance(val, str):
                continue
            v = val.strip().upper()
            if v in ("A", "MODEL OUTPUT A", "MODEL_OUTPUT_A"):
                return "A"
            if v in ("B", "MODEL OUTPUT B", "MODEL_OUTPUT_B"):
                return "B"
            if v in ("TIE", "BOTH", "NEITHER", "NULL", "NONE"):
                return "TIE"
        # Fallback: tally scores from per_criterion_eval if present
        criteria = data.get("per_criterion_eval", [])
        if isinstance(criteria, list) and criteria:
            total_a = sum(c.get("score_a", 0) for c in criteria if isinstance(c, dict))
            total_b = sum(c.get("score_b", 0) for c in criteria if isinstance(c, dict))
            if total_a > total_b:
                return "A"
            if total_b > total_a:
                return "B"
        return "TIE"

    def _robust_json_parse(self, raw_text: str) -> dict:
        """Parse JSON and normalise to always contain 'overall_winner'."""
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            winner_match = re.search(r'"(?:overall_winner|winner|recommended_output)"\s*:\s*"([^"]+)"', cleaned)
            winner_raw = winner_match.group(1) if winner_match else "TIE"
            data = {
                "per_criterion_eval": [],
                "overall_winner": winner_raw,
                "global_rationale": f"JSON parse failed. Raw: {cleaned[:120]}"
            }
        # Always normalise to a canonical overall_winner key
        data["overall_winner"] = self._extract_winner(data)
        return data

    def build_system_prompt(self) -> str:
        return """You are an unbiased LLM judge. Compare Model Output A vs Model Output B against the Expected Reference.

RUBRIC:
1. Correctness: Facts must match the Expected Reference. Penalize wrong facts.
2. Completeness: All parts of the question answered.
3. Faithfulness: No hallucinated details.
4. Length Calibration: Penalize verbosity that adds no information. Prefer concise correct answers.

You MUST respond with ONLY this exact JSON structure, no other keys:
{
  "per_criterion_eval": [
    {"criterion": "Correctness", "score_a": <0-1>, "score_b": <0-1>, "rationale": "..."},
    {"criterion": "Completeness", "score_a": <0-1>, "score_b": <0-1>, "rationale": "..."},
    {"criterion": "Faithfulness", "score_a": <0-1>, "score_b": <0-1>, "rationale": "..."},
    {"criterion": "Length Calibration", "score_a": <0-1>, "score_b": <0-1>, "rationale": "..."}
  ],
  "overall_winner": "A" or "B" or "TIE",
  "global_rationale": "one sentence explanation"
}

Do NOT use any other key names. overall_winner must be exactly "A", "B", or "TIE"."""

    def execute_single_judgment(self, test_case: dict, model_a_text: str, model_b_text: str) -> dict:
        user_content = f"""User Input: {test_case['input']}
Expected Reference Output: {test_case.get('expected_output', 'N/A')}

[START MODEL OUTPUT A]
{model_a_text}
[END MODEL OUTPUT A]

[START MODEL OUTPUT B]
{model_b_text}
[END MODEL OUTPUT B]

Evaluate both outputs meticulously according to the core criteria."""

        self.call_count += 1
        try:
            response = self.client.chat.completions.create(
                model=judge_settings.JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": user_content}
                ],
                temperature=judge_settings.TEMPERATURE,
                response_format={"type": "json_object"}
            )
            self.total_tokens_consumed += response.usage.total_tokens
            return self._robust_json_parse(response.choices[0].message.content)  # fixed: choices[0]
        except Exception as e:
            print(f"  [ERROR] case '{test_case.get('id')}': {e}")
            traceback.print_exc()
            return {"per_criterion_eval": [], "overall_winner": "TIE", "global_rationale": f"API error: {str(e)}"}

    def evaluate_with_position_swap(self, test_case: dict) -> dict:
        """
        Mitigates Position Bias:
        Evaluates the pair twice by swapping their presentation orders (A/B then B/A).
        """
        output_1 = test_case["model_output_v1"]
        output_2 = test_case["model_output_v2"]

        # Run 1: Normal Order
        verdict_1 = self.execute_single_judgment(test_case, output_1, output_2)
        
        # Run 2: Swapped Order
        verdict_2 = self.execute_single_judgment(test_case, output_2, output_1)

        # Normalize the swapped run's perspectives back to baseline
        v2_winner = verdict_2.get("overall_winner", "TIE")
        normalized_v2_winner = "TIE"
        if v2_winner == "A":
            normalized_v2_winner = "B"
        elif v2_winner == "B":
            normalized_v2_winner = "A"

        # Determine alignment and measure if a position flip occurred
        is_consistent = (verdict_1.get("overall_winner") == normalized_v2_winner)
        
        final_winner = verdict_1.get("overall_winner") if is_consistent else "TIE"

        return {
            "run_1_verdict": verdict_1,
            "run_2_verdict": verdict_2,
            "is_consistent": is_consistent,
            "final_winner": final_winner
        }

judge_engine = LLMJudgeEngine()
