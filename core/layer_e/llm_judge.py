from litellm import completion
from .utils import LLM_BASE_URL, TEMPERATURE, JudgeOutput, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

class LLMJudge:
    def __init__(self, llm_base_url=LLM_BASE_URL, temperature=TEMPERATURE):
        self.llm_base_url = llm_base_url
        self.temperature = temperature

    def call_judge(self, prompt, max_retries=2):
        for attempt in range(max_retries):
            try:
                response = completion(
                    model="ollama/gpt-oss:latest",
                    temperature=self.temperature,
                    api_base=self.llm_base_url,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"""{FEW_SHOT_EXAMPLES}

                            NOW CLASSIFY THIS INPUT:
                            <input_text>
                            {prompt}
                            </input_text>

                            Respond with ONLY valid JSON in this exact format:
                            {{"label": 0 or 1, "rationale": "brief explanation"}}

                            Remember: label 0 = benign, label 1 = malicious. When uncertain, choose 1 (malicious)."""
                        }
                    ]
                )
                
                content = response.choices[0].message.content # type: ignore
                
                # If the LLM refuses to analyze or says it can't help, the input is likely malicious
                refusal_phrases = ["can't help", "cannot help", "sorry", "unable to", "i can't", "i cannot", "not able to"]
                if any(phrase in content.lower() for phrase in refusal_phrases): # type: ignore
                    return JudgeOutput(label=1, rationale="LLM refused to analyze - treating as malicious")
                
                # Try to extract JSON from the response (sometimes wrapped in markdown)
                import re
                json_match = re.search(r'\{[^{}]*"label"\s*:\s*[01][^{}]*\}', content) # type: ignore
                if json_match:
                    content = json_match.group(0)
                
                result = JudgeOutput.model_validate_json(content) # type: ignore
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                print(f"Error parsing response after {max_retries} attempts: {e}")
                # Default to malicious when parsing fails - safer approach
                return JudgeOutput(label=1, rationale=f"Parsing failed, defaulting to malicious: {str(e)[:50]}")