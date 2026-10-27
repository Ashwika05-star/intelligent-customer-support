import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.api_key and self.api_key.strip():
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key.strip())
            except Exception as e:
                logger.warning(f"Could not initialize Google GenAI client: {e}")
                self.client = None
        else:
            self.client = None

    def is_configured(self) -> bool:
        return self.client is not None

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2
    ) -> str:
        """Call Gemini model with system instruction and temperature."""
        if not self.is_configured():
            return ""

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction
            )
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            # Try fallback model
            try:
                fallback_model = "gemini-1.5-flash"
                if model != fallback_model:
                    response = self.client.models.generate_content(
                        model=fallback_model,
                        contents=prompt
                    )
                    return response.text.strip() if response.text else ""
            except Exception as fallback_err:
                logger.error(f"Fallback generation error: {fallback_err}")
            return ""

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: str = "gemini-2.5-flash"
    ) -> Optional[Dict[str, Any]]:
        """Call Gemini expecting a JSON response."""
        json_prompt = (
            f"{prompt}\n\nIMPORTANT: Respond with ONLY a valid, parseable JSON object. "
            "Do NOT include markdown formatting or backticks around the json."
        )
        raw = self.generate(json_prompt, system_instruction=system_instruction, model=model, temperature=0.1)
        if not raw:
            return None

        # Clean potential markdown wrapping ```json ... ```
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        try:
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"Failed to parse JSON from LLM response: {clean[:100]}... Error: {e}")
            return None

# Global service instance
_llm_instance: Optional[GeminiService] = None

def get_llm_service(api_key: Optional[str] = None) -> GeminiService:
    global _llm_instance
    if _llm_instance is None or (api_key and api_key != _llm_instance.api_key):
        _llm_instance = GeminiService(api_key=api_key)
    return _llm_instance
