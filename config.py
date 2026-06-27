from pydantic_settings import BaseSettings, SettingsConfigDict

class JudgeSettings(BaseSettings):
    # Cross-model evaluation strategy: Avoid self-enhancement by keeping judge separate from generator family
    # Generator uses llama-3.1-8b-instant; Judge uses a different family (gemma2) to prevent self-enhancement bias
    GENERATOR_A_MODEL: str = "llama-3.1-8b-instant"   # Small/fast generator A
    GENERATOR_B_MODEL: str = "llama-3.1-70b-versatile"  # Larger generator B
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"  # Current Groq model; supports json_object mode; stronger judge than 8b generators

    GROQ_API_KEY: str = "YOUR_GROQ_API_KEY_HERE"
    TEMPERATURE: float = 0.0  # Force maximum determinism to reduce scoring noise

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

judge_settings = JudgeSettings()
