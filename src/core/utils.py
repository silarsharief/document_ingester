import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions

# Robust Retry Configuration
# 1. Wait 4s, then 8s, then 16s... (Exponential Backoff)
# 2. Stop after 5 failed attempts.
# 3. Catch 429 (Resource Exhausted) and 500 (Server Errors).
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(
        (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, exceptions.InternalServerError)
    )
)
def generate_content_with_retry(model, contents):
    """
    Wrapper for Gemini API calls that automatically handles Rate Limits.
    """
    return model.generate_content(contents)