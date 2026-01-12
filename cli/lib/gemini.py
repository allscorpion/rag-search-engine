import os
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("no gemini api key set")

client = genai.Client(api_key=api_key)


def fix_spelling_mistakes(query: str):
    content = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:""",
    )
    if not content.text:
        return ""

    return content.text.replace('"', "")
