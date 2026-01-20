import json
import os
from dotenv import load_dotenv
from google import genai


model_name = "gemini-2.5-flash-lite"
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("no gemini api key set")

client = genai.Client(api_key=api_key)


def fix_spelling_mistakes(query: str):
    content = client.models.generate_content(
        model=model_name,
        contents=f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:""",
    )
    if not content.text:
        return ""

    return content.text.replace('"', "")


def rewrite_query(query: str):
    content = client.models.generate_content(
        model=model_name,
        contents=f"""Rewrite this movie search query to be more specific and searchable.

        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:""",
    )
    if not content.text:
        return ""

    return content.text.replace('"', "")


def expand_query(query: str):
    content = client.models.generate_content(
        model=model_name,
        contents=f"""Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """,
    )
    if not content.text:
        return ""

    return content.text.replace('"', "")


def rerank_document(query: str, doc: dict):
    content = client.models.generate_content(
        model=model_name,
        contents=f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:""",
    )
    if not content.text:
        return ""

    return content.text.replace('"', "")


def rerank_document_batch(query: str, doc_list_str: str):
    content = client.models.generate_content(
        model=model_name,
        contents=f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """,
    )
    if not content.text:
        return ""

    return content.text.replace('"', "")


def evaluate_results(query: str, results: dict[int, dict]):
    formatted_results = []

    for key in results:
        doc = results[key]
        formatted_results.append(doc["title"])

    content = client.models.generate_content(
        model=model_name,
        contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]""",
    )
    if not content.text:
        return ""

    return json.loads(content.text.replace('"', ""))
