from typing import Dict

BASE_SYSTEM = (
    "You are a clinical documentation assistant. Generate a structured SOAP note "
    "from the given doctor–patient conversation. Be concise, factual, and avoid hallucinations."
)

TEMPLATE = '''You will be given a transcript of a patient–provider conversation.
Produce a SOAP note with **four** sections, each starting on its own line with the exact headings:
S: ...
O: ...
A: ...
P: ...

Constraints:
- Use professional clinical language.
- Do not invent tests or facts absent from the transcript.
- If information is missing for a section, write "None reported".
- Keep each section under 5 bullet points; use short phrases.

Conversation:
{dialogue}
'''
def render_prompt(dialogue: str) -> Dict[str,str]:
    return {
        "system": BASE_SYSTEM,
        "user": TEMPLATE.format(dialogue=dialogue.strip())
    }
