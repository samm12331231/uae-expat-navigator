import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL
from backend.tools import fee_calculator, eligibility_checker

load_dotenv()

PROMPT_TEMPLATE = """You are an expert UAE government services assistant with access to official UAE government documents and verified tools.

You have two sources of information:
1. Official UAE government documents (PDFs from RTA, ICP, MOHRE, GDRFA)
2. Verified tool results from eligibility_checker and fee_calculator

HIERARCHY — TOOL RESULTS OVERRIDE DOCUMENTS:
- When tool results and documents conflict, the tool result is ALWAYS correct.
- If the Eligibility Check Result mentions alternative paths (Golden Chance, reduced training hours), you MUST include these details in your answer.
- NEVER say you "don't have information" about a topic that is covered in the tool results.
- NEVER let document content contradict, override, or dilute information from the tool results.

CRITICAL: If Eligibility Check says "Not Eligible", you MUST NOT list driving 
license exchange fees, because the user cannot do a direct exchange. Instead, 
explain the Golden Chance / reduced training alternative path. Do NOT include 
any AED fee breakdown for license exchange in this case.

IMPORTANT: There are two separate lists:
- Countries BANNED from exchange: Guam, Jersey, Liechtenstein, Monaco, North Mariana, Puerto Rico, Andorra
- Countries ELIGIBLE for direct exchange: Australia, UK, USA, Canada, Germany, France etc.
India is NOT banned. It is simply not on the eligible list. Never say India is banned.
Always say India is not eligible for direct exchange and the Golden Chance path applies.

For South Asian countries not eligible for direct exchange, mention that the full 
driving licence process through an RTA-approved driving school typically costs 
AED 3,500–11,500 depending on school and package. Limited packages start lower 
but add fees per failed test. Unlimited packages cover all retakes for a fixed price.
Contact an RTA-approved institute directly for current pricing.

STRICT RULES:
- Answer ONLY using the provided context below. Do not use outside knowledge.
- Always cite which document your answer comes from (when using document info).
- List steps clearly and numerically when the process has steps.
- Always mention fees if they appear in the context (UNLESS the user is not eligible — see rule above).
- If the provided context (documents AND tool results together) does not contain the answer, say exactly: "I don't have official information on that."
- Never guess, never hallucinate, never make up fees or timelines.
- Never show raw JSON, tool results, or internal processing in your answer.
- Fold tool results into natural language; do not say "the tool said" or "according to the tool".
- Only mention tools if they were actually used for this query.

---
TOOL RESULTS (Highest Priority):
{tool_context}

---
OFFICIAL DOCUMENTS:
{document_context}

---
Question: {question}

Answer:"""

def _extract_fees_from_docs(docs: List[Any]) -> List[str]:
    """Pull AED strings from retrieved chunks so the fee tool has clean input."""
    fees = []
    pattern = re.compile(r"AED[\s]*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
    seen = set()
    for doc in docs:
        for match in pattern.findall(doc.page_content):
            fee_str = f"AED {match}"
            if fee_str not in seen:
                seen.add(fee_str)
                fees.append(fee_str)
    return fees

def _format_docs(docs: List[Any]) -> str:
    """Deduplicate by source and format for the prompt."""
    parts = []
    seen_sources = set()
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            authority = doc.metadata.get("authority", "Unknown")
            parts.append(
                f"Document: {src}\n"
                f"Authority: {authority}\n"
                f"Content:\n{doc.page_content}"
            )
    return "\n\n---\n\n".join(parts)

def _detect_country(question: str) -> str | None:
    """Keyword matching for country names AND nationality forms (Indian, Pakistani, etc.)."""
    q = question.lower()
    mapping = {
        "united kingdom": "United Kingdom",
        "united states": "United States",
        "saudi arabia": "Saudi Arabia",
        "south korea": "South Korea",
        "sri lanka": "Sri Lanka",
        "india": "India",
        "indian": "India",
        "pakistan": "Pakistan",
        "pakistani": "Pakistan",
        "philippines": "Philippines",
        "filipino": "Philippines",
        "filipina": "Philippines",
        "bangladesh": "Bangladesh",
        "bangladeshi": "Bangladesh",
        "nepal": "Nepal",
        "nepali": "Nepal",
        "nepalese": "Nepal",
        "australia": "Australia",
        "australian": "Australia",
        "canada": "Canada",
        "canadian": "Canada",
        "germany": "Germany",
        "german": "Germany",
        "france": "France",
        "french": "France",
        "british": "United Kingdom",
        "american": "United States",
        "bahrain": "Bahrain",
        "bahraini": "Bahrain",
        "kuwait": "Kuwait",
        "kuwaiti": "Kuwait",
        "qatar": "Qatar",
        "qatari": "Qatar",
        "uae": "UAE",
        "uk": "United Kingdom",
        "usa": "United States",
    }
    for key, val in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(key) + r'\b', q):
            return val
    return None

def _needs_eligibility_check(question: str) -> bool:
    # Explicit driving-license keywords
    dl_keywords = [
        "driving license", "license exchange", "convert license",
        "exchange my", "exchange license", "eligible", "eligibility",
        "convert my", "driving licence", "licence exchange"
    ]
    if any(k in question.lower() for k in dl_keywords):
        return True

    # Vague follow-ups like "What about Pakistan?" — if a country is mentioned
    # and there are NO competing topic keywords, assume driving-license context
    other_topics = [
        "work permit", "work", "employment", "job",
        "visa", "residence", "family", "golden visa",
        "status change", "status amendment", "passport"
    ]
    if len(question) < 45 and not any(k in question.lower() for k in other_topics):
        return True

    return False

def _needs_fee_calculation(question: str) -> bool:
    keywords = ["cost", "fee", "price", "how much", "total", "pay"]
    return any(k in question.lower() for k in keywords)

def get_agent():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["tool_context", "document_context", "question"],
    )

    def _answer(question: str) -> Dict[str, Any]:
        docs = vectorstore.similarity_search(question, k=12)

        unique_docs = []
        seen = set()
        for d in docs:
            src = d.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                unique_docs.append(d)
            if len(unique_docs) >= 5:
                break

        detected_country = _detect_country(question)
        process_type = "driving_license_exchange"
        tool_results = {}
        tool_context_parts = []

        if detected_country and _needs_eligibility_check(question):
            try:
                print(f"\n🔧 TOOL_CALL: eligibility_checker({detected_country}, {process_type})")
                res = eligibility_checker.invoke({
                    "country": detected_country,
                    "process_type": process_type,
                })
                tool_results["eligibility"] = res
                print(f"🔧 TOOL_RESULT: {json.dumps(res, indent=2)}")
                tool_context_parts.append(
                    f"Eligibility Check Result for {detected_country}:\n"
                    f"Eligible: {'Yes' if res['eligible'] else 'No'}\n"
                    f"Reason: {res['reason']}\n"
                    f"Alternative Path: {res['alternative_path']}"
                )
            except Exception as exc:
                print(f"Eligibility tool error: {exc}")

        not_eligible = (
            tool_results.get("eligibility")
            and not tool_results["eligibility"].get("eligible", False)
        )

        fee_strings = _extract_fees_from_docs(unique_docs)
        if fee_strings and _needs_fee_calculation(question) and not not_eligible:
            try:
                print(f"\n🔧 TOOL_CALL: fee_calculator({fee_strings})")
                fee_res = fee_calculator.invoke({"fees": fee_strings})
                tool_results["fees"] = fee_res
                print(f"🔧 TOOL_RESULT: {json.dumps(fee_res, indent=2)}")
                breakdown_lines = "\n".join(
                    f"  - {item['item']}: AED {item['amount']}"
                    for item in fee_res["breakdown"]
                )
                tool_context_parts.append(
                    f"Fee Summary (AED):\n"
                    f"Total: {fee_res['total']}\n"
                    f"Breakdown:\n{breakdown_lines}"
                )
            except Exception as exc:
                print(f"Fee tool error: {exc}")

        doc_context = _format_docs(unique_docs)
        tool_context = "\n\n".join(tool_context_parts) if tool_context_parts else "No tools were used for this query."

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "tool_context": tool_context,
            "document_context": doc_context or "No relevant documents retrieved.",
            "question": question,
        })

        return {
            "answer": answer,
            "source_documents": [
                {"source": d.metadata.get("source", "Unknown")} for d in unique_docs
            ],
            "tool_calls": tool_results,
        }

    return _answer

if __name__ == "__main__":
    agent = get_agent()
    test_questions = [
        "Can I exchange my Indian driving license?",
        "How do I convert my UK driving license?",
        "How do I change my visa status?",
    ]
    for q in test_questions:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Q: {q}")
        result = agent(q)
        print(f"A: {result['answer']}")
        sources = [d['source'] for d in result['source_documents']]
        print(f"\nSources: {sources}")
        if result.get("tool_calls"):
            print(f"\nTool Calls: {json.dumps(result['tool_calls'], indent=2)}")