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

from .config import CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL
from .tools import fee_calculator, eligibility_checker

load_dotenv()

PROMPT_TEMPLATE = """You are an expert UAE government services assistant with access to official documents and verified tools.

You have two tools available:
1. fee_calculator: Use this when the user asks about costs or fees.
2. eligibility_checker: Use this when the user asks about eligibility for a specific country and process.

STRICT RULES:
- Answer ONLY using the context provided below from official UAE documents
- Always mention which document your answer comes from
- List steps clearly and numerically when the process has steps
- Always mention fees if they appear in the context
- If the context does not contain the answer say exactly: "I don't have official information on that."
- Never guess, never hallucinate, never make up fees or timelines
- Never show raw JSON, tool results, or internal processing in your answer
- If a tool was used, fold its result into your answer naturally; do not say "the tool said" or "according to the tool".
- Only mention tools if they were actually used for this query. If no tool was called, answer directly from the documents without mentioning tools.

Context from official documents:
{context}

Question: {question}

Answer:"""


def _extract_fees_from_docs(docs: List[Any]) -> List[str]:
    """Pull AED strings from retrieved chunks so the fee tool has clean input."""
    fees = []
    # Handles AED 1,200.50, AED 200, etc.
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
    """Keyword matching for country names with word boundaries."""
    q = question.lower()
    mapping = {
        "united kingdom": "United Kingdom",
        "united states": "United States",
        "saudi arabia": "Saudi Arabia",
        "south korea": "South Korea",
        "india": "India",
        "pakistan": "Pakistan",
        "philippines": "Philippines",
        "bangladesh": "Bangladesh",
        "sri lanka": "Sri Lanka",
        "nepal": "Nepal",
        "australia": "Australia",
        "canada": "Canada",
        "germany": "Germany",
        "france": "France",
        "bahrain": "Bahrain",
        "kuwait": "Kuwait",
        "qatar": "Qatar",
        "uae": "UAE",
        "uk": "United Kingdom",
        "usa": "United States",
    }
    # Longer phrases first so "united kingdom" beats "uk"
    for key, val in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(key) + r'\b', q):
            return val
    return None


def _needs_eligibility_check(question: str) -> bool:
    keywords = [
        "driving license", "license exchange", "convert license",
        "exchange my", "exchange license", "eligible", "eligibility"
    ]
    return any(k in question.lower() for k in keywords)


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
        input_variables=["context", "question"],
    )

    def _answer(question: str) -> Dict[str, Any]:
        # 1. Retrieve
        docs = vectorstore.similarity_search(question, k=12)

        # 2. Deduplicate by source (max 5)
        unique_docs = []
        seen = set()
        for d in docs:
            src = d.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                unique_docs.append(d)
            if len(unique_docs) >= 5:
                break

        # 3. Detect intent & entities
        detected_country = _detect_country(question)
        process_type = "driving_license_exchange"
        tool_results = {}
        context_parts = []

        # 4. Conditional tool: Eligibility
        if detected_country and _needs_eligibility_check(question):
            try:
                print(f"\n🔧 TOOL_CALL: eligibility_checker({detected_country}, {process_type})")
                res = eligibility_checker.invoke({
                    "country": detected_country,
                    "process_type": process_type,
                })
                tool_results["eligibility"] = res
                print(f"🔧 TOOL_RESULT: {json.dumps(res, indent=2)}")
                context_parts.append(
                    f"Eligibility Check Result for {detected_country}:\n"
                    f"Eligible: {'Yes' if res['eligible'] else 'No'}\n"
                    f"Reason: {res['reason']}\n"
                    f"Alternative Path: {res['alternative_path']}"
                )
            except Exception as exc:
                print(f"Eligibility tool error: {exc}")

        # 5. Conditional tool: Fee calculation
        fee_strings = _extract_fees_from_docs(unique_docs)
        if fee_strings and _needs_fee_calculation(question):
            try:
                print(f"\n🔧 TOOL_CALL: fee_calculator({fee_strings})")
                fee_res = fee_calculator.invoke({"fees": fee_strings})
                tool_results["fees"] = fee_res
                print(f"🔧 TOOL_RESULT: {json.dumps(fee_res, indent=2)}")
                breakdown_lines = "\n".join(
                    f"  - {item['item']}: AED {item['amount']}"
                    for item in fee_res["breakdown"]
                )
                context_parts.append(
                    f"Fee Summary (AED):\n"
                    f"Total: {fee_res['total']}\n"
                    f"Breakdown:\n{breakdown_lines}"
                )
            except Exception as exc:
                print(f"Fee tool error: {exc}")

        # 6. Base document context
        doc_context = _format_docs(unique_docs)
        if doc_context:
            context_parts.insert(0, doc_context)

        final_context = "\n\n".join(context_parts)

        # 7. Invoke LLM
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": final_context, "question": question})

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
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = agent(q)
        print(f"A: {result['answer']}")
        print(f"\nSources: {[d['source'] for d in result['source_documents']]}")
        if result.get("tool_calls"):
            print(f"\nTool Calls: {json.dumps(result['tool_calls'], indent=2)}")
