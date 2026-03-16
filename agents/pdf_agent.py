from typing import Dict, List, Any, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from tools.reranker import Reranker


# ============================================================
# State Definition
# ============================================================
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    question: str               # Original user question
    standalone_question: str    # Reformulated question (context-aware)
    documents: List[Any]
    generation: str
    citations: List[Dict[str, Any]]
    iterations: int
    grade: str
    feedback: str


# ============================================================
# Structured Output for Reflection
# ============================================================
class GradeAnswer(BaseModel):
    """Binary score to assess answer quality."""
    binary_score: str = Field(description="'yes' if the answer fully addresses the question given the context, otherwise 'no'")
    feedback: str = Field(description="Specific improvement suggestions if score is 'no'")


# ============================================================
# PDF Agent — LangGraph Workflow
# ============================================================
class PDFAgent:
    """
    Orchestrates retrieval and generation with:
    1. Query reformulation (uses chat history for follow-ups)
    2. Hybrid retrieval (FAISS + BM25)
    3. Context-grounded generation
    4. Self-reflection loop
    """

    def __init__(
        self,
        retriever_callable,
        max_iterations: int = 2,
        ollama_model: str = "llama3.1:latest",
        ollama_base_url: Optional[str] = None,
        use_reranker: bool = False,
    ):
        self.retriever = retriever_callable
        self.max_iterations = max_iterations

        # Local Ollama model (no API key required)
        self.llm = ChatOllama(
            model=ollama_model,
            temperature=0,
            base_url=ollama_base_url,
        )

        # Cross-encoder re-ranker is optional (can be slow to download on first use)
        self.reranker = Reranker() if use_reranker else None

        # --- Prompt: Reformulate follow-up into standalone question ---
        self.reformulate_prompt = PromptTemplate(
            template="""Given the following conversation history and a new user question, 
reformulate the question into a clear, standalone question that can be understood without the conversation history.
If the question is already standalone, return it as-is.

Chat History:
{chat_history}

New Question: {question}

Standalone Question:""",
            input_variables=["chat_history", "question"]
        )

        # --- Prompt: Generate answer ---
        self.generate_prompt = PromptTemplate(
            template="""You are a precise document analysis assistant. You must ONLY answer based on the text provided inside the <CONTEXT> tags below. 

CRITICAL RULES:
- Your answer must be based EXCLUSIVELY on the text inside <CONTEXT>. 
- Do NOT use any prior knowledge, training data, or external information.
- Every claim in your answer must be directly traceable to a specific passage in the context.
- If the context does not contain the answer, respond: "The provided document excerpts do not contain this information."
- When listing items, ONLY include items that appear verbatim or are clearly described in the context text.
- DISTINGUISH between the document's main body content and any references/bibliography entries. When asked about what the document "mentions" or "describes", refer only to the main body.

<CONTEXT>
{context}
</CONTEXT>

Question: {question}

Answer (based ONLY on the above context):""",
            input_variables=["question", "context"],
        )

        # --- Prompt: Reflect on answer quality ---
        self.reflect_prompt = PromptTemplate(
            template="""You are a strict quality grader for AI-generated answers about PDF documents.

User Question: {question}

Retrieved Context: {context}

Agent Answer: {generation}

Evaluate strictly:
1. Does the answer ONLY use information from the context? (No hallucination)
2. Does it fully address the user's question?
3. Are any numbers, counts, or specific claims verifiable from the context?

If the answer is accurate and complete, score 'yes'.
If there are any issues (hallucination, incomplete, inaccurate counts), score 'no' and explain what's wrong.""",
            input_variables=["question", "context", "generation"]
        )

        # --- Prompt: Improve answer based on feedback ---
        self.improver_prompt = PromptTemplate(
            template="""Improve this answer based on the grader's feedback.
You must ONLY use information from the text inside <CONTEXT>. Do NOT add any external knowledge.

<CONTEXT>
{context}
</CONTEXT>

Question: {question}

Previous Answer: {generation}

Grader Feedback: {feedback}

Improved Answer (based ONLY on the above context):""",
            input_variables=["question", "context", "generation", "feedback"]
        )

        self.reflection_grader = self.llm.with_structured_output(GradeAnswer)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("reformulate", self.reformulate_query)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("reflect", self.reflect_and_improve_answer)

        workflow.add_edge(START, "reformulate")
        workflow.add_edge("reformulate", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "reflect")

        workflow.add_conditional_edges(
            "reflect",
            self.decide_to_generate,
            {"useful": END, "not_useful": "generate"}
        )

        return workflow.compile()

    # ----------------------------------------------------------
    # Node: Reformulate query using chat history
    # ----------------------------------------------------------
    def reformulate_query(self, state: AgentState) -> Dict:
        """Uses chat history to turn follow-up questions into standalone queries."""
        print("---REFORMULATE---")
        question = state["question"]
        messages = state.get("messages", [])

        # If there's no history, skip reformulation
        if not messages:
            print(f"No history. Using question as-is: {question}")
            return {"standalone_question": question}

        # Format history for the prompt
        history_text = ""
        for msg in messages[-6:]:  # Last 3 exchanges max
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:300]  # Truncate long messages
            history_text += f"{role}: {content}\n"

        prompt = self.reformulate_prompt.format(
            chat_history=history_text,
            question=question
        )
        response = self.llm.invoke(prompt)
        standalone = response.content.strip()
        print(f"Reformulated: '{question}' → '{standalone}'")
        return {"standalone_question": standalone}

    # ----------------------------------------------------------
    # Node: Retrieve relevant documents
    # ----------------------------------------------------------
    def retrieve_documents(self, state: AgentState) -> Dict:
        print("---RETRIEVE---")
        # Use the reformulated question for retrieval
        question = state.get("standalone_question", state["question"])

        documents = self.retriever.invoke(question)
        print(f"Initial retrieval: {len(documents)} chunks")

        # Re-rank with cross-encoder for precision (optional)
        if self.reranker is not None:
            documents = self.reranker.rerank(question, documents, top_k=5)
            print(f"After re-ranking: {len(documents)} chunks")

        # Extract citations
        citations = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", -1)
            if page != -1:
                page += 1  # 0-indexed → 1-indexed
            citations.append({"source": source, "page": page})

        # Deduplicate citations
        unique_citations = [dict(t) for t in {tuple(d.items()) for d in citations}]

        print(f"{len(unique_citations)} unique sources")
        return {"documents": documents, "citations": unique_citations}

    # ----------------------------------------------------------
    # Node: Generate answer
    # ----------------------------------------------------------
    def generate_answer(self, state: AgentState) -> Dict:
        print("---GENERATE---")
        question = state.get("standalone_question", state["question"])
        documents = state["documents"]
        iterations = state.get("iterations", 0)
        feedback = state.get("feedback", "")

        context = "\n\n---\n\n".join([d.page_content for d in documents])

        if iterations == 0 or not feedback:
            prompt = self.generate_prompt.format(question=question, context=context)
        else:
            generation = state.get("generation", "")
            prompt = self.improver_prompt.format(
                question=question, context=context,
                generation=generation, feedback=feedback
            )

        response = self.llm.invoke(prompt)

        # Update message history
        messages = list(state.get("messages", []))
        if iterations == 0:
            messages.append(HumanMessage(content=state["question"]))  # Original question
        messages.append(AIMessage(content=response.content))

        return {"generation": response.content, "iterations": iterations, "messages": messages}

    # ----------------------------------------------------------
    # Node: Reflect on answer quality
    # ----------------------------------------------------------
    def reflect_and_improve_answer(self, state: AgentState) -> Dict:
        print("---REFLECT---")
        question = state.get("standalone_question", state["question"])
        documents = state["documents"]
        generation = state["generation"]
        iterations = state["iterations"] + 1

        context = "\n\n---\n\n".join([d.page_content for d in documents])
        prompt = self.reflect_prompt.format(
            question=question, context=context, generation=generation
        )

        score_res = self.reflection_grader.invoke(prompt)
        print(f"Reflection: score={score_res.binary_score}, feedback={score_res.feedback[:100]}")

        return {
            "iterations": iterations,
            "grade": score_res.binary_score.lower(),
            "feedback": score_res.feedback
        }

    # ----------------------------------------------------------
    # Conditional Edge: Continue or stop
    # ----------------------------------------------------------
    def decide_to_generate(self, state: AgentState) -> str:
        grade = state.get("grade", "yes")
        iterations = state.get("iterations", 0)

        if grade == "yes" or iterations >= self.max_iterations:
            print("---DECISION: ACCEPT---")
            return "useful"
        else:
            print("---DECISION: RETRY---")
            return "not_useful"

    # ----------------------------------------------------------
    # Public Entry Point
    # ----------------------------------------------------------
    def run(self, question: str, chat_history: Sequence[BaseMessage] = None) -> Dict:
        if chat_history is None:
            chat_history = []

        initial_state = {
            "question": question,
            "standalone_question": question,
            "messages": chat_history,
            "iterations": 0,
            "documents": [],
            "generation": "",
            "citations": [],
            "grade": "yes",
            "feedback": ""
        }

        result = self.graph.invoke(initial_state)
        return {
            "answer": result.get("generation", ""),
            "citations": result.get("citations", []),
            "chat_history": result.get("messages", [])
        }
