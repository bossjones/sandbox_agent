# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/

# SOURCE: https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/
# SOURCE: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools
# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import json
import operator
import os

from collections.abc import Sequence
from fileinput import filename
from io import BytesIO
from typing import Annotated, Any, Dict, List, Literal, TypedDict, Union
from venv import create

import pysnooper

from langchain import hub
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from loguru import logger
from PIL import Image
from PIL.ImageFile import ImageFile

from sandbox_agent.agents.agent_executor import AgentExecutorFactory
from sandbox_agent.ai.graph import Act, Plan, PlanExecute, Response
from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.factories import (
    ChatModelFactory,
    DocumentLoaderFactory,
    EmbeddingModelFactory,
    EvaluatorFactory,
    MemoryFactory,
    RetrieverFactory,
    TextSplitterFactory,
    ToolFactory,
    VectorStoreFactory,
)


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

### Router

# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

### Retrieval Grader

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

### Generate

# Prompt
rag_prompt = """You are an assistant for question-answering tasks.

Here is the context to use to answer the question:

{context}

Think carefully about the above context.

Now, review the user question:

{question}

Provide an answer to this questions using only the above context.

Use three sentences maximum and keep the answer concise.

Answer:"""

### Hallucination Grader

# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

### Answer Grader

# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score.

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}.

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""


# Post-processing
def format_docs(docs: list[Document]) -> str:
    """
    Format a list of documents into a single string.

    This function takes a list of Document objects and concatenates their page_content
    attributes, separating each with two newline characters.

    Args:
        docs (List[Document]): A list of Document objects to format.

    Returns:
        str: A single string containing the formatted document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: list[str]  # List of retrieved documents


# model = ChatModelFactory.create()
# tools = ToolFactory.create_tools()
# model = model.bind_tools(tools)
# memory = MemoryFactory.create()

llm = ChatModelFactory.create()
llm_json_mode = EvaluatorFactory.create()
# Load documents
docs = [DocumentLoaderFactory.create(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = TextSplitterFactory.create()
doc_splits = text_splitter.split_documents(docs_list)
embeddings = EmbeddingModelFactory.create()
vector_store = VectorStoreFactory.create("sklearn").from_documents(documents=doc_splits, embedding=embeddings)
retriever = RetrieverFactory.create(vector_store)


web_search_tool = TavilySearchResults(k=3)


### Nodes


def retrieve(state: GraphState) -> dict[str, list[Document]]:
    """
    Retrieve documents from the vector store based on the user's question.

    This function takes the current graph state, which includes the user's question,
    and uses the retriever to find relevant documents from the vector store. The
    retrieved documents are then added to the state under the "documents" key.

    Args:
        state (GraphState): The current graph state containing the user's question.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "documents",
            containing the list of retrieved Document objects.

    Example:
        >>> state = {"question": "What is the capital of France?"}
        >>> result = retrieve(state)
        >>> result
        {"documents": [Document(page_content="...", ...), ...]}
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieve relevant documents from the vector store
    documents = retriever.invoke(question)

    return {"documents": documents}


def generate(state: GraphState):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}


def web_search(state: GraphState):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


### Edges


def route_question(state: GraphState):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(question=question, generation=generation.content)
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)] + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"


# Control Flow

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)


# Define a function to build the compile arguments
def build_compile_args() -> dict[str, Any]:
    """
    Build the compile arguments for the workflow graph.

    This function constructs a dictionary of arguments to be passed to the `workflow.compile()`
    method. The arguments include:

    - `checkpointer`: An instance of `MemorySaver` if `aiosettings.llm_memory_enabled` is True.
    - `interrupt_before`: A list of node names before which the execution should be interrupted
      to allow for human input. In this case, it is set to `["tools"]` if
      `aiosettings.llm_human_loop_enabled` is True.
    - `interrupt_after`: A list of node names after which the execution should be interrupted
      to allow for human input. This is currently commented out, but you can uncomment the line
      if you want to interrupt after the "tools" node as well.

    Returns:
        Dict[str, Any]: A dictionary containing the compile arguments.
    """
    compile_args = {}

    # if aiosettings.llm_memory_enabled:
    #     logger.info("Adding checkpointer to compile args")
    #     compile_args["checkpointer"] = memory

    # if aiosettings.llm_human_loop_enabled:
    #     logger.info("Adding interrupt_before to compile args")
    #     compile_args["interrupt_before"] = ["tools"]
    #     # Uncomment the following line if you want to interrupt after tools as well
    #     # compile_args["interrupt_after"] = ["tools"]

    logger.info(f"Compile args: {compile_args}")
    return compile_args


# Now we can compile the graph with conditional arguments
graph: CompiledStateGraph = workflow.compile(**build_compile_args())


def save_graph_image(graph: Any, fname: str = "graph_mermaid.png") -> None:
    """
    Save the graph image to a file.

    This function takes the graph object, generates a Mermaid PNG image,
    and saves it to the specified file.

    Args:
        graph (Any): The graph object to be visualized.
        fname (str, optional): The name of the file to save the image to.
            Defaults to "graph_mermaid.png".

    Returns:
        None
    """

    # Generate the Mermaid PNG image
    png_data = graph.get_graph().draw_mermaid_png()

    # Create a PIL Image from the PNG data
    image: ImageFile = Image.open(BytesIO(png_data))
    image.save(fname)

    logger.info(f"Graph image saved to {fname}")


# Save the graph image
save_graph_image(graph)

# Display the image (optional, for interactive environments)

# display(Image(graph.get_graph().draw_mermaid_png()))
