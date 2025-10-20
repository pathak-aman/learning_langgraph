# Aim:
# UPSC Essay Feeback Workflow

# We will use 3 parallel nodes to evaluate the essay feedback on the basis of:
# a. Clarity of thought
# b. Depth of analysis
# c. Overall language

# We will ask each node to provide a textual feedback and a score from 0 to 10 for each of the 3 categories.
# We will then use a node to combine the responses as follows:
# a. Generate a summary of all the feedbacks
# b. Average the scores

# Challenges:
# 1. How to handle that LLMs make sure that the output from each node is in the same format?
# 2. We need to deal how to get average score -> We will need a reducer so that a single list can update and store scores from all nodes.

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import os
import operator
from dotenv import load_dotenv
load_dotenv()

class UPSC_Essay_Feedback(TypedDict):
    essay_topic : str
    essay_text : str

    clarity_of_thought_feedback : str
    depth_of_analysis_feedback : str
    overall_language_feedback: str

    scores : Annotated[list[int], operator.add]
    average_score : float

    final_summary : str

class Evaluation_Feedback(BaseModel):
    feedback: str = Field(description="Feedback on the essay")
    score: int = Field(description="Score for the feedback", ge=0, le=10)

eval_model = ChatOpenAI(
    base_url= "https://api.ai.it.ufl.edu",
    model = "gpt-oss-20b",
    temperature=0.1,
    api_key=os.getenv("NAVIGATOR_TOOLKIT_API_KEY")
)

summary_model = ChatOpenAI(
    base_url= "https://api.ai.it.ufl.edu",
    model = "gpt-oss-20b",
    temperature=0.1,
    api_key=os.getenv("NAVIGATOR_TOOLKIT_API_KEY")
)

def get_feedback(state: UPSC_Essay_Feedback, eval_category: str):
    essay_topic = state["essay_topic"]
    essay_text = state["essay_text"]
    messages = [
        SystemMessage(content=f"Evaluate the essay feedback on the basis of {eval_category}, provide a feedback containing no more than 5 bullet points and score from 0 to 10"),
        HumanMessage(content=f"Essay topic {essay_topic} \nEssay: {essay_text}")
    ]

    struct_model = eval_model.with_structured_output(Evaluation_Feedback)
    response = struct_model.invoke(messages)

    formatted_feedback = {
        f"{eval_category}_feedback" : response.feedback,
        "scores" : [response.score]
    }
    return formatted_feedback


def generate_feedback_summary(state: UPSC_Essay_Feedback) -> dict:

    print(state.keys())
    clarity_of_thought_feedback = state["clarity_of_thought_feedback"]
    depth_of_analysis_feedback = state["depth_of_analysis_feedback"]
    overall_language_feedback = state["overall_language_feedback"]

    messages = [
        SystemMessage(content=f"Generate a total of 6 bullet point summary of all the 3 kinds of feedbacks (2 per category) for the UPSC Essay"),
        HumanMessage(content=f"Clarity of thought feedback: {clarity_of_thought_feedback}\nDepth of analysis feedback: {depth_of_analysis_feedback}\nOverall language feedback: {overall_language_feedback}")
    ]
    response = summary_model.invoke(messages)
    summary = response.content

    average_score = sum(state["scores"]) / len(state["scores"])
    return {"final_summary" : summary, "average_score" : average_score}


# Create the graph
graph = StateGraph(UPSC_Essay_Feedback)

# Add nodes
graph.add_node("clarity_eval", lambda x: get_feedback(x, "clarity_of_thought"))
graph.add_node("depth_eval", lambda x: get_feedback(x, "depth_of_analysis"))
graph.add_node("language_eval", lambda x: get_feedback(x, "overall_language"))
graph.add_node("summary", generate_feedback_summary)

# Add edges for parallel evaluation
graph.add_edge(START, "clarity_eval")
graph.add_edge(START, "depth_eval")
graph.add_edge(START, "language_eval")

# Add edges from evaluations to summary
graph.add_edge("clarity_eval", "summary")
graph.add_edge("depth_eval", "summary")
graph.add_edge("language_eval", "summary")
graph.add_edge("summary", END)

# Compile the graph
workflow = graph.compile()

initial_state = UPSC_Essay_Feedback(
    essay_topic = "Poverty Anywhere is a Threat to Prosperity Everywhere.",
    essay_text = '''Introduction

The modern globalized world, with its interconnected economies and shared digital space, has rendered geographical borders irrelevant to shared human destiny. The saying, "Poverty anywhere is a threat to prosperity everywhere," transcends mere altruism; it is a statement of pragmatic self-interest and a reflection of global economic reality. Prosperity, often viewed as a national achievement, is in fact a fragile edifice that cannot stand securely on a foundation surrounded by human deprivation. The threat posed by localized poverty is manifold, manifesting in economic, social, and political instabilities that inevitably ripple across the globe.

Economic Threat: The Chain of Supply and Demand

Localized poverty acts as a fundamental drag on global prosperity by shrinking markets and disrupting supply chains. A poor community is a non-consuming community. When large populations are excluded from the economic process, the potential demand base for global goods and services is severely limited. For instance, low per capita income in developing nations restricts the growth of multinational corporations, whose profits often drive innovation and investment globally.

Furthermore, poverty fuels instability, which directly impacts the cost of doing business. Conflict, migration crises, and health emergencies—all of which thrive in conditions of poverty—create uncertainty, driving up insurance premiums, deterring foreign investment, and forcing nations to divert capital from productive ventures to security and crisis management. Prosperity cannot be fully realized when billions remain outside the formal economy, converting potential consumers and producers into dependent burdens.

Social and Security Threats: Contagion and Instability

Beyond economics, poverty breeds critical social instability that directly jeopardizes the security of prosperous regions.

Historically, disparities in wealth and opportunity are the most potent drivers of social unrest, extremism, and organized crime. A populace with no stake in the prevailing social order becomes fertile ground for radical ideologies, leading to terrorism and internal conflict that demand international intervention. The effects are immediate and visible: mass migration strains the resources and political consensus of host nations, while trans-national crimes like drug trafficking and human smuggling flourish in failed states defined by high poverty. Health crises, such as the COVID-19 pandemic, demonstrated that a disease born of poor sanitation and poverty in one corner of the world can instantly shut down the most advanced economies. Prosperity is, therefore, always one pandemic or one refugee crisis away from destabilization.

Conclusion and Way Forward

The threat posed by poverty is not a matter of 'them' versus 'us'; it is a challenge to the collective 'we.' True, enduring prosperity requires inclusive growth. This necessitates a global commitment to investing in human capital in developing nations through universal education, healthcare, and digital literacy. Policies like fair trade agreements and technology transfer must prioritize poverty alleviation, transforming vulnerable populations into economic participants. The future of global well-being depends on recognizing that a rising tide lifts all boats, but a hole in any one boat endangers the entire fleet. To secure prosperity at home, we must actively work to eliminate poverty everywhere.'''
)

final_state = workflow.invoke(initial_state)
for key, value in final_state.items():
    print(f"{key}: {value}")
    print("\n\n")


