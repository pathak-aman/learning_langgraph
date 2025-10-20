from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()


class ReviewReplyState(TypedDict):
    review_text : str
    sentiment : str
    reply_text : str
    order_number : str
    category : str
    is_urgent : bool
    summary : str


# Two models - identify sentiment and generate reply
sentiment_model = ChatOpenAI(base_url= "https://api.ai.it.ufl.edu",model = "gpt-oss-20b",temperature=0.1,api_key=os.getenv("NAVIGATOR_TOOLKIT_API_KEY"))
reply_model = ChatOpenAI(base_url= "https://api.ai.it.ufl.edu",model = "gpt-oss-20b",temperature=0.1,api_key=os.getenv("NAVIGATOR_TOOLKIT_API_KEY"))
diagnosis_model = ChatOpenAI(base_url= "https://api.ai.it.ufl.edu",model = "gpt-oss-120b",temperature=0.1,api_key=os.getenv("NAVIGATOR_TOOLKIT_API_KEY"))

# Pydantic model for sentiment model response
class SentimentModelResponse(BaseModel):
    sentiment: str = Field(description="Sentiment of the review")

class DiagnosisModelResponse(BaseModel):
    order_number: str = Field(description="Order number of the customer")
    category: str = Field(description="Category of the customer review")
    is_urgent: bool = Field(description="Whether the customer review is urgent")
    summary: str = Field(description="Summary of the customer review")

sentiment_model_str = sentiment_model.with_structured_output(SentimentModelResponse)
diagnosis_model_str = diagnosis_model.with_structured_output(DiagnosisModelResponse)

# Functions

def get_sentiment(ReviewReplyState):
    review_text = ReviewReplyState["review_text"]
    messages = [
        SystemMessage(content="Return only a valid JSON output identifying the sentiment of the review as positive, negative or neutral. JSON Schema: {{'sentiment' : 'positive' | 'negative' | 'neutral'}}"),
        HumanMessage(content=f"Review Text: {review_text}")
    ]
    response = sentiment_model_str.invoke(messages)
    return {"sentiment" : response.sentiment}

def check_sentiment(state: ReviewReplyState):
    sentiment = state["sentiment"]
    if sentiment == "positive":
        return "generate_positive_reply"
    elif sentiment == "negative":
        return "run_diagnosis_for_negative_reply"
    else:
        return "generate_neutral_reply"

def generate_reply_for_pos_review(state: ReviewReplyState):
    review_text = state["review_text"]
    messages = [
        SystemMessage(content="Generate a thankful reply to the customer's review text."),
        HumanMessage(content=f"Review Text: {review_text}")
    ]
    response = reply_model.invoke(messages)
    return {"reply_text" : response.content}

def generate_reply_for_neu_review(state: ReviewReplyState):
    review_text = state["review_text"]
    messages = [
        SystemMessage(content="Generate a friendly reply to the customer's review text, mentioning that we strive to improve our services."),
        HumanMessage(content=f"Review Text: {review_text}")
    ]
    response = reply_model.invoke(messages)
    return {"reply_text" : response.content}

def run_diagnosis(state: ReviewReplyState):
    review_text = state["review_text"]
    messages = [
        SystemMessage(content='''You are a API generating valid JSON response. Given a negative customer review text, generate a JSON response with the following JSON fields:
          {{
“order_number” : “string or null”,
“sentiment” : “string”, // Must of one of: "positive", "neutral", "negative"
“category” : “string”, // Must be only one of: "Billing Issue", "Shipping Delay", "Product Quality", "Return Request", "Other"
“is_urgent” : bool,
“summary” : “string”  // Must be non-null
}}'''
                      ),
        HumanMessage(content=f"Review Text: {review_text}")
    ]
    response = diagnosis_model_str.invoke(messages)

    return {
        "order_number" : response.order_number, "category" : response.category, "is_urgent" : response.is_urgent, "summary" : response.summary
    }


def generate_reply_for_neg_review(state: ReviewReplyState):
    order_number = state["order_number"]
    category = state["category"]
    is_urgent = state["is_urgent"]
    summary = state["summary"]
    messages = [
        SystemMessage(content="Generate a reassuring reply to the customer who shared a negative review, emphasising that we apologize and resolving your case."),
        HumanMessage(content=f"Order Number: {order_number}\nCategory: {category}\nIs Urgent: {is_urgent}\nSummary: {summary}")
    ]

    response = reply_model.invoke(messages)
    return {"reply_text" : response.content}


# Graph
graph = StateGraph(ReviewReplyState)

graph.add_node("sentiment_analysis", get_sentiment)
graph.add_node("check_sentiment",check_sentiment)
graph.add_node("generate_positive_reply", generate_reply_for_pos_review)
graph.add_node("generate_neutral_reply", generate_reply_for_neu_review)
graph.add_node("run_diagnosis_for_negative_reply", run_diagnosis)
graph.add_node("generate_negative_reply", generate_reply_for_neg_review)



graph.add_edge(START, "sentiment_analysis")
graph.add_conditional_edges("sentiment_analysis", check_sentiment)

graph.add_edge("generate_positive_reply", END)
graph.add_edge("generate_neutral_reply", END)
graph.add_edge("run_diagnosis_for_negative_reply", "generate_negative_reply")

graph.add_edge("generate_negative_reply", END)

workflow = graph.compile()

print("Starting workflow")
print(workflow)


initial_state = {
    "review_text" : '''Hello,
I'm writing about my recent order, #A-9341-B. To be honest, I'm really disappointed. I bought the 'Aura Pro' wireless earbuds, and the battery life is the main problem. The website said they last 12 hours, but mine are dying in under 3, even after a full charge. This is incredibly frustrating as I've been a loyal customer for years.
I need these for a work trip this Friday, so I need a resolution ASAP. Please let me know what my options are for a replacement.
Thanks,
Sarah Jenkins'''
}

final_state = workflow.invoke(initial_state)
print(final_state)