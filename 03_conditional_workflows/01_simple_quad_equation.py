# Calculate the discriminant and add a conditional graph node to compute the roots
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

import os
from dotenv import load_dotenv
load_dotenv()

class QuadEquationState(TypedDict):
    a : float
    b : float
    c : float
    discriminant : float
    root1 : float
    root2 : float

# Functions
def calculate_d(state: QuadEquationState):
    a = state["a"]
    b = state["b"]
    c = state["c"]
    discriminant = b**2 - 4*a*c
    return {"discriminant" : discriminant}

def check_d_condition(state: QuadEquationState):
    """ This will output the next node based on the condition
        if d < 0, then go to node unreal roots
        elif d == 0, then go to identical roots
        else go to real roots
    """
    d = state["discriminant"]
    if d < 0:
        return "unreal_roots"
    elif d == 0:
        return "identical_roots"
    else:
        return "real_roots"

def cal_real_roots(state: QuadEquationState):
    a = state["a"]
    b = state["b"]
    c = state["c"]
    d = state["discriminant"]
    root1 = (-b + d**0.5) / (2*a)
    root2 = (-b - d**0.5) / (2*a)
    return {"root1" : root1, "root2" : root2}

def cal_identical_roots(state: QuadEquationState):
    a = state["a"]
    b = state["b"]
    c = state["c"]
    d = state["discriminant"]
    root1 = (-b + d**0.5) / (2*a)
    root2 = root1
    return {"root1" : root1, "root2" : root2}


def cal_unreal_roots(state: QuadEquationState):
    return {"root1" : None, "root2" : None}

def print_equation_and_roots(state: QuadEquationState):
    a = state["a"]
    b = state["b"]
    c = state["c"]
    d = state["discriminant"]
    root1 = state["root1"]
    root2 = state["root2"]
    equation = f"{a}x^2 + {b}x + {c} = 0"
    print(equation)
    print(f"Discriminant value: {d}")
    print(f"The roots are {root1} and {root2}")

# Graph
graph = StateGraph(QuadEquationState)

graph.add_node("calculate_discriminant", calculate_d)
graph.add_node("check_condition", check_d_condition)
graph.add_node("real_roots", cal_real_roots)
graph.add_node("identical_roots", cal_identical_roots)
graph.add_node("unreal_roots", cal_unreal_roots)
graph.add_node("print_equation_and_roots", print_equation_and_roots)

graph.add_edge(START, "calculate_discriminant")
graph.add_conditional_edges("calculate_discriminant", check_d_condition)
graph.add_edge("real_roots", "print_equation_and_roots")
graph.add_edge("identical_roots", "print_equation_and_roots")
graph.add_edge("unreal_roots", "print_equation_and_roots")
graph.add_edge("print_equation_and_roots", END)

workflow = graph.compile()

print("Starting workflow")
print(workflow)

initial_state = {
    "a" : 1,
    "b" : 6,
    "c" : 9
}

final_state = workflow.invoke(initial_state)
print(final_state)