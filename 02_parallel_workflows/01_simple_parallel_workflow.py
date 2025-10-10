from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define the state
class Batsmans_State(TypedDict):
    runs: int
    fours : int
    sixes : int
    balls : int

    summary : str
    strike_rate : float
    balls_per_boundary:float
    boundary_percentage:float


# Define the functions for the node
def compute_SR(state: Batsmans_State) -> dict:
    strike_rate = state["runs"] * 100 / state["balls"]
    return {"strike_rate": strike_rate}

def compute_balls_per_boundary(state: Batsmans_State) -> dict:
    """
    Tells average balls that batsman hits fours and six
    """
    balls_per_boundary = state["balls"] / (state["fours"] + state["sixes"])
    return {"balls_per_boundary" : balls_per_boundary}

def compute_boundary_percentage(state: Batsmans_State) -> Batsmans_State:
    """
    Tells the % of runs that the batsman scored by hitting fours and sixes
    """

    boundary_percentage = (state["fours"] * 4 + state["sixes"] * 6) * 100 / state["runs"]
    return {"boundary_percentage" : boundary_percentage}

def generate_batsmans_summary(state: Batsmans_State) -> dict:
    """
    Generate summary of batsman stats
    """
    summary = f'''The strike rate of the batsman is {state['strike_rate']:.2f}%.\n,
    The boundary percentage is {state['boundary_percentage']:.2f}%.\n,
    The balls per boundary is {state["balls_per_boundary"]}
    '''
    return {"summary": summary}

               
# Define the graph
#                   ----> compute_SR
# START    [SPLIT]  ----> compute_balls_per_boundary   ----> generate_batsmans_summary ---> END
#                   ----> compute_boundary_percentage   
    
graph = StateGraph(Batsmans_State)

# Add nodes
graph.add_node("compute_SR", compute_SR)
graph.add_node("compute_boundary_percentage", compute_boundary_percentage)
graph.add_node("compute_balls_per_boundary", compute_balls_per_boundary)
graph.add_node("generate_batsmans_summary", generate_batsmans_summary)

# Add edges
# Fan-out
graph.add_edge(START, "compute_SR")
graph.add_edge(START, "compute_boundary_percentage")
graph.add_edge(START, "compute_balls_per_boundary")

# Fan-in
graph.add_edge("compute_SR", "generate_batsmans_summary")
graph.add_edge("compute_boundary_percentage", "generate_batsmans_summary")
graph.add_edge("compute_balls_per_boundary", "generate_batsmans_summary")
graph.add_edge("generate_batsmans_summary", END)

# Compile the graph
workflow = graph.compile()

initial_state = {
    "runs": 100,
    "fours": 10,
    "sixes": 3,
    "balls" : 78
}

final_state = workflow.invoke(initial_state)
print(final_state)