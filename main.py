from pprint import pprint
from graph_agents import app
from dotenv import load_dotenv
load_dotenv()

# LangSmith Tracing

def main():
    inputs = {
        "question": "What was the last ufc event and who won?"
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':") 
            pprint(f"Value: {value}") # question, generation, documents 
            # pprint.pprint(value.keys(), indent=2, width=80, depth=None) # Optional: print full state at each node
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"]) # generate: {question, generation, documents}

if __name__ == "__main__":
    main()
