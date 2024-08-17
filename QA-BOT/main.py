import pandas as pd
from system import System

def main():

    system = System()

    """
    df = pd.read_csv("data.csv")
    print("Processing and indexing data...")
    system.process_and_index_data(df)
    print("Data processing and indexing completed.")
     """

    question = "Where did ip 10.216.113.172 get 15779 datas from at 02:51 am"
    print(f"Asking: {question}")

    answer = system.give_us_answer(question)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
