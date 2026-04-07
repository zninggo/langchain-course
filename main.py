import os

from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

print(api_key)

def main():
    print("Hello from langchain-course!")



if __name__ == "__main__":
    main()
