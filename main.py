from dotenv import load_dotenv

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
openai_llm = OpenAI(temperature=0.6)
agent = initialize_agent(
    tools=load_tools(["wikipedia", "serpapi"], llm=openai_llm),
    llm=openai_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
prompt = PromptTemplate(
    input_variables=["country_name"],
    template="give me a good business idea for {country_name}. Answer in only one word.",
)
business_chain = LLMChain(llm=openai_llm, prompt=prompt, output_key="business_idea")
analysis_prompt = PromptTemplate(
    input_variables=["business_idea", "country_name"],
    template="give me an analysis of {business_idea} in {country_name}.",
)
analysis_chain = LLMChain(
    llm=openai_llm, prompt=analysis_prompt, output_key="business_analysis"
)
financial_prompt = PromptTemplate(
    input_variables=["business_idea", "country_name"],
    template="give me an financial data of {business_idea} in {country_name}. Need data untill 2023",
)
financial_chain = LLMChain(
    llm=agent, prompt=financial_prompt, output_key="financial_data"
)
chain = SequentialChain(
    chains=[business_chain, analysis_chain, financial_chain],
    input_variables=["country_name"],
    output_variables=["business_idea", "business_analysis", "financial_data"],
    verbose=True,
)
if __name__ == "__main__":
    country_name = input("Enter country name: ")
    print(chain({"country_name": country_name}))
