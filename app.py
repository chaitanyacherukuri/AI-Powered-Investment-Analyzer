import streamlit as st
from typing import TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper

import os

#Set API Keys
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

#Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

#Intialize Tools
wrapper = GoogleFinanceAPIWrapper(serp_api_key=st.secrets["SERP_API_KEY"])
tool = GoogleFinanceQueryRun(api_wrapper=wrapper)

tools = [tool]

#Initialize Agent
agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

#Define State
class State(TypedDict):
    investment_asset: str
    fundamental_insights: str
    technical_insights: str
    sentiment_insights: str
    risk_evaluation: str
    final_report: str


