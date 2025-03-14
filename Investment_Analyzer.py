import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

import os

#Set API Keys
os.environ["GOOGLE_API_KEY"] = st.secrets["GROQ_API_KEY"]

#Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

#Intialize Tools
wrapper = GoogleFinanceAPIWrapper(serp_api_key=st.secrets["SERP_API_KEY"])
gfinance = GoogleFinanceQueryRun(api_wrapper=wrapper)

yfinance = YahooFinanceNewsTool()

tools = [gfinance, yfinance]

#Initialize Agent
agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

#Define State
class State(TypedDict):
    investment_asset: str
    fundamental_insights: str
    technical_insights: str
    sentiment_insights: str
    risk_evaluation: str
    final_report: str

#Define Functions For Each Node in the Langgraph Workflow
def fundamental_analysis(state: State):
    """Performs Fundamental Analysis (P/E Ratio, Revenue, Earnings)"""

    asset = state["investment_asset"]

    prompt = f"""
    Perform a detailed fundamental analysis on {asset} covering:
    1. Valuation Metrics: P/E ratio, P/B ratio, P/S ratio compared to industry averages
    2. Financial Health: Debt-to-equity ratio, current ratio, quick ratio
    3. Growth Metrics: Revenue growth rate, earnings growth rate, projected growth
    4. Profitability: Profit margins, ROE, ROA, and their trends
    5. Cash Flow Analysis: Free cash flow, operating cash flow trends
    6. Dividend Analysis (if applicable): Yield, payout ratio, dividend growth
    7. Fundamental Outlook: Overall assessment based on fundamentals
    
    For cryptocurrencies, adapt metrics to include market cap, trading volume, network metrics, and development activity.
    For real estate, focus on cap rates, NOI, vacancy rates, and regional market trends.
    """
    response = agent.invoke(prompt)
    return {"fundamental_insights": response["output"]}


def technical_analysis(state: State):
    """Performs Technical Analysis (Moving Averages, RSI, MACD, Bollinger Bands)"""

    asset = state["investment_asset"]

    prompt = f"""
    Perform a comprehensive technical analysis on {asset} with the following structure:
    1. Current Price Trend: Identify whether the asset is in an uptrend, downtrend, or trading sideways
    2. Support and Resistance Levels: Identify key price levels
    3. RSI Analysis: Current RSI value and whether the asset is overbought or oversold
    4. MACD Analysis: Current signal and potential crossovers
    5. Moving Averages: Relationship between short-term and long-term moving averages
    6. Volume Analysis: Recent volume trends and what they indicate
    7. Chart Patterns: Identify key chart patterns (head and shoulders, double top/bottom, etc.)
    8. Technical Outlook: Overall assessment based on technical indicators

    Be specific with numbers where possible and explain the technical significance.
    """
    response = agent.invoke(prompt)
    return {"technical_insights": response["output"]}


def sentiment_analysis(state: State):
    """Performs Sentiment Analysis (News, Social Media, Forums)"""

    asset = state["investment_asset"]

    prompt = f"""
    Analyze the current market sentiment for {asset} by examining:
    1. Recent News Coverage: Summarize major news and its impact (positive/negative)
    2. Social Media Sentiment: General tone on X, Stocktwits, Reddit, and other platforms
    3. Analyst Opinions: Recent analyst ratings, price targets, and consensus
    4. Institutional Interest: Recent institutional buying or selling activity
    5. Retail Investor Sentiment: Retail investor interest and sentiment trends
    6. Market Narratives: Dominant narratives or stories surrounding this asset
    7. Sentiment Outlook: Overall sentiment assessment and potential market psychology factors
    
    Provide specific examples of recent sentiment drivers where possible.
    """
    response = agent.invoke(prompt)
    return {"sentiment_insights": response["output"]}

def risk_assessment(state: State):
    """Conducts a Risk Evaluation (Market, Industry, Company, Financial, Regulatory, Competitive, Macro)"""
    
    asset = state["investment_asset"]

    prompt = f"""Conduct a comprehensive risk assessment for {asset} by analyzing:
    1. Market Risk: General market conditions and potential impact on the asset
    2. Volatility Metrics: Historical volatility, beta (for stocks), and comparison to benchmarks
    3. Downside Risk: Maximum drawdown history, potential downside scenarios
    4. Correlation: Correlation with broader market and diversification potential
    5. Liquidity Risk: Trading volume, bid-ask spreads, and liquidity concerns
    6. Regulatory/Legal Risks: Pending regulations or legal challenges
    7. Industry-Specific Risks: Competitive threats, disruption potential, industry headwinds
    8. Macroeconomic Sensitivity: How economic factors (interest rates, inflation) affect this asset
    9. Risk Mitigation Strategies: Potential hedging or risk management approaches
    
    Provide a risk rating (Low/Medium/High) with justification.
    """
    response = agent.invoke(prompt)
    return {"risk_evaluation": response["output"]}

def generate_report(state: State):
    """Combines all investment insights into a final report"""
    asset = state["investment_asset"]

    prompt = f"""
    Create a comprehensive investment report for {asset} by synthesizing:
    
    Fundamental Analysis: {state['fundamental_insights']}

    Technical Analysis: {state['technical_insights']}
    
    Sentiment Analysis: {state['sentiment_insights']}
    
    Risk Assessment: {state['risk_evaluation']}
    
    Based on the above analyses, provide:
    1. Investment Thesis: Core reasoning for bullish or bearish outlook
    2. Key Strengths: Most compelling reasons to invest
    3. Key Concerns: Most significant risks or red flags
    4. Time Horizon: Suitable investment timeframe (short, medium, long-term)
    5. Price Targets: Potential upside and downside scenarios with percentages
    6. Final Recommendation: Clear buy/hold/sell recommendation with confidence level
    7. Suggested Position Sizing: Based on the risk profile of this investment
    8. Alternative Investments: Similar assets that might be worth considering

    Format as a clean, professional investment report with clear sections.
    """
    response = agent.invoke(prompt)
    return {"final_report": response["output"]}

def asset(state):
    return state

#Define State Graph
builder = StateGraph(State)


#Add Nodes
builder.add_node("Investment Asset", asset)
builder.add_node("Fundamental Analysis", fundamental_analysis)
builder.add_node("Technical Analysis", technical_analysis)
builder.add_node("Sentiment Analysis", sentiment_analysis)
builder.add_node("Risk Assessment", risk_assessment)
builder.add_node("Generate Report", generate_report)

#Connect Nodes With Edges
builder.add_edge(START, "Investment Asset")
builder.add_edge("Investment Asset", "Fundamental Analysis")
builder.add_edge("Investment Asset", "Technical Analysis")
builder.add_edge("Investment Asset", "Sentiment Analysis")
builder.add_edge("Investment Asset", "Risk Assessment")

builder.add_edge("Fundamental Analysis", "Generate Report")
builder.add_edge("Technical Analysis", "Generate Report")
builder.add_edge("Sentiment Analysis", "Generate Report")
builder.add_edge("Risk Assessment", "Generate Report")

builder.add_edge("Generate Report", END)

#Complete the Workflow
graph = builder.compile()

# Streamlit UI
st.title("AI-Powered Investment Analyzer")
st.write("Analyze stocks, crypto, or real estate assets using AI-driven technical, fundamental, sentiment, and risk analysis.")

# User Input
asset_text = st.text_input("Enter an asset (e.g., Tesla, Nvidia, Apple, Bitcoin, NYC Real Estate)")
with st.sidebar:
    st.subheader("Workflow Diagram")

    # ✅ Generate Mermaid Workflow Diagram
    mermaid_diagram = graph.get_graph().draw_mermaid_png()

    # ✅ Save and Display the Image in Sidebar
    image_path = "workflow_diagram.png"
    with open(image_path, "wb") as f:
        f.write(mermaid_diagram)

    st.image(image_path, caption="Workflow Execution")


if st.button("Analyze Investment"):
    if asset_text:
        state = graph.invoke({"investment_asset": asset_text})
            
        with st.expander("💰 Fundamental Analysis", expanded=False):
            st.markdown(state["fundamental_insights"])

        with st.expander("📊 Technical Analysis", expanded=False):
            st.markdown(state["technical_insights"])
            
        with st.expander("📰 Sentiment Analysis", expanded=False):
            st.markdown(state["sentiment_insights"])
            
        with st.expander("⚠️ Risk Assessment", expanded=False):
            st.markdown(state["risk_evaluation"])

        # Display Investment Report
        st.subheader("Investment Report")
        st.markdown(state["final_report"])
    else:
        st.warning("Please enter an asset to analyze.")

    st.markdown("#### 🔗 Powered by LangGraph, Groq, Langchain ReAct Agents")

