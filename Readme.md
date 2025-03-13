# AI-Powered Investment Analyzer

Analyze stocks, cryptocurrencies, or real estate assets using AI-driven technical, fundamental, sentiment, and risk analysis.

## Features

- **Technical Analysis**: Moving Averages, RSI, MACD, Bollinger Bands, and more.
- **Fundamental Analysis**: Valuation Metrics, Financial Health, Growth Metrics, Profitability, Cash Flow Analysis, and more.
- **Sentiment Analysis**: News, Social Media, Analyst Opinions, Institutional Interest, and more.
- **Risk Assessment**: Market Risk, Volatility Metrics, Downside Risk, Correlation, Liquidity Risk, Regulatory/Legal Risks, and more.
- **Investment Report**: Comprehensive report combining all insights with a final recommendation.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/AI-Investment-Analyzer.git
    cd AI-Investment-Analyzer
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.streamlit/secrets.toml` file and add your API keys:
    ```toml

    GROQ_API_KEY = "your_groq_api_key_here"
    SERP_API_KEY = "your_serp_api_key_here"
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to the provided URL (usually `http://localhost:8501`).

3. Enter an asset (e.g., Tesla, Nvidia, Apple, Bitcoin, NYC Real Estate) and click "Analyze Investment".

4. View the analysis results and the comprehensive investment report.

## Dependencies

- `streamlit`
- `langchain`
- `python-dotenv`
- `langchain-openai`
- `langchain-core`
- `langchain-community`
- `langchain-huggingface`
- `langchain-groq`
- `langgraph`
- `google-search-results`