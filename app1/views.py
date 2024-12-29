from django.shortcuts import render
from django.contrib import messages
from .forms import StockAnalysisForm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from alpha_vantage.timeseries import TimeSeries
from crewai import Agent
import logging
import os
# Set up logging
logger = logging.getLogger(__name__)
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')
# Authenticate Hugging Face
login(HUGGING_FACE_API_KEY)

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
except Exception as e:
    logger.error("Error loading Hugging Face model: %s", e)

# AlphaVantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')


# Initialize AlphaVantage API
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Define Agent classes
class DataAnalystAgent(Agent):
    def analyze_market_data(self, symbol="AAPL"):
        try:
            data, meta_data = ts.get_intraday(symbol=symbol, interval='5min', outputsize='full')
            return data
        except Exception as e:
            logger.error("Error fetching market data for %s: %s", symbol, e)
            raise

data_analyst = DataAnalystAgent(
    name="Data Analyst",
    description="Monitors and analyzes market data.",
    role="Monitor and analyze market data in real-time.",
    goal="Provide critical insights for trading decisions.",
    backstory="An expert in financial markets with a focus on statistical modeling."
)

class TradingStrategyDeveloper(Agent):
    def develop_strategy(self, market_data, risk_tolerance):
        try:
            market_data['SMA_10'] = market_data['4. close'].rolling(window=10).mean()
            market_data['SMA_50'] = market_data['4. close'].rolling(window=50).mean()
            market_data['Signal'] = np.where(market_data['SMA_10'] > market_data['SMA_50'], "Buy", "Sell")
            return market_data[['4. close', 'SMA_10', 'SMA_50', 'Signal']]
        except Exception as e:
            logger.error("Error developing strategy: %s", e)
            raise

strategy_developer = TradingStrategyDeveloper(
    name="Trading Strategy Developer",
    description="Develops and refines trading strategies.",
    role="Create profitable and risk-averse trading strategies.",
    goal="Optimize trading performance based on market insights.",
    backstory="A quantitative analyst with a strong background in financial markets."
)

class TradeAdvisor(Agent):
    def advise_trade(self, strategy_data):
        try:
            signals = strategy_data[strategy_data['Signal'] == "Buy"]
            return signals.tail(5)  # Example: Last 5 buy signals
        except Exception as e:
            logger.error("Error advising trade: %s", e)
            raise

trade_advisor = TradeAdvisor(
    name="Trade Advisor",
    description="Provides advice on trade execution.",
    role="Recommend optimal trade execution strategies.",
    goal="Ensure trades are executed efficiently.",
    backstory="An experienced trader focused on execution strategies."
)

class RiskAdvisor(Agent):
    def assess_risks(self, strategy_data):
        try:
            close_column = next((col for col in strategy_data.columns if 'close' in col.lower()), None)
            if not close_column:
                raise KeyError("Close price column not found in the data.")

            risk_metric = strategy_data[close_column].pct_change().std()
            return "High Risk" if risk_metric > 0.02 else "Low Risk"
        except Exception as e:
            logger.error("Error assessing risks: %s", e)
            raise

risk_advisor = RiskAdvisor(
    name="Risk Advisor",
    description="Evaluates trading risks.",
    role="Assess risks associated with trading strategies.",
    goal="Deliver comprehensive risk analysis.",
    backstory="A risk management specialist with a deep understanding of market dynamics."
)

class LLMResponder:
    """Generates conversational responses using an LLM model from Hugging Face."""
    def generate_response(self, prompt):
        try:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # Set eos_token as pad_token

            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(inputs['input_ids'], max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=inputs['attention_mask'])
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error("Error generating response: %s", e)
            raise

llm_responder = LLMResponder()

def analyze_and_trade(symbol, risk_tolerance):
    try:
        market_data = data_analyst.analyze_market_data(symbol)
        strategy = strategy_developer.develop_strategy(market_data, risk_tolerance)
        trade_signals = trade_advisor.advise_trade(strategy)
        risk_report = risk_advisor.assess_risks(strategy)
        
        llm_prompts = {
            "trade_advice": f"Stock: {symbol}\nCurrent Price: {strategy['4. close'].iloc[-1]:.2f}\n"
                            f"Recommended trade signal: {trade_signals['Signal'].iloc[-1]}.\n"
                            f"10-period SMA: {strategy['SMA_10'].iloc[-1]:.2f}, 50-period SMA: {strategy['SMA_50'].iloc[-1]:.2f}.",
            "risk_assessment": f"Risk Assessment for {symbol} (Risk tolerance: {risk_tolerance}): {risk_report}"
        }

        trade_advice_response = llm_responder.generate_response(llm_prompts["trade_advice"])
        risk_assessment_response = llm_responder.generate_response(llm_prompts["risk_assessment"])

        return trade_advice_response, risk_assessment_response
    except Exception as e:
        logger.error("Error in analyze_and_trade: %s", e)
        return "Error generating trade advice.", "Error generating risk assessment."

def stock_analysis_view(request):
    if request.method == "POST":
        form = StockAnalysisForm(request.POST)
        if form.is_valid():
            try:
                symbol = form.cleaned_data["symbol"]
                risk_tolerance = form.cleaned_data["risk_tolerance"]
                
                trade_advice, risk_assessment = analyze_and_trade(symbol, risk_tolerance)
                
                return render(request, "results.html", {
                    "trade_advice": trade_advice,
                    "risk_assessment": risk_assessment
                })
            except Exception as e:
                # Log the error for debugging
                logger.error("Error processing form: %s", e)
                messages.error(request, "An error occurred during analysis. Please try again.")
                return render(request, "index.html", {"form": form})
    else:
        form = StockAnalysisForm()
    
    return render(request, "index.html", {"form": form})
