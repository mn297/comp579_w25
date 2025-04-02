#!/usr/bin/env python3
"""
Stock Screener

This script implements a stock screening strategy that:
1. Identifies stocks that have just crossed above their 200-day moving average
2. Checks for unusually high trading volume
3. Looks for news catalysts
4. Applies technical indicators for additional buy signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import os
from tabulate import tabulate

class StockScreener:
    def __init__(self, stock_universe=None, volume_threshold=1.5, days_to_check=2):
        """
        Initialize the stock screener.
        
        Parameters:
        -----------
        stock_universe : list or None
            List of stock tickers to screen. If None, uses S&P 500 stocks.
        volume_threshold : float
            Minimum volume ratio (compared to average) to consider 'high volume'
        days_to_check : int
            Number of recent days to check for MA crossover
        """
        self.volume_threshold = volume_threshold
        self.days_to_check = days_to_check
        
        if stock_universe is None:
            # Default to S&P 500 stocks
            print("No stock universe provided. Using S&P 500 stocks...")
            self.stock_universe = self._get_sp500_tickers()
        else:
            self.stock_universe = stock_universe
            
        print(f"Initialized stock screener with {len(self.stock_universe)} stocks.")
        
    def _get_sp500_tickers(self):
        """Get list of S&P 500 tickers."""
        try:
            # Retrieve S&P 500 constituents from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable sortable'})
            
            tickers = []
            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text.strip()
                tickers.append(ticker)
            
            return tickers
        except Exception as e:
            print(f"Error retrieving S&P 500 tickers: {e}")
            # Fallback to a small sample of popular stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'NFLX']
    
    def _get_stock_data(self, ticker, lookback_days=365):
        """Fetch historical data for a ticker."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 210:  # Need at least 210 days for 200-day MA
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _check_ma_crossover(self, data):
        """Check if stock has crossed above its 200-day moving average recently."""
        if data is None or len(data) < 210:
            return False, None
        
        # Calculate 200-day moving average
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        
        # Check for crossover in the last few days
        recent_data = data.iloc[-self.days_to_check-5:].copy()  # Get a few extra days for comparison
        
        # Check if the close crossed above the MA
        for i in range(1, min(self.days_to_check+1, len(recent_data))):
            current_idx = -i
            prev_idx = -i-1
            
            if (recent_data['Close'].iloc[prev_idx] <= recent_data['MA_200'].iloc[prev_idx] and 
                recent_data['Close'].iloc[current_idx] > recent_data['MA_200'].iloc[current_idx]):
                # Return True and the crossing date
                return True, recent_data.index[current_idx].date()
        
        return False, None
    
    def _check_high_volume(self, data, crossover_date):
        """Check if volume was unusually high on crossover date."""
        if data is None or crossover_date is None:
            return False, 0
        
        # Convert crossover_date to the correct format if needed
        if isinstance(crossover_date, str):
            crossover_date = datetime.strptime(crossover_date, '%Y-%m-%d').date()
        
        # Find the row corresponding to the crossover date
        crossover_data = None
        for idx, date in enumerate(data.index.date):
            if date == crossover_date:
                crossover_data = data.iloc[idx]
                break
        
        if crossover_data is None:
            return False, 0
        
        # Calculate average volume over the last 50 days (approximately 2 months of trading)
        avg_volume = data['Volume'].iloc[-50:].mean()
        crossover_volume = crossover_data['Volume']
        volume_ratio = crossover_volume / avg_volume
        
        # Check if volume is higher than the threshold
        is_high_volume = volume_ratio > self.volume_threshold
        
        return is_high_volume, volume_ratio
    
    def _get_news_catalysts(self, ticker, date):
        """Look for news catalysts around the given date."""
        if date is None:
            return []
        
        # Convert date to string if it's a datetime object
        if not isinstance(date, str):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
        
        # Get news from Yahoo Finance
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Filter news around the date (±1 day)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            start_date = (date_obj - timedelta(days=1)).timestamp()
            end_date = (date_obj + timedelta(days=1)).timestamp()
            
            relevant_news = []
            for item in news:
                if start_date <= item.get('providerPublishTime', 0) <= end_date:
                    relevant_news.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'publisher': item.get('publisher', ''),
                        'type': item.get('type', '')
                    })
            
            return relevant_news
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    def _calculate_technical_indicators(self, data):
        """Calculate technical indicators for buy signals."""
        if data is None or len(data) < 50:
            return {}
        
        indicators = {}
        
        # 1. RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['RSI'] = rsi.iloc[-1]
        
        # 2. MACD (Moving Average Convergence Divergence)
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators['MACD'] = macd.iloc[-1]
        indicators['MACD_Signal'] = signal.iloc[-1]
        indicators['MACD_Histogram'] = macd.iloc[-1] - signal.iloc[-1]
        
        # 3. Bollinger Bands
        sma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        indicators['SMA_20'] = sma20.iloc[-1]
        indicators['BB_Upper'] = upper_band.iloc[-1]
        indicators['BB_Lower'] = lower_band.iloc[-1]
        indicators['BB_Width'] = (upper_band.iloc[-1] - lower_band.iloc[-1]) / sma20.iloc[-1]
        
        # 4. Stochastic Oscillator
        lowest_low = data['Low'].rolling(window=14).min()
        highest_high = data['High'].rolling(window=14).max()
        k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        indicators['Stochastic_K'] = k.iloc[-1]
        indicators['Stochastic_D'] = k.rolling(window=3).mean().iloc[-1]
        
        return indicators
    
    def _evaluate_buy_signals(self, indicators):
        """Evaluate technical indicators for buy signals."""
        if not indicators:
            return [], 0
        
        buy_signals = []
        signal_score = 0
        
        # RSI Buy Signal (oversold conditions)
        if 30 <= indicators['RSI'] <= 50:
            buy_signals.append("RSI indicates potential upward momentum (exiting oversold)")
            signal_score += 1
        
        # MACD Buy Signal (MACD line crossing above signal line)
        if indicators['MACD_Histogram'] > 0 and indicators['MACD_Histogram'] < 0.5:
            buy_signals.append("MACD recently crossed above signal line (bullish)")
            signal_score += 1
        
        # Bollinger Bands Buy Signal (price near lower band)
        current_price = indicators.get('current_price', 0)
        if current_price > indicators['BB_Lower'] and current_price < indicators['SMA_20']:
            buy_signals.append("Price bouncing off lower Bollinger Band (potential reversal)")
            signal_score += 1
        
        # Stochastic Oscillator Buy Signal
        if 20 <= indicators['Stochastic_K'] <= 50 and indicators['Stochastic_K'] > indicators['Stochastic_D']:
            buy_signals.append("Stochastic K-line crossing above D-line in low range (bullish)")
            signal_score += 1
        
        return buy_signals, signal_score
    
    def scan(self, max_stocks=None, save_results=True):
        """
        Scan for stocks matching the strategy criteria.
        
        Parameters:
        -----------
        max_stocks : int or None
            Maximum number of stocks to scan (for quicker testing)
        save_results : bool
            Whether to save results to a CSV file
        
        Returns:
        --------
        list
            List of dictionaries with screening results
        """
        results = []
        tickers_to_scan = self.stock_universe[:max_stocks] if max_stocks else self.stock_universe
        
        print(f"Scanning {len(tickers_to_scan)} stocks...")
        
        for i, ticker in enumerate(tickers_to_scan):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(tickers_to_scan)} stocks scanned")
            
            # Fetch historical data
            data = self._get_stock_data(ticker)
            if data is None:
                continue
            
            # Check for MA crossover
            has_crossover, crossover_date = self._check_ma_crossover(data)
            if not has_crossover:
                continue
            
            # Check for high volume
            high_volume, volume_ratio = self._check_high_volume(data, crossover_date)
            
            # Get news catalysts
            news = self._get_news_catalysts(ticker, crossover_date)
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(data)
            if indicators:
                indicators['current_price'] = data['Close'].iloc[-1]
            
            # Evaluate buy signals
            buy_signals, signal_score = self._evaluate_buy_signals(indicators)
            
            # Record the result if there's high volume or news catalysts or strong buy signals
            if high_volume or news or signal_score >= 2:
                result = {
                    'ticker': ticker,
                    'price': round(data['Close'].iloc[-1], 2),
                    'ma_crossover_date': crossover_date,
                    'high_volume': high_volume,
                    'volume_ratio': round(volume_ratio, 2),
                    'news_count': len(news),
                    'news': news[:3],  # Limit to first 3 news items
                    'buy_signals': buy_signals,
                    'signal_score': signal_score,
                    'indicators': {k: round(v, 2) if isinstance(v, (int, float)) else v 
                                   for k, v in indicators.items()}
                }
                results.append(result)
        
        print(f"Found {len(results)} stocks matching criteria.")
        
        if save_results and results:
            self._save_results(results)
            
        return results
    
    def _save_results(self, results):
        """Save results to CSV file."""
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"stock_screening_results_{today}.csv"
        
        # Convert the nested structure to a flat DataFrame
        rows = []
        for result in results:
            row = {
                'Ticker': result['ticker'],
                'Price': result['price'],
                'Crossover Date': result['ma_crossover_date'],
                'High Volume': result['high_volume'],
                'Volume Ratio': result['volume_ratio'],
                'News Count': result['news_count'],
                'Signal Score': result['signal_score'],
                'RSI': result['indicators'].get('RSI', ''),
                'MACD': result['indicators'].get('MACD', ''),
                'Buy Signals': ', '.join(result['buy_signals'])
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def display_results(self, results, limit=10):
        """Display results in a readable format."""
        if not results:
            print("No stocks matching criteria found.")
            return
        
        # Create a simplified table for display
        display_data = []
        for r in results[:limit]:
            news_titles = '; '.join([n['title'][:30] + '...' for n in r['news'][:2]]) if r['news'] else 'No recent news'
            signals = '; '.join(r['buy_signals'][:2]) if r['buy_signals'] else 'No buy signals'
            
            display_data.append([
                r['ticker'], 
                r['price'],
                r['ma_crossover_date'],
                'Yes' if r['high_volume'] else 'No',
                r['volume_ratio'],
                r['signal_score'],
                news_titles,
                signals
            ])
        
        headers = ['Ticker', 'Price', 'MA Crossover', 'High Volume', 'Vol Ratio', 'Signal Score', 'Recent News', 'Buy Signals']
        print(tabulate(display_data, headers=headers, tablefmt='grid'))
        
        if len(results) > limit:
            print(f"\nShowing {limit} of {len(results)} results. Check the CSV file for complete results.")

def main():
    """Main function to run the stock screener."""
    # Create a stock screener instance
    print("Initializing stock screener...")
    screener = StockScreener(volume_threshold=1.8, days_to_check=5)
    
    # For testing with a smaller subset of stocks
    test_mode = input("Run in test mode with fewer stocks? (y/n): ").lower() == 'y'
    max_stocks = 20 if test_mode else None
    
    # Run the screener
    print("Running stock screen. This may take a while...")
    results = screener.scan(max_stocks=max_stocks)
    
    # Display results
    print("\nScreening Results:")
    screener.display_results(results)
    
    # Plot charts for top results
    plot_charts = input("Plot charts for top results? (y/n): ").lower() == 'y'
    if plot_charts and results:
        for result in results[:3]:  # Plot for top 3 results
            ticker = result['ticker']
            print(f"Plotting chart for {ticker}...")
            data = screener._get_stock_data(ticker)
            if data is not None:
                plot_stock_chart(ticker, data)

def plot_stock_chart(ticker, data):
    """Plot a stock chart with MA and volume."""
    # Calculate moving averages
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and MAs
    ax1.plot(data.index, data['Close'], label='Close Price')
    ax1.plot(data.index, data['MA_50'], label='50-day MA', linestyle='--')
    ax1.plot(data.index, data['MA_200'], label='200-day MA', linestyle='-.')
    
    # Add volume subplot
    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.5)
    ax2.set_ylabel('Volume')
    
    # Set labels and title
    ax1.set_title(f'{ticker} Price and Volume')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Set x-axis limits to be the same for both subplots
    ax1.set_xlim(data.index[0], data.index[-1])
    ax2.set_xlim(data.index[0], data.index[-1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
