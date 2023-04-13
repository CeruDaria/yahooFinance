# Initializing
import pandas as pd
import numpy as np
import datetime as dtm
import yfinance as yf
import streamlit as st
import requests
import re
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_icon="🍞", layout="wide")

# Initializing Session State
if "tickerList" not in st.session_state:
  tickerList = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0][["Symbol","Security"]]
  # There is a small inconsistency in the Wikipedia table where certain tickers are registered differently from how they
  # are in Yahoo Finance (e.g. BRK-B is written as BRK.B in the Wikipedia table)
  tickerList["Symbol"] = tickerList["Symbol"].apply(lambda x: re.sub(r"\.", r"-", x))
  tickerList["concated"] = tickerList.apply(lambda x: f"{x['Symbol']} — {x['Security']}", axis = 1)
  st.session_state["tickerList"] = tickerList
if "currentTicker" not in st.session_state:
  st.session_state["currentTicker"] = None
if "currentLoadedTicker" not in st.session_state:
  st.session_state["currentLoadedTicker"] = None
if "currentSumChartOptions" not in st.session_state:
  st.session_state["currentSumChartOptions"] = "6M"
if "currentBigChartOptions" not in st.session_state:
  st.session_state["currentBigChartOptions"] = "6M"
if "currentBigChartDateBool" not in st.session_state:
  st.session_state["currentBigChartDateBool"] = False
if "reportType" not in st.session_state:
  st.session_state["reportType"] = "financials"
if "reportPeriod" not in st.session_state:
  st.session_state["reportPeriod"] = "annual"
  
# Initializing a class to house the entire ticker. This way it is easier to control/track what info is used for the dashboard
class DashboardObject:
  def __init__(self, ticker_str):
    self.ticker                 = yf.Ticker(ticker_str)
    self.tickerinfo             = dict(self.ticker.info)
    self.tickercal              = self.ticker.calendar
    self.major_holders          = self.ticker.major_holders
    self.major_holders          = self.major_holders.rename(columns={0: "Statistics", 1: "Description"})
    self.institutional_holders  = self.ticker.institutional_holders
    self.name                   = self.tickerinfo["longName"]
    self.cur_price              = self.tickerinfo["currentPrice"]
    self.prv_price              = self.tickerinfo["previousClose"]
    self.logo                   = self.tickerinfo["logo_url"]
    self.symbol                 = self.tickerinfo["symbol"]
    self.longDesc               = f"""
*Headquartered at {self.tickerinfo['address1']}, {self.tickerinfo['city']}, {self.tickerinfo['country']}*

{self.tickerinfo['longBusinessSummary']}

**Visit their website [here]({self.tickerinfo["website"]} '{self.tickerinfo["shortName"]}')**
    """
    self.sumcol1 = {
      "Previous Close"    : self.get_info("previousClose"),
      "Open"              : self.get_info("open"),
      "Bid"               : f"{self.get_info('bid')} x {self.get_info('bidSize', False, False)}",
      "Ask"               : f"{self.get_info('ask')} x {self.get_info('askSize', False, False)}",
      "Day's Range"       : f"{self.get_info('dayLow')} - {self.get_info('dayHigh')}",
      "52 Week Range"     : f"{self.get_info('fiftyTwoWeekLow')} - {self.get_info('fiftyTwoWeekHigh')}",
      "Volume"            : self.get_info("volume", dec=False), 
      "Avg. Volume"       : self.get_info("averageVolume", dec=False),
    }
    self.sumcol2 = {
      "Market Cap"              : self.get_info("marketCap", dec=False),
      "Beta (5Y Monthly)"       : self.get_info("beta"),
      "PE Ratio (TTM)"          : self.get_info("trailingPE"),
      "EPS (TTM)"               : self.get_info("trailingEps"),
      "Earnings Date"           : self.get_calinfo(),
      "Forward Dividend & Yield": f"{self.get_info('dividendRate')} ({self.get_info('dividendYield', pct=True)})",
      "Ex-Dividend Date"        : self.get_info("exDividendDate"), 
      "1y Target Est"           : self.get_info("targetMeanPrice", False, False),
    }

  def get_info(self, info, format=True, dec=True, pct=False):
    try:
      cur_info = self.tickerinfo[info]
      if info == "exDividendDate":
        return dtm.datetime.strftime(dtm.datetime.fromtimestamp(cur_info), '%b %d, %Y')
      if pct:
        return f"{cur_info:.2%}"
      elif dec:
        return f"{cur_info:,.2f}"
      elif format:
        return f"{cur_info:,}"
      else:
        return f"{cur_info}"
    except:
      return "N/A"
    
  def get_calinfo(self):
    try:
      start_date = "N/A" if self.tickercal.loc['Earnings Date', 0] == None else self.tickercal.loc['Earnings Date', 0]
      end_date = "N/A" if self.tickercal.loc['Earnings Date', 1] == None else self.tickercal.loc['Earnings Date', 1]

      result = f"{dtm.datetime.strftime(start_date, '%b %d, %Y')} - {dtm.datetime.strftime(end_date, '%b %d, %Y')}"
    except:
      result = "N/A - N/A"

    return result

  def showChart(self, period="6mo", interval="1d", type="Area", movingAvg=False, startDate=None, endDate=None):
    if interval == "1y":
      itv = "1d"
    else:
      itv = interval

    # Auto-adjust is set to false to better match historical data on Yahoo Finance. For example, Yahoo Finance reports much
    # higher prices in 1983 for ticker AOS compared to the adjusted prices
    if startDate:
      df = self.ticker.history(start=startDate, end=endDate, interval=itv, auto_adjust=False, actions=False)
    else:
      # Formatting input argument
      period = period.lower() + "o" if period[-1] == "M" else period.lower()
      # Getting historical data
      df = self.ticker.history(period=period, interval=itv, auto_adjust=False, actions=False)
    # Setting y-axis's range for better display
    y1minRange = df["Close"].min() - df["Close"].min() * .1
    y1maxRange = df["Close"].max() + df["Close"].max() * .1
    y2maxRange = df["Volume"].max() * 2.5

    # If the big chart is used, defaults to always get the maximum amount of data to draw continous moving average line,
    # get the correct colors for bar and candle charts. This also emulates the same chart setup on Yahoo Finance (which is
    # a zoomed in portion of the full max-chart)
    xminrange = df.index.min()
    xmaxrange = df.index.max()
    if type != "Area":
      df = self.ticker.history(period="max", interval=itv, actions=False)

    if interval == "1y":
      df.reset_index(inplace=True)
      cutoffmonth = df["Date"].min().month

      df["Year"] = np.where(df["Date"].dt.month < cutoffmonth, df["Date"].dt.year - 1, df["Date"].dt.year)
      df = df.groupby("Year").agg(
          High=("High","max"), 
          Low=("Low","min"),
          Open=("Close", "first"),
          Close=("Close", "last"),
          Volume=("Volume", "sum")
      )
      df.reset_index(inplace=True)
      df["pct_change"] = df["Close"].pct_change()

      y1minRange = df["Close"].min() - df["Close"].min() * .1
      y1maxRange = df["Close"].max() + df["Close"].max() * .1
      y2maxRange = df["Volume"].max() * 2.5
      if startDate:
        start = pd.to_datetime(startDate, format="%Y-%m-%d").year
        end = pd.to_datetime(endDate, format="%Y-%m-%d").year
        try:
          xmaxrange = list(df["Year"]).index(end)
          xminrange = list(df["Year"]).index(start)     
        except:
          st.write("Range selected out of bound")
      elif period in ["3y", "5y"]:
        xmaxrange = df.shape[0] - 1
        xminrange = xmaxrange - int(period[0])
      elif period == "max":
        xmaxrange = df.shape[0] - 1
        xminrange = 0
      else:
        st.write("Period selected is too short for meaningful visuals")
        return
      
    else:
      df = df[df["Close"].isnull() == False]
      try:
        xminrange = list(df.index).index(xminrange)
      except:
        xminrange = list(df.index).index(df.index[df.index > xminrange][0])
      try:
        xmaxrange = list(df.index).index(xmaxrange)
      except:
        xmaxrange = list(df.index).index(df.index[df.index < xmaxrange][-1])
      # Categorical range.
      df.reset_index(inplace=True)
      df["pct_change"] = df["Close"].pct_change() 
      df[df.columns[0]] = df[df.columns[0]].dt.strftime("%d/%m/%Y")

    if type == "Area":
      pct_change = df["pct_change"] >= 0
      pct_change = pct_change.value_counts()
      color = "firebrick" if pct_change.loc[False,] > pct_change.loc[True,] else "forestgreen"
      fig_obj = go.Scatter(
          x=df[df.columns[0]], 
          y=df["Close"], 
          fill='tozeroy', 
          mode='lines',
          line=dict(width=0, color=color),
          name="Close Price"
          ) 
    elif type == "Line":
      fig_obj = go.Scatter(
          x=df[df.columns[0]], 
          y=df["Close"],
          mode='lines', 
          name="Close Price"
          )
    elif type == "Candle":
      fig_obj = go.Candlestick(
        x=df[df.columns[0]],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'])         
    else:
      print("Wrong chart type")
      return

    # Creating overlaying plots 
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if type == "Area":
      fig.add_trace( 
        go.Bar(
          x=df[df.columns[0]], 
          y=df["Volume"], 
          name="Volume"
          ) 
        )
    else:
      # This is done to plot alternating red-green bars as seen on Yahoo Finance
      df["volumeRed"] = np.where(df["pct_change"] < 0, df["Volume"], np.NaN)
      df["volumeGreen"] = np.where(df["pct_change"] >= 0, df["Volume"], np.NaN)
      fig.add_trace( 
        go.Bar(
          x=df[df.columns[0]], 
          y=df["volumeRed"], 
          name="Volume",
          marker={"color":"firebrick"}
          )
        )
      fig.add_trace( 
        go.Bar(
          x=df[df.columns[0]], 
          y=df["volumeGreen"], 
          name="Volume",
          marker={"color":"forestgreen"}
          )
        )

    fig.add_trace(fig_obj, secondary_y=True)
    if movingAvg:
      df["movingAvg"] = df["Close"].rolling(50).mean()
      fig.add_trace(
        go.Scatter(
          x=df[df.columns[0]], 
          y=df["movingAvg"], 
          mode='lines',
          line=dict(width=1, color="darkorchid"),
          name="Moving Average"
        )
        , secondary_y=True
      )
    
    fig.update_yaxes(
      visible=False, 
      range=[0, y2maxRange], 
      secondary_y=False
      )
    fig.update_yaxes(
      dtick=20, 
      showspikes=True, 
      spikemode="across", 
      range=[y1minRange, y1maxRange], 
      secondary_y=True
      )
    fig.update_xaxes(
      showspikes=True, 
      spikemode="across",
      type="category",
      # Getting 20 ticks
      dtick=(xmaxrange - xminrange) // 20,
      # To remove the ugly gaps with the 2 y-axes
      range=[0, df.shape[0] - 1]
      )
    fig.update_layout(
      showlegend=False,
      title={
        "text": f"Currently selected period: {period.upper()}",
        "x": 0.5,
        "xanchor": "center"
        }, 
      xaxis_rangeslider_visible=False,
      barmode="overlay"
      )
    
    if startDate:
      fig.update_layout(
        title = {
          "text": f"Currently selected period: {startDate} - {endDate}",
          }
        )

    # Continuing from above, this is to make sure the chart is zoomed in on the relevant timeframe when it's rendered
    if type != "Area":
      fig.update_xaxes(
        range=[xminrange, xmaxrange]
      )

    st.plotly_chart(fig, use_container_width=True)
  
  def showRec(self, type):
    mapping = {
      "Outperform": "Buy",
      "Overweight": "Buy",
      "Buy": "Strong Buy",
      "Neutral": "Hold",
      "Underweight": "Underperform",
      "Positive": "Buy",
      "Negative": "Underperform",
      "Equal-Weight": "Hold",
      "Market Outperform": "Buy",
      "Market Perform": "Hold",
      "Peer Perform": "Hold"
    }

    df = self.ticker.recommendations
    df = df[df.index >= (pd.Timestamp.today() - pd.Timedelta(365, "D")).strftime("%Y-%m-%d")].copy()
    df["To Grade"].replace(mapping, inplace=True)
    df = df.groupby([df.index.strftime("%Y/%m"), "To Grade"])["Firm"].agg(count="count")
    df.reset_index(inplace=True)

    df = df.pivot_table(
        values = "count",
        index = "Date",
        columns = "To Grade",
        fill_value = 0
    ).rename_axis(None,axis=1)

    df.reset_index(inplace=True)
    figs = []
    ratings = ['Sell', 'Underperform', 'Hold', 'Buy', 'Strong Buy']
    colors = ['orangered', 'orange', 'Gold', 'MediumSeaGreen', 'SeaGreen']

    fig = go.Figure()

    if type == "trends":
      for r, c in zip(ratings, colors):
        try:
          figs.append(go.Bar(x=df["Date"], y = df[r], marker={"color": c}, name= r))
        except:
          next
      for f in figs:
        fig.add_trace(f)
      fig.update_layout(
        barmode="stack"
      )
    elif type == "ratings":
      df = dict(df.apply(lambda x: x.sum(), axis=0))
      score = 0
      sum = 0
      for i, r in enumerate(ratings):
        try:
            score += df[r] * (i + 1) 
            sum += df[r]
        except:
            next
      score = score / sum - 1

      for r, c in zip(ratings, colors):
          figs.append(go.Bar(x=[r], y =[1], text=[r], marker={"color": c}, name= r))
      for f in figs:
          fig.add_trace(f)
          
      fig.update_layout(
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
        showlegend=False, 
        bargap = 0, 
        plot_bgcolor='rgba(0, 0, 0, 0)',
      )
      fig.update_yaxes(
        visible=False, 
        dtick=1, 
        range=[-20,20])
      fig.update_xaxes(visible=False)

      fig.add_annotation( 
        x=score, 
        y=5,
        text=f"On average, over the last year, <br> experts have been rating the stock as <br> <b>{ratings[int(score)]}</b> <br> with a score of <br> <b>{(score + 1):.2f}</b>",
        bgcolor=colors[int(score)],
        showarrow=True,
        arrowhead=5)
      
    st.plotly_chart(fig, use_container_width=True)

def showWelcomeScreen():
  col1, col2, col3 = st.columns([3, 4, 3])
  col1.write("")
  with col2:
    st.title("Your home to all things finance")
    selectBox = st.selectbox("Select your stock name here", [""] + list(st.session_state['tickerList']["concated"]))
    st.button("Start Exploring", on_click=updateLoadedTicker)
    # Interactive elements
    if selectBox:
      st.session_state["currentTicker"] = st.session_state["tickerList"].loc[st.session_state["tickerList"]["Symbol"] == selectBox.split(" — ")[0], "Symbol"].values[0]
  col3.write("")

def showTopbar():
  # UI elements
  with st.expander("Explore another company here"):
    col1, col2 = st.columns([10,1])
    current_idx = list(st.session_state["tickerList"]["Symbol"]).index(st.session_state["currentTicker"])
    selectBox = col1.selectbox("Select another ticker here", list(st.session_state["tickerList"]["concated"]), index = current_idx, label_visibility="collapsed")
    col2.button("Update", on_click=updateLoadedTicker)
    # Interactive elements
    if selectBox:
      st.session_state["currentTicker"] = st.session_state["tickerList"].loc[st.session_state["tickerList"]["Symbol"] == selectBox.split(" — ")[0], "Symbol"].values[0]
  
  col1, col2= st.columns([1,4])
  
  col1.markdown(f"## {st.session_state['currentLoadedTicker'].symbol} \n\n **{st.session_state['currentLoadedTicker'].name}**")
  with col2:
    cur_price = st.session_state['currentLoadedTicker'].cur_price
    prv_price = st.session_state['currentLoadedTicker'].prv_price
    st.metric("At close", f"{cur_price:.2f}", f"{(cur_price - prv_price):.2f} ({((cur_price - prv_price) / prv_price):.2%})")

def updateLoadedTicker():
  # If-else is set to prevent blank user input
  if st.session_state["currentTicker"]:
    st.session_state["currentLoadedTicker"] = DashboardObject(st.session_state["currentTicker"])
  else:
    st.write("Please select a ticker")

def get_annuals(ticker, type="financials", headers = {"User-agent": "Mozilla/5.0"}):
  url = f"https://finance.yahoo.com/quote/{ticker}/{type}?p={ticker}"
  html = requests.get(url=url, headers = headers).text

  json_str = html.split('root.App.main =')[1].split('(this)')[0].split(';\n}')[0].strip()
      
  data = json.loads(json_str)['context']['dispatcher']['stores']['QuoteTimeSeriesStore']['timeSeries']
  df = pd.DataFrame.from_dict(data, orient="index")
  sorted_cols = list(df.columns)
  sorted_cols.sort(reverse=True)
  df = df[sorted_cols]
  timestamps = df.loc["timestamp",]
  df = df.drop(index = "timestamp")
  df.columns = [dtm.datetime.strftime(dtm.datetime.fromtimestamp(t), "%m/%d/%Y") for t in timestamps]

  df = df[df.apply(lambda x: sum(x.isnull()), axis = 1) < 4]

  for col in df.columns:
      df[col] = df[col].apply(lambda x: "-" if x == None else f'{(x["reportedValue"]["raw"]/1000):,.3f}')
  df.index.name = "Breakdown"
  df.index = pd.Series(df.index).apply(lambda x: re.sub(r"(?<=[A-Z])([A-Z])(?=[a-z0-9])", r" \1", re.sub(r"([a-z0-9])(?=[A-Z])", r"\1 ", x))) 

  df_annual = df[df.index.str[0:8] != "trailing"]
  df_annual.index = df_annual.index.str[7:]

  if type != "balance-sheet":
      df_ttm = df[df.index.str[0:8] == "trailing"]
      df_ttm.index = df_ttm.index.str[9:]

      df_ttm = df_ttm[df_ttm.columns[-1]]
      df_ttm.name = "TTM"

      df = pd.merge(df_ttm, df_annual, how="outer", on="Breakdown")
      df = df.fillna("-")
  else:
      df = df_annual
  
  return df

def renderFinancials(type="financials", interval="annual"):
  tbl = None

  if interval == 'annual':
      tbl = get_annuals(st.session_state['currentTicker'], type)
  elif interval == 'quarterly':
    if type == "balance-sheet":
      tbl = st.session_state['currentLoadedTicker'].ticker.quarterly_balance_sheet
      sorted_cols = list(tbl.columns)
      sorted_cols.sort(reverse=True)
      tbl = tbl[sorted_cols]
      try:
        tbl.columns = tbl.columns.strftime("%m/%d/%Y")
      except:
        print("Problem with stringifying columns", st.session_state["reportType"], st.session_state["reportPeriod"])
    else:
      if type == "financials":
        tbl = st.session_state['currentLoadedTicker'].ticker.quarterly_financials
      elif type == "cash-flow":
        tbl = st.session_state['currentLoadedTicker'].ticker.quarterly_cashflow
      sorted_cols = list(tbl.columns)
      sorted_cols.sort(reverse=True)
      tbl = tbl[sorted_cols]
      try:
        tbl.columns = tbl.columns.strftime("%m/%d/%Y")
      except:
        print("Problem with stringifying columns", st.session_state["reportType"], st.session_state["reportPeriod"])
      # Getting only the last 4 quarters
      tbl["TTM"] = tbl[tbl.columns[:4]].apply(lambda x: x.sum(), axis=1)
      tbl = tbl[["TTM"] + list(tbl.columns[:-1])]
    for col in tbl.columns:
      tbl[col] = tbl[col].apply(lambda x: "-" if x == None or x == np.NaN or x == 0 else f'{(x/1000):,.3f}')
    
  else:
    print("Incorrect interval problem")
    return

  return tbl

def runMCS(n=200, days=30):
  np.random.seed(1994)
  today = pd.Timestamp.today().date()
  start = today - pd.Timedelta(days*3, "D")

  hist = st.session_state["currentLoadedTicker"].ticker.history(start=start, end=today, actions=False)
  close_price = hist["Close"]
  daily_return = close_price.pct_change()
  daily_volatility = np.std(daily_return)
  sims = pd.DataFrame()
  
  for i in range(n):
    last_price = close_price[-1]
    prices = []
    for j in range(days):
      future_return = np.random.normal(0, daily_volatility)
      future_price = last_price * (1 + future_return)
      prices += [future_price]
      last_price = future_price
    sims = pd.concat([sims, pd.Series(prices).rename('sim' + str(i))], axis=1)

  ending_price = sims.iloc[-1:, :].values[0, ]
  future_price_95ci = np.percentile(ending_price, 5)
  VaR = close_price[-1] - future_price_95ci
  
  st.line_chart(sims)
  st.markdown(f'Value at risk at 95% confidence interval is: {str(np.round(VaR, 2))} USD')

def updateDatebool():
  st.session_state["currentBigChartDateBool"] = True
  print("Updating")

def showFirstTab():
  # Info area
  with st.expander("Learn more about the company"):
    st.markdown(st.session_state["currentLoadedTicker"].longDesc)
  btn_lbls = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
  intervals = ["1d", "1d", "1d", "1wk", "1wk", "1mo", "1mo", "1mo"]

  cols = st.columns([len(btn_lbls)] + [1] * len(btn_lbls))

  # Button area
  buttons = {}
  for col, period in zip(cols[1:], btn_lbls):
    buttons[period] = col.button(label=period)

  table_disp = cols[0].selectbox("Select table to display", ["Summary table", "Major holders breakdown"], label_visibility="collapsed")
  
  if table_disp == "Summary table":
  # Table area
    cols = st.columns([1] * 4 + [4])
    for i, text, col in zip(range(4), [
      st.session_state["currentLoadedTicker"].sumcol1.keys(), 
      st.session_state["currentLoadedTicker"].sumcol1.values(), 
      st.session_state["currentLoadedTicker"].sumcol2.keys(), 
      st.session_state["currentLoadedTicker"].sumcol2.values()], cols[:4]):
      col.markdown("\n\n \n\n")
      for t in text:
        if i % 2 == 1:
          t = "**" + t + "**"
        col.markdown(t)
  # Holder area
  else:
    cols = st.columns(2)
    cols[0].dataframe(st.session_state["currentLoadedTicker"].major_holders)
    cols[0].dataframe(st.session_state["currentLoadedTicker"].institutional_holders.head(5))

  # Chart area
  with cols[-1]:
    for k, v in buttons.items():
      if v:
        st.session_state["currentSumChartOptions"] = k
    st.session_state["currentLoadedTicker"].showChart(
      st.session_state["currentSumChartOptions"], 
      intervals[btn_lbls.index(st.session_state["currentSumChartOptions"])]
      )

def showSecondTab():
  btn_lbls = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
  intervals = {
    "1 day"   : "1d",
    "1 month" : "1mo",
    "1 year"  : "1y"
  }
  plot_types = ["📈 Line", "🕯️ Candle"]

  nbr_btn = len(btn_lbls)
  cols = st.columns([nbr_btn / 2] + [1] * nbr_btn + [nbr_btn / 4] * 2)

  buttons = {}
  for col, period in zip(cols[1:9], btn_lbls):
    buttons[period] = col.button(label=period, key="bigChart"+period)
 
  today = pd.Timestamp.today().date()
  start = today - pd.Timedelta(30, "D")
  date_range = cols[0].date_input("StartDate", [start, today], label_visibility="collapsed", on_change=updateDatebool)
  chart_box = cols[-1].selectbox("Chart Type", plot_types, label_visibility="collapsed")
  interval_box = cols[-2].selectbox("Interval", list(intervals.keys()), label_visibility="collapsed")

  for k, v in buttons.items():
    if v:
      st.session_state["currentBigChartOptions"] = k
      st.session_state["currentBigChartDateBool"] = False

  if st.session_state["currentBigChartDateBool"] and len(date_range) > 1:
    st.session_state["currentLoadedTicker"].showChart(
      startDate=date_range[0],
      endDate=date_range[1],
      interval=intervals[interval_box],
      type=chart_box[(chart_box.index(" ") + 1):],
      movingAvg=True
    )
  else:
    st.session_state["currentLoadedTicker"].showChart(
      period=st.session_state["currentBigChartOptions"],
      interval=intervals[interval_box],
      type=chart_box[(chart_box.index(" ") + 1):],
      movingAvg=True
      )

def showThirdTab():
  report_types = ["Income Statement", "Balance Sheet", "Cash Flow"]
  states = ["financials", "balance-sheet", "cash-flow", "annual", "quarterly"]
  buttons = {}

  cols = st.columns([2]*5 + [2]*2)
  for col, report, state in zip(cols[0:3], report_types, states[0:3]):
    buttons[state] = col.button(label=report)
  buttons[states[-1]] = cols[-1].button(label="Quarterly")
  buttons[states[-2]] = cols[-2].button(label="Annual")

  for k, v in buttons.items():
    if v and k in states[0:3]:
      st.session_state["reportType"] = k
    elif v and k in states[-2:]:
      st.session_state["reportPeriod"] = k
  
  st.markdown(f'## {st.session_state["reportPeriod"].capitalize()} {report_types[states.index(st.session_state["reportType"])]}')
  st.markdown(f"*All numbers in thousands*")
  tbl = renderFinancials(st.session_state["reportType"], st.session_state["reportPeriod"])
  st.dataframe(tbl, use_container_width=True)

def showFourthTab():
  col1, col2= st.columns(2)
  nbr_sim = col1.selectbox("Select the number of simulations", [200, 500, 1000])
  nbr_days = col2.selectbox("Select the number of days from today", [30, 60, 90])
  btn_mcs = st.button(label="Run Simulation")

  st.markdown(f"### Monte Carlo Simulation for the next {nbr_days} days from today")
  if btn_mcs:
    runMCS(n=nbr_sim, days=nbr_days)

def showFifthTab():
  col1, col2 = st.columns(2)
  with col1:
    st.markdown(f"""
## Recommendation Trends
Analysts and firms reveal their evaluations of the company's stock
    """)
    st.session_state['currentLoadedTicker'].showRec("trends")
  with col2:
    st.markdown("## Recommendation Ratings")
    st.session_state['currentLoadedTicker'].showRec("ratings")    


# FINAL
if st.session_state['currentLoadedTicker']:
  showTopbar()

  tabs = ["⭐ Summary", "📈 Chart", "💵 Financials", "💸 Monte Carlo Simulation", "☀️ Recommendations"]
  tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
  with tab1:
    showFirstTab()
  
  with tab2:
    showSecondTab()

  with tab3:
    showThirdTab()

  with tab4:
    showFourthTab()

  with tab5:
    showFifthTab()

else:
  showWelcomeScreen()
