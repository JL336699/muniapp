import streamlit as st
import pandas as pd
import numpy as np

# Correct file paths
data_path = r"C:\Users\joseph.lapsley\muni_app\static\MuniData.xlsx"

# Function to load data with error handling
@st.cache_data
def load_data():
    try:
        # Try to load data
        data = pd.read_excel(data_path, sheet_name='spreads', header=0, index_col=0)
        returns = pd.read_excel(data_path, sheet_name='totalreturns', header=0, index_col=0)
        data.index = pd.to_datetime(data.index)
        return data, returns
    except ImportError as e:
        st.error(f"ImportError: {e}. Please install the required dependencies, e.g., openpyxl.")
        return None, None
    except FileNotFoundError as e:
        st.error(f"FileNotFoundError: {e}. Please check the file path.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

data, returns = load_data()

# Stop the app if data is not loaded
if data is None or returns is None:
    st.stop()

# Convert data to numpy array
spreads = data.to_numpy()

# Parameters
lb = 252
rebal = [1, 2, 3, 4, 5, 6, 9, 12]
nb_constituents = 5

# Streamlit app title
st.title('Muni Rotation Strategy')

# Compute the strategies
def compute_strategies():
    ts_long = np.zeros((data.shape[0], len(rebal)))
    ts_short = np.zeros((data.shape[0], len(rebal)))
    top_positions_summary = {}
    
    for i in range(len(rebal)):
        dum = 0
        dum_rebal = pd.DataFrame(np.zeros((rebal[i], data.shape[1])), columns=data.columns)
        rebal_signal_long = pd.DataFrame(np.zeros(returns.shape), columns=data.columns, index=data.index)
        rebal_signal_short = pd.DataFrame(np.zeros(returns.shape), columns=data.columns, index=data.index)
        
        for t in range(260, data.shape[0]):
            if data.index[t-1].month != data.index[t].month:
                spreads_change = pd.DataFrame(np.empty((1, data.shape[1])), columns=data.columns)
                spreads_change[:] = np.nan
                for n in range(data.shape[1]):
                    spreads_change.iloc[0, n] = spreads[t-2, n] - np.mean(spreads[t-lb-2:t-2, n])
                
                test = spreads_change.dropna(axis=1)
                ind = np.array(test).argsort(axis=1)
                columns_array = test.columns.to_numpy()
                indLong = columns_array[ind[:, -nb_constituents:]].flatten()
                indShort = columns_array[ind[:, :nb_constituents]].flatten()
                
                if dum < rebal[i]:
                    dum_rebal.loc[dum, :] = 0
                    dum_rebal.loc[dum, dum_rebal.columns.intersection(indLong.tolist())] = 1 / nb_constituents / rebal[i]
                    dum_rebal.loc[dum, dum_rebal.columns.intersection(indShort.tolist())] = -1 / nb_constituents / rebal[i]
                else:
                    dum = 0
                    dum_rebal.loc[dum, :] = 0
                    dum_rebal.loc[dum, dum_rebal.columns.intersection(indLong.tolist())] = 1 / nb_constituents / rebal[i]
                    dum_rebal.loc[dum, dum_rebal.columns.intersection(indShort.tolist())] = -1 / nb_constituents / rebal[i]
                
                rebal_signal_long.loc[data.index[t], :] = dum_rebal[dum_rebal > 0].sum() * (1 / np.sum(dum_rebal[dum_rebal > 0].sum()))
                rebal_signal_short.loc[data.index[t], :] = dum_rebal[dum_rebal < 0].sum() * (1 / np.sum(dum_rebal[dum_rebal > 0].sum()))
                dum = dum + 1
            else:
                rebal_signal_long.iloc[t, :] = rebal_signal_long.iloc[t-1, :]
                rebal_signal_short.iloc[t, :] = rebal_signal_short.iloc[t-1, :]
        
        ts_long[:, i] = np.cumsum(np.nansum(np.array(returns) * np.array(rebal_signal_long[rebal_signal_long > 0]) - \
                                           np.vstack((np.zeros((1, data.shape[1])), np.abs(np.array(rebal_signal_long.iloc[:-1, :]) - np.array(rebal_signal_long.iloc[1:, :])) * 0.1)), axis=1))
        ts_short[:, i] = np.cumsum(np.nansum(np.array(returns) * np.array(-rebal_signal_short[rebal_signal_short < 0]) - \
                                            np.vstack((np.zeros((1, data.shape[1])), np.abs(np.array(rebal_signal_short.iloc[:-1, :]) - np.array(rebal_signal_short.iloc[1:, :])) * 0.1)), axis=1))
        
        top_longs, top_shorts = get_top_10_positions(rebal_signal_long, rebal_signal_short, nb_constituents)
        top_positions_summary[f'Rebal {rebal[i]} Months'] = {
            'Top Longs': top_longs,
            'Top Shorts': top_shorts
        }
    
    return ts_long, ts_short, top_positions_summary

# Function to get the top 10 longs and shorts
def get_top_10_positions(rebal_signal_long, rebal_signal_short, nb_constituents=10):
    avg_long_positions = rebal_signal_long.mean(axis=0)
    avg_short_positions = rebal_signal_short.mean(axis=0)
    
    top_longs = avg_long_positions.nlargest(nb_constituents)
    top_shorts = avg_short_positions.nsmallest(nb_constituents)
    
    return top_longs, top_shorts

# Run the computation
ts_long, ts_short, top_positions_summary = compute_strategies()

# Visualize results
st.subheader('Cumulative Returns of Long Strategies')
for i in range(len(rebal)):
    st.line_chart(ts_long[:, i])

st.subheader('Cumulative Returns of Short Strategies')
for i in range(len(rebal)):
    st.line_chart(ts_short[:, i])

st.subheader('Top 10 Long and Short Positions for Each Strategy')
for strategy, positions in top_positions_summary.items():
    st.write(f"**Strategy: {strategy}**")
    st.write("**Top 10 Longs**")
    st.table(positions['Top Longs'].to_frame('Average Position Size'))
    st.write("**Top 10 Shorts**")
    st.table(positions['Top Shorts'].to_frame('Average Position Size'))
    st.write("\n" + "-"*50)
