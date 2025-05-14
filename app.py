# import streamlit as st
# import sqlite3
# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import MinMaxScaler
#
#
# # =============================================
# # BACKGROUND IMAGE AND STYLING
# # =============================================
# import streamlit as st
# import base64
#
# def add_bg_image():
#     with open("img.jpeg", "rb") as image_file:
#         encoded = base64.b64encode(image_file.read()).decode()
#
#     st.markdown(
#         f"""
#         <style>
#         .main .block-container {{
#             background-color: rgba(255, 255, 255, 0.93);
#             border-radius: 10px;
#             padding: 2rem;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             margin-top: 2rem;
#             margin-bottom: 2rem;
#         }}
#
#         h1, h2, h3 {{
#             color: #2c3e50;
#         }}
#
#         .stButton>button {{
#             background-color: #3498db;
#             color: white;
#             border-radius: 5px;
#             padding: 0.5rem 1rem;
#             border: none;
#             transition: all 0.3s;
#         }}
#
#         .stButton>button:hover {{
#             background-color: #2980b9;
#             transform: translateY(-1px);
#             box-shadow: 0 2px 4px rgba(0,0,0,0.2);
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
#
# add_bg_image()
#
#
#
# # =============================================
# # DATABASE FUNCTIONS (UPDATED)
# # =============================================
# def get_db_connection():
#     return sqlite3.connect("users.db", check_same_thread=False)
#
#
# def init_db():
#     conn = get_db_connection()
#     c = conn.cursor()
#
#     try:
#         # Create users table if not exists
#         c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT UNIQUE, password TEXT)")
#
#         # Check if old predictions table exists
#         c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
#         table_exists = c.fetchone()
#
#         if table_exists:
#             # Check if timestamp column exists
#             c.execute("PRAGMA table_info(predictions)")
#             columns = [col[1] for col in c.fetchall()]
#             if 'timestamp' not in columns:
#                 # Backup old data
#                 c.execute("SELECT * FROM predictions")
#                 old_data = c.fetchall()
#
#                 # Drop old table
#                 c.execute("DROP TABLE predictions")
#
#                 # Create new table with timestamp
#                 c.execute("""
#                     CREATE TABLE predictions (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         username TEXT,
#                         stock_symbol TEXT,
#                         predicted_price REAL,
#                         prediction_date TEXT,
#                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#                     )
#                 """)
#
#                 # Restore data if any
#                 if old_data:
#                     for row in old_data:
#                         c.execute("""
#                             INSERT INTO predictions
#                             (username, stock_symbol, predicted_price, prediction_date)
#                             VALUES (?, ?, ?, ?)
#                         """, row[1:5])
#         else:
#             # Create new table if doesn't exist
#             c.execute("""
#                 CREATE TABLE predictions (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     username TEXT,
#                     stock_symbol TEXT,
#                     predicted_price REAL,
#                     prediction_date TEXT,
#                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#                 )
#             """)
#
#         conn.commit()
#     except sqlite3.Error as e:
#         st.error(f"Database initialization error: {e}")
#     finally:
#         c.close()
#         conn.close()
#
#
# def register_user(username, password):
#     conn = get_db_connection()
#     c = conn.cursor()
#     try:
#         c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
#         conn.commit()
#         return True
#     except sqlite3.IntegrityError:
#         st.warning("⚠ Username already exists! Please choose another.")
#         return False
#     except sqlite3.Error as e:
#         st.error(f"Registration error: {e}")
#         return False
#     finally:
#         c.close()
#         conn.close()
#
#
# def login_user(username, password):
#     conn = get_db_connection()
#     c = conn.cursor()
#     try:
#         c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
#         user = c.fetchone()
#         return user
#     except sqlite3.Error as e:
#         st.error(f"Login error: {e}")
#         return None
#     finally:
#         c.close()
#         conn.close()
#
#
# def get_predictions(username):
#     conn = get_db_connection()
#     c = conn.cursor()
#     try:
#         c.execute("""
#             SELECT stock_symbol, predicted_price, prediction_date
#             FROM predictions
#             WHERE username = ?
#             ORDER BY timestamp DESC
#         """, (username,))
#         return c.fetchall()
#     except sqlite3.Error as e:
#         st.error(f"Error fetching predictions: {e}")
#         return []
#     finally:
#         c.close()
#         conn.close()
#
#
# def save_prediction(username, stock_symbol, predicted_price, prediction_date):
#     conn = get_db_connection()
#     c = conn.cursor()
#     try:
#         c.execute("""
#             INSERT INTO predictions
#             (username, stock_symbol, predicted_price, prediction_date)
#             VALUES (?, ?, ?, ?)
#         """, (username, stock_symbol, predicted_price, prediction_date))
#         conn.commit()
#     except sqlite3.Error as e:
#         st.error(f"Error saving prediction: {e}")
#     finally:
#         c.close()
#         conn.close()
#
#
# # =============================================
# # APP INITIALIZATION
# # =============================================
# init_db()
#
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
#     st.session_state.username = ""
#     st.session_state.selected_stock = ""
#     st.session_state.prediction_data = None
#
# # =============================================
# # MAIN APP INTERFACE
# # =============================================
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import yfinance as yf
# from yahooquery import search
# st.title("📈 Stock Price Predictor")
#
# if not st.session_state.logged_in:
#     option = st.radio("Select an option", ["Sign In", "Create Account"], horizontal=True)
#
#     if option == "Create Account":
#         with st.form("create_account"):
#             new_user = st.text_input("Enter Username")
#             new_pass = st.text_input("Enter Password", type="password")
#             if st.form_submit_button("Create Account"):
#                 if new_user and new_pass:
#                     if register_user(new_user, new_pass):
#                         st.success("Account Created! Please Sign In.")
#                 else:
#                     st.warning("Please enter both username and password")
#
#     elif option == "Sign In":
#         with st.form("sign_in"):
#             username = st.text_input("Username")
#             password = st.text_input("Password", type="password")
#             if st.form_submit_button("Sign In"):
#                 user = login_user(username, password)
#                 if user:
#                     st.session_state.logged_in = True
#                     st.session_state.username = username
#                     st.success(f"👋 Welcome, {username}!")
#                     st.rerun()
#                 else:
#                     st.error("Invalid Credentials")
#
# else:
#     import yfinance as yf
#     from yahooquery import search
#     from tensorflow.keras.models import Sequential, load_model
#     from tensorflow.keras.layers import LSTM, Dense
#
#     st.subheader(f"👋 Welcome back, {st.session_state.username}!")
#
#     if st.button("🚪 Logout"):
#         st.session_state.logged_in = False
#         st.session_state.username = ""
#         st.session_state.selected_stock = ""
#         st.session_state.prediction_data = None
#         st.success("Logged out successfully!")
#         st.rerun()
#
#     with st.expander("🔍 Search for Stocks", expanded=True):
#         company_name = st.text_input("Enter Company Name", key="search_input")
#         if st.button("Search", key="search_btn"):
#             with st.spinner("🔍 Searching for companies..."):
#                 results = search(company_name)
#                 if "quotes" in results and results['quotes']:
#                     stock_options = {res['shortname']: res['symbol'] for res in results['quotes'] if
#                                      'symbol' in res and 'shortname' in res}
#                     if stock_options:
#                         selected_company = st.selectbox("Select Company", list(stock_options.keys()))
#                         if selected_company:
#                             st.session_state.selected_stock = stock_options[selected_company]
#                             st.success(f"✅ Selected: {selected_company} ({stock_options[selected_company]})")
#                     else:
#                         st.error("No valid companies found. Try another name.")
#                 else:
#                     st.error("No results found. Try another name.")
#
#     if st.session_state.selected_stock:
#         st.markdown(f"### 📊 Analyzing: **{st.session_state.selected_stock}**")
#         days_to_predict = st.slider("Select number of days to predict", min_value=1, max_value=30, value=7)
#
#         if st.button("Predict Future Prices", key="predict_btn", type="primary"):
#             stock_symbol = st.session_state.selected_stock
#
#             # Data fetching stage
#             with st.spinner("📥 Downloading historical stock data..."):
#                 try:
#                     stock_data = yf.download(stock_symbol, period="6mo", progress=False)
#                     if stock_data.empty:
#                         st.error("⚠ No data found. Please check the stock symbol.")
#                         st.stop()
#                 except Exception as e:
#                     st.error(f"❌ Download failed: {str(e)}")
#                     st.stop()
#
#             st.success("✅ Historical data retrieved successfully!")
#
#             # Data processing stage
#             with st.spinner("⚙️ Processing data for model..."):
#                 data = stock_data[['Close']].values
#                 scaler = MinMaxScaler(feature_range=(0, 1))
#                 data_scaled = scaler.fit_transform(data)
#
#                 X, y = [], []
#                 for i in range(60, len(data_scaled) - days_to_predict):
#                     X.append(data_scaled[i - 60:i, 0])
#                     y.append(data_scaled[i:i + days_to_predict, 0])
#                 X, y = np.array(X), np.array(y)
#                 X = X.reshape((X.shape[0], X.shape[1], 1))
#
#             # Model training/prediction stage
#             with st.spinner("🧠 Training model and making predictions..."):
#                 model = Sequential([
#                     LSTM(30, return_sequences=True, input_shape=(60, 1)),
#                     LSTM(30),
#                     Dense(days_to_predict)
#                 ])
#                 model.compile(optimizer='adam', loss='mean_squared_error')
#                 model.fit(X, y, epochs=5, batch_size=32, verbose=0)
#
#                 last_60_days = data_scaled[-60:].reshape(1, 60, 1)
#                 pred_scaled = model.predict(last_60_days)[0]
#                 predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
#                 predictions = predictions[:days_to_predict]
#
#                 # Store prediction data in session state
#                 prediction_df = pd.DataFrame(predictions, columns=["Predicted Price"])
#                 prediction_df.index = pd.date_range(
#                     start=pd.to_datetime(stock_data.index[-1]) + pd.Timedelta(days=1),
#                     periods=len(predictions),
#                     freq='B'
#                 )
#                 st.session_state.prediction_data = {
#                     'df': prediction_df,
#                     'last_price': data[-1][0],
#                     'predicted_price': predictions[-1][0],
#                     'stock_symbol': stock_symbol,
#                     'prediction_date': str(prediction_df.index[-1].date())
#                 }
#
#             # Save to database
#             save_prediction(
#                 st.session_state.username,
#                 stock_symbol,
#                 float(predictions[-1][0]),
#                 str(prediction_df.index[-1].date())
#             )
#
#             st.success("🎉 Prediction complete! Showing results below...")
#             st.rerun()
#
#         # Display current prediction if exists
#         if st.session_state.prediction_data:
#             st.line_chart(st.session_state.prediction_data['df'])
#
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Last Actual Price", f"₹{st.session_state.prediction_data['last_price']:.2f}")
#             with col2:
#                 st.metric(f"Predicted Price (Day {days_to_predict})",
#                           f"₹{st.session_state.prediction_data['predicted_price']:.2f}")
#
#         # Prediction history
#         if st.checkbox("📜 Show My Prediction History", key="history_checkbox"):
#             history = get_predictions(st.session_state.username)
#
#             if history:
#                 df = pd.DataFrame(history, columns=["Stock", "Predicted Price", "Date"])
#                 st.dataframe(
#                     df.style.format({
#                         "Predicted Price": "₹{:.2f}",
#                         "Date": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')
#                     }),
#                     height=400
#                 )
#             else:
#                 st.info("No prediction history found.")

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


# =============================================
# BACKGROUND IMAGE AND STYLING
# =============================================
def add_bg_image():
    with open("img.jpeg", "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.93);
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
            margin-bottom: 2rem;
        }}

        h1, h2, h3 {{
            color: #2c3e50;
        }}

        .stButton>button {{
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
        }}

        .stButton>button:hover {{
            background-color: #2980b9;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_image()


# =============================================
# DATABASE FUNCTIONS (UPDATED)
# =============================================
def get_db_connection():
    return sqlite3.connect("users.db", check_same_thread=False)


def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    try:
        # Create users table if not exists
        c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT UNIQUE, password TEXT)")

        # Check if old predictions table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        table_exists = c.fetchone()

        if table_exists:
            # Check if timestamp column exists
            c.execute("PRAGMA table_info(predictions)")
            columns = [col[1] for col in c.fetchall()]
            if 'timestamp' not in columns:
                # Backup old data
                c.execute("SELECT * FROM predictions")
                old_data = c.fetchall()

                # Drop old table
                c.execute("DROP TABLE predictions")

                # Create new table with timestamp
                c.execute("""
                    CREATE TABLE predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        stock_symbol TEXT,
                        predicted_price REAL,
                        prediction_date TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Restore data if any
                if old_data:
                    for row in old_data:
                        c.execute("""
                            INSERT INTO predictions 
                            (username, stock_symbol, predicted_price, prediction_date)
                            VALUES (?, ?, ?, ?)
                        """, row[1:5])
        else:
            # Create new table if doesn't exist
            c.execute("""
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    stock_symbol TEXT,
                    predicted_price REAL,
                    prediction_date TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")
    finally:
        c.close()
        conn.close()


def register_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.warning("⚠ Username already exists! Please choose another.")
        return False
    except sqlite3.Error as e:
        st.error(f"Registration error: {e}")
        return False
    finally:
        c.close()
        conn.close()


def login_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        return user
    except sqlite3.Error as e:
        st.error(f"Login error: {e}")
        return None
    finally:
        c.close()
        conn.close()


def get_predictions(username):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("""
            SELECT stock_symbol, predicted_price, prediction_date 
            FROM predictions 
            WHERE username = ? 
            ORDER BY timestamp DESC
        """, (username,))
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error fetching predictions: {e}")
        return []
    finally:
        c.close()
        conn.close()


def save_prediction(username, stock_symbol, predicted_price, prediction_date):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO predictions 
            (username, stock_symbol, predicted_price, prediction_date)
            VALUES (?, ?, ?, ?)
        """, (username, stock_symbol, predicted_price, prediction_date))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error saving prediction: {e}")
    finally:
        c.close()
        conn.close()


# =============================================
# APP INITIALIZATION
# =============================================
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.selected_stock = ""
    st.session_state.prediction_data = None

# =============================================
# MAIN APP INTERFACE
# =============================================
st.title("📈 Stock Price Predictor")

if not st.session_state.logged_in:
    option = st.radio("Select an option", ["Sign In", "Create Account"], horizontal=True)

    if option == "Create Account":
        with st.form("create_account"):
            new_user = st.text_input("Enter Username")
            new_pass = st.text_input("Enter Password", type="password")
            if st.form_submit_button("Create Account"):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("Account Created! Please Sign In.")
                else:
                    st.warning("Please enter both username and password")

    elif option == "Sign In":
        with st.form("sign_in"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In"):
                user = login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"👋 Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid Credentials")

else:
    # Lazy load heavy ML and finance libraries only when needed
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    import yfinance as yf
    from yahooquery import search

    st.subheader(f"👋 Welcome back, {st.session_state.username}!")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.selected_stock = ""
        st.session_state.prediction_data = None
        st.success("Logged out successfully!")
        st.rerun()

    with st.expander("🔍 Search for Stocks", expanded=True):
        company_name = st.text_input("Enter Company Name", key="search_input")
        if st.button("Search", key="search_btn"):
            with st.spinner("🔍 Searching for companies..."):
                results = search(company_name)
                if "quotes" in results and results['quotes']:
                    stock_options = {res['shortname']: res['symbol'] for res in results['quotes'] if
                                     'symbol' in res and 'shortname' in res}
                    if stock_options:
                        selected_company = st.selectbox("Select Company", list(stock_options.keys()))
                        if selected_company:
                            st.session_state.selected_stock = stock_options[selected_company]
                            st.success(f"✅ Selected: {selected_company} ({stock_options[selected_company]})")
                    else:
                        st.error("No valid companies found. Try another name.")
                else:
                    st.error("No results found. Try another name.")

    if st.session_state.selected_stock:
        st.markdown(f"### 📊 Analyzing: **{st.session_state.selected_stock}**")
        days_to_predict = st.slider("Select number of days to predict", min_value=1, max_value=30, value=7)

        if st.button("Predict Future Prices", key="predict_btn", type="primary"):
            stock_symbol = st.session_state.selected_stock

            # Data fetching stage
            with st.spinner("📥 Downloading historical stock data..."):
                try:
                    stock_data = yf.download(stock_symbol, period="6mo", progress=False)
                    if stock_data.empty:
                        st.error("⚠ No data found. Please check the stock symbol.")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Download failed: {str(e)}")
                    st.stop()

            st.success("✅ Historical data retrieved successfully!")

            # Data processing stage
            with st.spinner("⚙️ Processing data for model..."):
                data = stock_data[['Close']].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(data)

                X, y = [], []
                for i in range(60, len(data_scaled) - days_to_predict):
                    X.append(data_scaled[i - 60:i, 0])
                    y.append(data_scaled[i:i + days_to_predict, 0])
                X, y = np.array(X), np.array(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))

            # Model training/prediction stage
            with st.spinner("🧠 Training model and making predictions..."):
                model = Sequential([
                    LSTM(30, return_sequences=True, input_shape=(60, 1)),
                    LSTM(30),
                    Dense(days_to_predict)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)

                last_60_days = data_scaled[-60:].reshape(1, 60, 1)
                pred_scaled = model.predict(last_60_days)[0]
                predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                predictions = predictions[:days_to_predict]

                # Store prediction data in session state
                prediction_df = pd.DataFrame(predictions, columns=["Predicted Price"])
                prediction_df.index = pd.date_range(
                    start=pd.to_datetime(stock_data.index[-1]) + pd.Timedelta(days=1),
                    periods=len(predictions),
                    freq='B'
                )
                st.session_state.prediction_data = {
                    'df': prediction_df,
                    'last_price': data[-1][0],
                    'predicted_price': predictions[-1][0],
                    'stock_symbol': stock_symbol,
                    'prediction_date': str(prediction_df.index[-1].date())
                }

            # Save to database
            save_prediction(
                st.session_state.username,
                stock_symbol,
                float(predictions[-1][0]),
                str(prediction_df.index[-1].date())
            )

            st.success("🎉 Prediction complete! Showing results below...")
            st.rerun()

        # Display current prediction if exists
        if st.session_state.prediction_data:
            st.line_chart(st.session_state.prediction_data['df'])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Last Actual Price", f"₹{st.session_state.prediction_data['last_price']:.2f}")
            with col2:
                st.metric(f"Predicted Price (Day {days_to_predict})",
                          f"₹{st.session_state.prediction_data['predicted_price']:.2f}")

        # Prediction history
        if st.checkbox("📜 Show My Prediction History", key="history_checkbox"):
            history = get_predictions(st.session_state.username)

            if history:
                df = pd.DataFrame(history, columns=["Stock", "Predicted Price", "Date"])
                st.dataframe(
                    df.style.format({
                        "Predicted Price": "₹{:.2f}",
                        "Date": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')
                    }),
                    height=400
                )
            else:
                st.info("No prediction history found.")