# pages/dashboard.py - Updated version

# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def dashboard_page():
    # Page configuration
    
    page_title="Stock Purchase Prediction",
    page_icon="📊",
    layout="wide"
    

    st.title("📊 Stock Purchase Prediction Dashboard")
    st.markdown("---")

    # Sidebar for file upload
    with st.sidebar:
        st.title(f"👋 Welcome, {st.session_state.username}")
        st.info("Logistix Inventory System v1.0")
        st.markdown("---")
        st.sidebar.header("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        # Spacer
        st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)
        # Bottom container for logout
        bottom_container = st.container()
        with bottom_container:
            if st.button("Logout", type="primary", use_container_width=True):
                st.session_state.login_status = "logged_out"
                st.rerun()


    if uploaded_file is not None:
        # Load and preprocess data
        @st.cache_data
        def load_and_preprocess_data(file):
            df = pd.read_csv(file)
            
            # Product selection
            produk_pilihan = [
                'TEPUNG SEGITIGA 1kg', 'EKOMIE 3,6kg (isi6)', 'EKOMIE 2 (renteng)', 'INDOMILK BOTOL', 'INDOMILK KIDS UHT', 'INDOMIE GORENG',
                'TEPUNG CAKRA 1kg', 'TEPUNG CAKRA SAK (25kg)', 'TEPUNG KANJI ROSE BRAND', 'TEPUNG KETAN ROSE BRAND', 'DDS SEDAP KARE SP', 'DDS SEDAP SOTO',
                'DDS SEDAP KOREA', 'GULA RAJA GULA/NUSA KITA/KBA(50kg)','TEPUNG PAYUNG(25kg)','INDOMIE GRG ACEH', 'INDOMIE KRIUK', 'INDOMIE RENDANG', 'INDOMIE SOTO SP', 'SARIMIE 2 KECAP',
                'SARIMIE 2 KREMES', 'POPMIE B AYAM BAWANG', 'POPMIE B BASO', 'POPMIE B AYAM', 'POPMIE B KARI', 'INDOMIE CABE IJO',
                'DDS SEDAP GORENG','TEPUNG BERAS ROSE BRAND', 'MINYAK KITA REFF 1LT', 'DCW CO RENTENG', 'DCW INST RENTENG',
                'DDS TOP GULA', 'INDOMIE AYAM BAWANG', 'INDOMIE GEPREK', 'INDOMIE KARI', 'INDOMIE INTERMI', 'MILO RENTENG',
                'INDOMIE SEBLAK','INDOMIE TORI MISO', 'INDOMIE TAKO YAKI', 'INDOMIE GORENG RAWON','TEPUNG LOMBOK KUNING', 'TEPUNG SEGITIGA 1/2kg',
                'TEPUNG SEGITIGA SAK (25kg)', 'GARAM KAPAL 250gr', 'TEPUNG LOMBOK PUTIH', 'POPMIE B SOTO', 'DDS EKOMIE BAKSO isi 6',
                'MINYAK KITA REFF 2LT', 'INDOMIE KALDU', 'SUSU ZEE RENTENG', 'DDS SUKSES AYAM KECAP', 'DDS SUKSES AYAM KREMES',
                'MINYAK SABRINA REFF 1LT', 'INDOMIE RAWON', 'SARIMIE 2 KOREA','MINYAK SABRINA REFF 2LT', 'POPMIE PEDAS GLEDEK','MIE TELUR 3 AYAM',
                'MINYAK KITA BOTOL 1LT', 'MINYAK SUNCO REFF 2lt','SAJIKU 220gr','TEPUNG DRAGONFLY', 'TEPUNG VIRGO PEYEK','SUSU ZEE CO RTG',
                'SUSU ZEE VAN RTG','POPMIE GORENG SP', 'DDS SEDAP GORENG BAG','GULA (50kg)','3 SAPI JUMBO',
                'INDOMIE RENDANG JUMBO', 'INDOMIE GORENG JUMBO',  'INTERMIE PEDAS', 'DDS SEDAP BUMBU','POPMIE MI GORENG','DDS EKOMIE 3,6kg (isi6)',
                'TEPUNG GEPREK SAKURA'
            ]
            
            # Filter selected products
            df = df[df['nama_stok'].isin(produk_pilihan)].reset_index(drop=True)
            
            # Clean currency function
            def clean_currency(x):
                if isinstance(x, str):
                    return int(x.replace(',', '').replace(' ', '').replace('"', ''))
                return x
            
            # Clean currency columns if they exist
            if 'harga_satuan' in df.columns:
                df['harga_satuan'] = df['harga_satuan'].apply(clean_currency)
            if 'jumlah' in df.columns:
                df['jumlah'] = df['jumlah'].apply(clean_currency)
            
            # Convert date
            df['tanggal'] = pd.to_datetime(df['tanggal'], format='%Y-%m-%d')
            
            # Drop unnecessary columns
            columns_to_drop = ['nota', 'suplier', 'id_transaksi', 'kode_stok']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Drop missing values
            df = df.dropna(subset=['nama_stok']).reset_index(drop=True)
            
            # Feature Engineering
            df['day_of_week'] = df['tanggal'].dt.day_name()
            df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
            
            return df, produk_pilihan
        
        # Train model and make predictions
        @st.cache_data
        def train_and_predict(df, produk_pilihan):
            # Prepare model data
            df_model = df[['nama_stok', 'day_of_week', 'is_weekend', 'quantity']]
            
            # Split data
            X = df_model.drop(columns=['quantity'])
            y = df_model['quantity']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            categorical_features = ['nama_stok', 'day_of_week']
            model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                loss_function='RMSEWithUncertainty',
                verbose=0  # Silent training
            )
            
            model.fit(X_train, y_train, cat_features=categorical_features)
            
            # Evaluate model
            test_preds = model.predict(X_test, prediction_type='RMSEWithUncertainty')
            test_mean = test_preds[:, 0]
            
            mae = mean_absolute_error(y_test, test_mean)
            mse = mean_squared_error(y_test, test_mean)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, test_mean)
            
            # Future predictions
            future_dates = pd.date_range(start=pd.Timestamp.today(), periods=30)
            
            future_data = []
            for date in future_dates:
                day_of_week = date.day_name()
                is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
                for produk in produk_pilihan:
                    future_data.append({
                        'nama_stok': produk,
                        'day_of_week': day_of_week,
                        'is_weekend': is_weekend
                    })
            
            future_df = pd.DataFrame(future_data)
            
            # Predict future
            future_preds = model.predict(future_df, prediction_type='RMSEWithUncertainty')
            future_mean = future_preds[:, 0]
            future_std = future_preds[:, 1]
            
            # Confidence intervals
            k_confidence = 1.96  # 95%
            lower_bound = future_mean - (k_confidence * future_std)
            upper_bound = future_mean + (k_confidence * future_std)
            
            # Create forecast results
            result_forecast = pd.DataFrame({
                'tanggal': [d for d in future_dates for _ in range(len(produk_pilihan))],
                'nama_stok': produk_pilihan * len(future_dates),
                'predicted_quantity': future_mean,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            # Categorize products
            def categorize(row):
                if row['lower_bound'] > 2:
                    return 'Pasti Dibeli'
                elif 0 < row['lower_bound'] <= 2:
                    return 'Ragu'
                else:
                    return 'Tidak Perlu Dibeli'
            
            result_forecast['kategori'] = result_forecast.apply(categorize, axis=1)
            
            metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}
            
            return result_forecast, metrics
        
        # Load data
        with st.spinner('Loading and processing data...'):
            df, produk_pilihan = load_and_preprocess_data(uploaded_file)
        
        # Train model and predict
        with st.spinner('Training model and making predictions...'):
            result_forecast, metrics = train_and_predict(df, produk_pilihan)
        
        # Display results
        st.success("✅ Model trained successfully!")
        
        # Prediction Results
        st.header("🔮 30-Day Prediction Results")
        
        # Category summary
        category_counts = result_forecast['kategori'].value_counts()
        product_counts = result_forecast.groupby('kategori')['nama_stok'].nunique()
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🟢 Pasti Restok", f"{category_counts.get('Pasti Dibeli', 0)} kali")
            st.metric("🟢 Produk Pasti Dibeli", f"{product_counts.get('Pasti Dibeli', 0)} produk")

        with col2:
            st.metric("🟡 Mungkin Restok", f"{category_counts.get('Ragu', 0)} kali")
            st.metric("🟡 Produk Mungkin Dibeli", f"{product_counts.get('Ragu', 0)} produk")

        with col3:
            st.metric("🔴 Tidak Perlu Restok", f"{category_counts.get('Tidak Perlu Dibeli', 0)} kali")
            st.metric("🔴 Produk Tidak Perlu Dibeli", f"{product_counts.get('Tidak Perlu Dibeli', 0)} produk")


        
        # Category distribution chart
        fig_pie = px.pie(
            values=product_counts.values, 
            names=category_counts.index,
            title="Distribution of Purchase Recommendations Based on Product",
            color_discrete_map={
                'Pasti Dibeli': '#00ff00',
                'Ragu': '#ffff00', 
                'Tidak Perlu Dibeli': '#ff0000'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Detailed predictions by category
        st.header("📋 Detailed Predictions by Category")

        tab1, tab2, tab3 = st.tabs(["🟢 Pasti Dibeli", "🟡 Ragu", "🔴 Tidak Perlu Dibeli"])

        def create_filter_panel(df, tab_key):
            """Helper function to create consistent filter panel"""
            with st.expander("🔍 Filter Options", expanded=True):
                # Column selection with display names
                column_display_names = {
                    'tanggal': 'Tanggal',
                    'nama_stok': 'Nama Stok',
                    'predicted_quantity': 'Predicted Quantity',
                    'lower_bound': 'Lower Bound',
                    'upper_bound': 'Upper Bound',
                    'kategori': 'Kategori'
                }
                
                # Get available columns from the dataframe that exist in our display names mapping
                available_columns = [col for col in df.columns if col in column_display_names]
                
                default_cols = ['tanggal', 'nama_stok', 'predicted_quantity', 'lower_bound', 'upper_bound']
                cols = st.multiselect(
                    "Columns to display:",
                    options=available_columns,
                    format_func=lambda x: column_display_names[x],
                    default=default_cols,
                    key=f"cols_{tab_key}"
                )
                
                # Date range filter
                min_date = df['tanggal'].min()
                max_date = df['tanggal'].max()
                date_range = st.date_input(
                    "Date range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"date_{tab_key}"
                )
                
                # Product filter
                products = st.multiselect(
                    "Filter products:",
                    options=df['nama_stok'].unique(),
                    key=f"product_{tab_key}"
                )
                
                # Quantity range filter
                min_qty = float(df['predicted_quantity'].min())
                max_qty = float(df['predicted_quantity'].max())
                qty_range = st.slider(
                    "Quantity range:",
                    min_value=min_qty,
                    max_value=max_qty,
                    value=(min_qty, max_qty),
                    key=f"qty_{tab_key}"
                )
                
                # Download button
                st.download_button(
                    "📥 Download CSV",
                    df[cols].to_csv(index=False).encode('utf-8'),
                    f"{tab_key}_products.csv",
                    "text/csv",
                    key=f'download_{tab_key}'
                )
            
            return cols, date_range, products, qty_range

        # Then modify the dataframe display part in each tab to use the display names:

        with tab1:
            df_pasti_dibeli = result_forecast[result_forecast['kategori'] == 'Pasti Dibeli']
            if not df_pasti_dibeli.empty:
                col_table, col_filter = st.columns([3, 1])
                
                with col_filter:
                    cols, date_range, products, qty_range = create_filter_panel(df_pasti_dibeli, "past_dibeli")
                
                # Apply filters
                filtered_df = df_pasti_dibeli.copy()
                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['tanggal'] >= pd.to_datetime(date_range[0])) &
                        (filtered_df['tanggal'] <= pd.to_datetime(date_range[1]))
                    ]
                if products:
                    filtered_df = filtered_df[filtered_df['nama_stok'].isin(products)]
                filtered_df = filtered_df[
                    (filtered_df['predicted_quantity'] >= qty_range[0]) &
                    (filtered_df['predicted_quantity'] <= qty_range[1])
                ]
                
                with col_table:
                    # Create a copy of the dataframe with renamed columns for display
                    display_df = filtered_df[cols].round(2).rename(columns={
                        'tanggal': 'Tanggal',
                        'nama_stok': 'Nama Stok',
                        'predicted_quantity': 'Predicted Quantity',
                        'lower_bound': 'Lower Bound',
                        'upper_bound': 'Upper Bound',
                        'kategori': 'Kategori'
                    })
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=600
                    )
                    
                    # Top products visualization
                    top_products = filtered_df.groupby('nama_stok')['predicted_quantity'].sum().sort_values(ascending=False).head(10)
                    fig_bar = px.bar(
                        x=top_products.values,
                        y=top_products.index,
                        orientation='h',
                        title="Top 10 Products With High Possibility to Restock",
                        labels={'x': 'Total Predicted Quantity', 'y': 'Product Name'},
                        category_orders={"y": top_products.index.tolist()},
                        text=top_products.values.round(2)
                    )
                    fig_bar.update_traces(
                        texttemplate='%{text}',
                        textposition='inside',  
                        textfont_color='white', 
                        insidetextanchor='end',  
                        insidetextfont=dict(size=12),  
                        marker_color='rgb(55, 83, 109)'  
                    )
                    fig_bar.update_layout(height=450)
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No products in 'Pasti Dibeli' category")

        # Repeat the same modification for tab2 and tab3
        with tab2:
            df_ragu = result_forecast[result_forecast['kategori'] == 'Ragu']
            if not df_ragu.empty:
                col_table, col_filter = st.columns([3, 1])
                
                with col_filter:
                    cols, date_range, products, qty_range = create_filter_panel(df_ragu, "ragu")
                
                # Apply filters
                filtered_df = df_ragu.copy()
                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['tanggal'] >= pd.to_datetime(date_range[0])) &
                        (filtered_df['tanggal'] <= pd.to_datetime(date_range[1]))
                    ]
                if products:
                    filtered_df = filtered_df[filtered_df['nama_stok'].isin(products)]
                filtered_df = filtered_df[
                    (filtered_df['predicted_quantity'] >= qty_range[0]) &
                    (filtered_df['predicted_quantity'] <= qty_range[1])
                ]
                
                with col_table:
                    display_df = filtered_df[cols].round(2).rename(columns={
                        'tanggal': 'Tanggal',
                        'nama_stok': 'Nama Stok',
                        'predicted_quantity': 'Predicted Quantity',
                        'lower_bound': 'Lower Bound',
                        'upper_bound': 'Upper Bound',
                        'kategori': 'Kategori'
                    })
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=600
                    )                                        
            else:
                st.info("No products in 'Ragu' category")

        with tab3:
            df_tidak_perlu_dibeli = result_forecast[result_forecast['kategori'] == 'Tidak Perlu Dibeli']
            if not df_tidak_perlu_dibeli.empty:
                col_table, col_filter = st.columns([3, 1])
                
                with col_filter:
                    cols, date_range, products, qty_range = create_filter_panel(df_tidak_perlu_dibeli, "tidak_dibeli")
                
                # Apply filters
                filtered_df = df_tidak_perlu_dibeli.copy()
                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['tanggal'] >= pd.to_datetime(date_range[0])) &
                        (filtered_df['tanggal'] <= pd.to_datetime(date_range[1]))
                    ]
                if products:
                    filtered_df = filtered_df[filtered_df['nama_stok'].isin(products)]
                filtered_df = filtered_df[
                    (filtered_df['predicted_quantity'] >= qty_range[0]) &
                    (filtered_df['predicted_quantity'] <= qty_range[1])
                ]
                
                with col_table:
                    display_df = filtered_df[cols].round(2).rename(columns={
                        'tanggal': 'Tanggal',
                        'nama_stok': 'Nama Stok',
                        'predicted_quantity': 'Predicted Quantity',
                        'lower_bound': 'Lower Bound',
                        'upper_bound': 'Upper Bound',
                        'kategori': 'Kategori'
                    })
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=600
                    )
                    
                    # Bottom products visualization
                    top_products = filtered_df.groupby('nama_stok')['predicted_quantity'].sum().sort_values(ascending=True).head(10)
                    fig_bar = px.bar(
                        x=top_products.values,
                        y=top_products.index,
                        orientation='h',
                        title="Bottom 10 Products to Buy",
                        labels={'x': 'Total Predicted Quantity', 'y': 'Product Name'},
                        category_orders={"y": top_products.index.tolist()}
                    )
                    fig_bar.update_traces(marker_color='rgb(55, 83, 109)')
                    fig_bar.update_layout(height=450)
                    st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.info("No products in 'Tidak Perlu Dibeli' category")

        
        # Time series visualization
        st.header("📊 Prediction Trends Over Time")
        
        # Select product for detailed view
        selected_product = st.selectbox(
            "Select a product to view prediction trend:",
            options=produk_pilihan
        )
        
        if selected_product:
            product_data = result_forecast[result_forecast['nama_stok'] == selected_product]
            
            fig_line = go.Figure()
            
            # Add prediction line
            fig_line.add_trace(go.Scatter(
                x=product_data['tanggal'],
                y=product_data['predicted_quantity'],
                mode='lines+markers',
                name='Predicted Quantity',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence interval
            fig_line.add_trace(go.Scatter(
                x=product_data['tanggal'],
                y=product_data['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(color='lightblue', width=1),
                showlegend=False
            ))
            
            fig_line.add_trace(go.Scatter(
                x=product_data['tanggal'],
                y=product_data['lower_bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(color='lightblue', width=1),
                fill='tonexty',
                fillcolor='rgba(173,216,230,0.3)',
                showlegend=True
            ))
            
            fig_line.update_layout(
                title=f"30-Day Prediction for {selected_product}",
                xaxis_title="Date",
                yaxis_title="Predicted Quantity",
                height=400
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Download predictions
        st.header("💾 Download Predictions")
        
        csv = result_forecast.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Predictions CSV",
            data=csv,
            file_name=f"stock_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Please upload a CSV file to begin the analysis")
        
        # Show sample data format
        st.header("📋 Expected Data Format")
        st.markdown("""
        Your CSV file should contain the following columns:
        - `nama_stok`: Product name
        - `quantity`: Purchase quantity 
        - `tanggal`: Date (YYYY-MM-DD format)
        - `harga_satuan`: Unit price (optional)
        - `jumlah`: Total amount (optional)
        
        The application will automatically filter for predefined product categories and make 30-day predictions.
        """)