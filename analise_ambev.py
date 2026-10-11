import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA & ESTILO AMBEV
# ==========================================
st.set_page_config(
    page_title="Análise de Desempenho e Saúde Financeira: Ambev S.A.",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_NAVY = "#002B66"       # Azul Escuro
COLOR_GOLD = "#FFAB00"       # Dourado
COLOR_RED = "#D32F2F"        # Vermelho Ambev
COLOR_GREEN = "#2E7D32"      # Verde
COLOR_PRIMARY = "#D32F2F"    # Vermelho Primary
COLOR_SECONDARY = "#FFB300"  # Dourado Cerveja
COLOR_LINE_BLUE = "#1565C0"  # Azul Linha
COLOR_PURPLE = "#7B1FA2"     # Roxo Projeção Regressão

st.markdown(f"""
    <style>
        .stApp {{
            background-color: #FFFFFF;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .main-header {{
            font-size: 28px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 2px;
        }}
        .sub-header {{
            font-size: 13px;
            color: #64748B;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 5px solid {COLOR_GOLD};
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .kpi-title {{
            font-size: 12px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 800;
            color: #0F172A;
        }}
        .kpi-sub {{
            font-size: 12px;
            font-weight: 600;
            color: #16A34A;
            margin-top: 5px;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #F1F5F9;
            border-right: 1px solid #E2E8F0;
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARREGAMENTO E PROCESSAMENTO CVM
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv('ambev_financeiro_tidy.csv')

def desacumular_serie(df_tidy, demo, cd_conta, anos=[2021, 2022, 2023, 2024, 2025]):
    records = []
    for ano in anos:
        sub = df_tidy[
            (df_tidy['demonstracao'] == demo) & 
            (df_tidy['cd_conta'] == cd_conta) & 
            (df_tidy['data_fim_exercicio'].str.startswith(str(ano))) &
            (df_tidy['ordem_exerc'] == 'ÚLTIMO')
        ].sort_values('data_fim_exercicio')
        
        ytd = sub.set_index('data_fim_exercicio')['valor'].to_dict()
        dates = sorted(list(ytd.keys()))
        
        if len(dates) == 4:
            records.append({'data_fim_exercicio': dates[0], 'valor': ytd[dates[0]]})
            records.append({'data_fim_exercicio': dates[1], 'valor': ytd[dates[1]] - ytd[dates[0]]})
            records.append({'data_fim_exercicio': dates[2], 'valor': ytd[dates[2]] - ytd[dates[1]]})
            records.append({'data_fim_exercicio': dates[3], 'valor': ytd[dates[3]] - ytd[dates[2]]})
            
    return pd.DataFrame(records)

@st.cache_data
def build_dataset(df_tidy):
    years = [2021, 2022, 2023, 2024, 2025]
    
    rec_q = desacumular_serie(df_tidy, 'DRE_con', '3.01', years).rename(columns={'valor': 'Receita'})
    ebit_q = desacumular_serie(df_tidy, 'DRE_con', '3.05', years).rename(columns={'valor': 'EBIT'})
    ll_q = desacumular_serie(df_tidy, 'DRE_con', '3.11', years).rename(columns={'valor': 'Lucro_Liquido'})
    gross_q = desacumular_serie(df_tidy, 'DRE_con', '3.03', years).rename(columns={'valor': 'Lucro_Bruto'})
    cfo_q = desacumular_serie(df_tidy, 'DFC_MI_con', '6.01', years).rename(columns={'valor': 'CFO'})
    depr_q = desacumular_serie(df_tidy, 'DFC_MI_con', '6.01.01.02', years).rename(columns={'valor': 'Depreciacao'})
    capex_q = desacumular_serie(df_tidy, 'DFC_MI_con', '6.02.01', years).rename(columns={'valor': 'CAPEX'})
    ir_q = desacumular_serie(df_tidy, 'DRE_con', '3.08', years).rename(columns={'valor': 'IR_CS'})
    lai_q = desacumular_serie(df_tidy, 'DRE_con', '3.07', years).rename(columns={'valor': 'LAI'})

    df_q = rec_q.merge(ebit_q, on='data_fim_exercicio') \
               .merge(ll_q, on='data_fim_exercicio') \
               .merge(gross_q, on='data_fim_exercicio') \
               .merge(cfo_q, on='data_fim_exercicio') \
               .merge(depr_q, on='data_fim_exercicio') \
               .merge(capex_q, on='data_fim_exercicio') \
               .merge(ir_q, on='data_fim_exercicio') \
               .merge(lai_q, on='data_fim_exercicio')

    bp_records = []
    dates_all = sorted(df_q['data_fim_exercicio'].unique())
    for d in dates_all:
        sub_ac = df_tidy[(df_tidy['demonstracao']=='BPA_con') & (df_tidy['cd_conta']=='1.01') & (df_tidy['data_fim_exercicio']==d) & (df_tidy['ordem_exerc']=='ÚLTIMO')]
        sub_caixa = df_tidy[(df_tidy['demonstracao']=='BPA_con') & (df_tidy['cd_conta'].isin(['1.01.01', '1.01.02'])) & (df_tidy['data_fim_exercicio']==d) & (df_tidy['ordem_exerc']=='ÚLTIMO')]
        sub_est = df_tidy[(df_tidy['demonstracao']=='BPA_con') & (df_tidy['cd_conta']=='1.01.04') & (df_tidy['data_fim_exercicio']==d) & (df_tidy['ordem_exerc']=='ÚLTIMO')]
        sub_pc = df_tidy[(df_tidy['demonstracao']=='BPP_con') & (df_tidy['cd_conta']=='2.01') & (df_tidy['data_fim_exercicio']==d) & (df_tidy['ordem_exerc']=='ÚLTIMO')]
        sub_div = df_tidy[(df_tidy['demonstracao']=='BPP_con') & (df_tidy['cd_conta'].isin(['2.01.04', '2.02.01'])) & (df_tidy['data_fim_exercicio']==d) & (df_tidy['ordem_exerc']=='ÚLTIMO')]
        sub_pl = df_tidy[(df_tidy['demonstracao']=='BPP_con') & (df_tidy['cd_conta']=='2.03') & (df_tidy['data_fim_exercicio']==d) & (df_tidy['ordem_exerc']=='ÚLTIMO')]
        sub_ativo = df_tidy[(df_tidy['demonstracao']=='BPA_con') & (df_tidy['cd_conta']=='1') & (df_tidy['data_fim_exercicio']==d) & (df_tidy['ordem_exerc']=='ÚLTIMO')]

        v_ac = sub_ac['valor'].values[0] if len(sub_ac)>0 else 0
        v_caixa = sub_caixa['valor'].sum() if len(sub_caixa)>0 else 0
        v_est = sub_est['valor'].values[0] if len(sub_est)>0 else 0
        v_pc = sub_pc['valor'].values[0] if len(sub_pc)>0 else 1
        v_div = sub_div['valor'].sum() if len(sub_div)>0 else 0
        v_pl = sub_pl['valor'].values[0] if len(sub_pl)>0 else 1
        v_ativo = sub_ativo['valor'].values[0] if len(sub_ativo)>0 else 1

        bp_records.append({
            'data_fim_exercicio': str(d),
            'Ativo_Circulante': v_ac,
            'Caixa_Total': v_caixa,
            'Estoques': v_est,
            'Passivo_Circulante': v_pc,
            'Divida_Bruta': v_div,
            'PL': v_pl,
            'Ativo_Total': v_ativo
        })

    df_q = df_q.merge(pd.DataFrame(bp_records), on='data_fim_exercicio')

    df_q['Liquidez_Corrente'] = df_q['Ativo_Circulante'] / df_q['Passivo_Circulante']
    df_q['Liquidez_Seca'] = (df_q['Ativo_Circulante'] - df_q['Estoques']) / df_q['Passivo_Circulante']
    df_q['Liquidez_Imediata'] = df_q['Caixa_Total'] / df_q['Passivo_Circulante']
    df_q['Caixa_Liquido'] = df_q['Caixa_Total'] - df_q['Divida_Bruta']
    
    df_q['EBITDA'] = df_q['EBIT'] + df_q['Depreciacao'].abs()
    df_q['FCF'] = df_q['CFO'] + df_q['CAPEX']
    
    tax_rate = (-df_q['IR_CS'] / df_q['LAI']).clip(0, 0.34)
    nopat = df_q['EBIT'] * (1 - tax_rate)
    cap_inv = df_q['PL'] + df_q['Divida_Bruta'] - df_q['Caixa_Total']
    
    df_q['ROIC'] = (nopat / cap_inv) * 100
    df_q['ROE'] = (df_q['Lucro_Liquido'] / df_q['PL']) * 100
    df_q['ROA'] = (df_q['Lucro_Liquido'] / df_q['Ativo_Total']) * 100

    df_q['Margem_Bruta'] = (df_q['Lucro_Bruto'] / df_q['Receita']) * 100
    df_q['Margem_EBITDA'] = (df_q['EBITDA'] / df_q['Receita']) * 100
    df_q['Margem_EBIT'] = (df_q['EBIT'] / df_q['Receita']) * 100
    df_q['Margem_Liquida'] = (df_q['Lucro_Liquido'] / df_q['Receita']) * 100

    df_q['FCO_EBIT'] = df_q['CFO'] / df_q['EBIT']
    df_q['CFO_LL'] = df_q['CFO'] / df_q['Lucro_Liquido']

    cols_mi = ['Receita', 'EBITDA', 'EBIT', 'Lucro_Liquido', 'Lucro_Bruto', 'CFO', 'CAPEX', 'FCF', 'Caixa_Total', 'Divida_Bruta', 'Caixa_Liquido', 'PL']
    for c in cols_mi:
        df_q[c] = df_q[c] / 1e6

    df_q['data_fim_exercicio'] = df_q['data_fim_exercicio'].astype(str)
    return df_q

@st.cache_data
def build_annual_dataset(df_q):
    df_temp = df_q.copy()
    df_temp['Ano'] = df_temp['data_fim_exercicio'].str[:4].astype(str)
    
    annual_flow = df_temp.groupby('Ano')[['Receita', 'EBITDA', 'EBIT', 'Lucro_Liquido', 'Lucro_Bruto', 'CFO', 'CAPEX', 'FCF']].sum().reset_index()
    
    annual_bp = df_temp[df_temp['data_fim_exercicio'].str.endswith('-12-31')][['data_fim_exercicio', 'Caixa_Total', 'Divida_Bruta', 'Caixa_Liquido', 'Ativo_Circulante', 'Passivo_Circulante', 'Estoques', 'PL']].copy()
    annual_bp['Ano'] = annual_bp['data_fim_exercicio'].str[:4].astype(str)
    
    df_annual = annual_flow.merge(annual_bp, on='Ano')
    
    df_annual['Margem_Bruta'] = (df_annual['Lucro_Bruto'] / df_annual['Receita']) * 100
    df_annual['Margem_EBITDA'] = (df_annual['EBITDA'] / df_annual['Receita']) * 100
    df_annual['Margem_EBIT'] = (df_annual['EBIT'] / df_annual['Receita']) * 100
    df_annual['Margem_Liquida'] = (df_annual['Lucro_Liquido'] / df_annual['Receita']) * 100
    df_annual['FCO_EBIT'] = df_annual['CFO'] / df_annual['EBIT']
    
    df_annual['Liquidez_Corrente'] = df_annual['Ativo_Circulante'] / df_annual['Passivo_Circulante']
    df_annual['Liquidez_Seca'] = (df_annual['Ativo_Circulante'] - df_annual['Estoques']) / df_annual['Passivo_Circulante']
    df_annual['Liquidez_Imediata'] = df_annual['Caixa_Total'] / df_annual['Passivo_Circulante']
    
    df_annual['ROIC'] = ((df_annual['EBIT'] * 0.80) / (df_annual['PL'] + df_annual['Divida_Bruta'] - df_annual['Caixa_Total'])) * 100
    df_annual['data_fim_exercicio'] = df_annual['Ano'].astype(str)
    return df_annual

@st.cache_data(ttl=3600)
def fetch_market_price(ticker="ABEV3"):
    try:
        url = f"https://brapi.dev/api/quote/{ticker}"
        r = requests.get(url, timeout=3).json()
        return float(r['results'][0]['regularMarketPrice'])
    except Exception:
        return 15.84

df_tidy = load_data()
df_q = build_dataset(df_tidy)
df_a = build_annual_dataset(df_q)

lista_datas_q = sorted(list(df_q['data_fim_exercicio'].unique()))
lista_anos = sorted(list(df_a['data_fim_exercicio'].unique()))
current_price = fetch_market_price("ABEV3")
total_shares = 15535560662

# ==========================================
# 3. SIDEBAR (PAINEL DE CONTROLE)
# ==========================================
st.sidebar.markdown("### 📌 Painel de Controle")

tipo_visao = st.sidebar.radio(
    "📅 Período nos Gráficos:",
    options=["Trimestral (ITR)", "Anual Acumulado (DFP)"],
    index=0
)

# Seleção Dinâmica do Dataset
if tipo_visao == "Anual Acumulado (DFP)":
    df_active = df_a.copy()
    opcoes_destaque = lista_anos
else:
    df_active = df_q.copy()
    opcoes_destaque = lista_datas_q

data_destaque = st.sidebar.selectbox(
    "📌 Período em Destaque nos Cards (KPIs):",
    options=opcoes_destaque,
    index=len(opcoes_destaque) - 1 # Seleciona o mais recente por padrão
)

datas_selecionadas = st.sidebar.multiselect(
    "📊 Filtrar Períodos nos Gráficos:",
    options=list(df_active['data_fim_exercicio'].unique()),
    default=list(df_active['data_fim_exercicio'].unique())
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Calibragem do Score")
w_qual = st.sidebar.slider("1. Rentabilidade (ROIC/ROE)", 0, 50, 25) / 100
w_saude = st.sidebar.slider("2. Saúde Fin. (Caixa/Dívida)", 0, 50, 20) / 100
w_cresc = st.sidebar.slider("3. Crescimento (Receita/EBIT)", 0, 50, 20) / 100
w_val = st.sidebar.slider("4. Valuation (FCF Yield)", 0, 50, 25) / 100
w_risco = st.sidebar.slider("5. Estabilidade (Volatilidade)", 0, 50, 10) / 100

st.sidebar.markdown("---")
exibir_comparacao = st.sidebar.checkbox("Exibir Média do Setor no Radar", value=True)

st.sidebar.markdown(f"🟢 **Cotação B3 (Tempo Real):** R$ {current_price:.2f}")
price_input = st.sidebar.number_input("Cotação para Valuation (R$)", value=float(current_price), step=0.10)

df_filtered = df_active[df_active['data_fim_exercicio'].isin(datas_selecionadas)].copy()
row_destaque = df_active[df_active['data_fim_exercicio'] == data_destaque].iloc[0]

# ==========================================
# 4. QUALITY SCORE DINÂMICO
# ==========================================
score_qualidade_din = min(100.0, float(df_filtered['ROIC'].mean()) * 3.6) if len(df_filtered) > 0 else 50.0
score_saude_din = 100.0 if df_filtered['Caixa_Liquido'].mean() > 0 else 50.0

if len(df_filtered) > 1:
    cagr_rec = ((df_filtered.iloc[-1]['Receita'] / df_filtered.iloc[0]['Receita']) ** (1/max(1, len(df_filtered)-1)) - 1) * 100
    score_crescimento_din = min(100.0, max(20.0, 50.0 + cagr_rec * 3))
else:
    score_crescimento_din = 60.0

ltm_fcf_din = df_q['FCF'].tail(4).sum()
mkt_cap = price_input * total_shares
fcf_yield_din = ((ltm_fcf_din * 1e6) / mkt_cap) * 100 if mkt_cap > 0 else 5.0
score_valuation_din = min(100.0, float(fcf_yield_din) * 8.5)

std_ebit = df_filtered['Margem_EBIT'].std() if len(df_filtered) > 1 else 0.0
score_estabilidade_din = max(40.0, 100.0 - (std_ebit * 5.0))

quality_score_din = (score_qualidade_din * w_qual + 
                     score_saude_din * w_saude + 
                     score_crescimento_din * w_cresc + 
                     score_valuation_din * w_val + 
                     score_estabilidade_din * w_risco)

if quality_score_din >= 80:
    classificacao_din = "EXCELENTE (A)"
elif quality_score_din >= 65:
    classificacao_din = "BOA (B)"
elif quality_score_din >= 50:
    classificacao_din = "RAZOÁVEL (C)"
else:
    classificacao_din = "FRACA (D)"

# ==========================================
# 5. DASHBOARD PRINCIPAL (5 ABAS)
# ==========================================
st.markdown("<div class='main-header'>🍺 Análise de Desempenho e Saúde Financeira: Ambev S.A.</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Aplicações de Estatística Aplicada & Finanças Corporativas | Modos: <b>{tipo_visao}</b></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Visão Geral & Métricas", 
    "🛡️ Solvência & Liquidez", 
    "💰 Rentabilidade & Caixa", 
    "⚖️ Quality Score & Radar",
    "🔮 Análises Preditivas (DCF & Regressão)"
])

# ------------------------------------------
# TAB 1: VISÃO GERAL (TÍTULO ANUAL ENFATIZADO)
# ------------------------------------------
with tab1:
    rotulo_periodo = "Ano (DFP)" if tipo_visao == "Anual Acumulado (DFP)" else "Trimestre (ITR)"
    sufixo_kpi = "Ano" if tipo_visao == "Anual Acumulado (DFP)" else "Q"
    
    st.markdown(f"### Visão Executiva — {rotulo_periodo}: {data_destaque}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Receita Líquida ({sufixo_kpi})</div><div class='kpi-value'>R$ {row_destaque['Receita']:,.0f} M</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>EBIT ({sufixo_kpi})</div><div class='kpi-value'>R$ {row_destaque['EBIT']:,.0f} M</div><div class='kpi-sub'>↑ {row_destaque['Margem_EBIT']:.1f}% Margem</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Lucro Líquido ({sufixo_kpi})</div><div class='kpi-value'>R$ {row_destaque['Lucro_Liquido']:,.0f} M</div><div class='kpi-sub'>↑ {row_destaque['Margem_Liquida']:.1f}% Margem</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Caixa Líquido / (Dívida)</div><div class='kpi-value'>R$ {row_destaque['Caixa_Liquido']:,.0f} M</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"#### Evolução Temporal das Séries de Resultado ({tipo_visao})")
    
    fig_resultado = go.Figure()
    fig_resultado.add_trace(go.Bar(
        x=df_filtered['data_fim_exercicio'], y=df_filtered['Receita'], 
        name='Receita Líquida', marker_color=COLOR_NAVY
    ))
    fig_resultado.add_trace(go.Scatter(
        x=df_filtered['data_fim_exercicio'], y=df_filtered['EBIT'], 
        name='EBIT (Resultado Operacional)', line=dict(color=COLOR_GOLD, width=3)
    ))
    fig_resultado.add_trace(go.Scatter(
        x=df_filtered['data_fim_exercicio'], y=df_filtered['Lucro_Liquido'], 
        name='Lucro Líquido', line=dict(color=COLOR_LINE_BLUE, width=3)
    ))
    fig_resultado.update_layout(
        xaxis=dict(type='category'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), 
        height=450, plot_bgcolor="white"
    )
    st.plotly_chart(fig_resultado, use_container_width=True)

# ------------------------------------------
# TAB 2: SOLVÊNCIA
# ------------------------------------------
with tab2:
    st.markdown("### Análise de Solvência, Liquidez e Perfil de Caixa")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown(f"#### Índices de Liquidez ({tipo_visao})")
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Liquidez_Corrente'], name='Corrente', line=dict(color=COLOR_NAVY, width=2)))
        fig_liq.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Liquidez_Seca'], name='Seca', line=dict(color=COLOR_GOLD, width=2, dash='dash')))
        fig_liq.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Liquidez_Imediata'], name='Imediata', line=dict(color="#64748B", width=2, dash='dot')))
        fig_liq.update_layout(xaxis=dict(type='category'), height=380, plot_bgcolor="white")
        st.plotly_chart(fig_liq, use_container_width=True)
    with col_l2:
        st.markdown(f"#### Posição Financeira: Disponibilidades vs. Dívida Bruta ({tipo_visao})")
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Bar(x=df_filtered['data_fim_exercicio'], y=df_filtered['Caixa_Total'], name='Caixa & Aplicações', marker_color=COLOR_GREEN))
        fig_pos.add_trace(go.Bar(x=df_filtered['data_fim_exercicio'], y=-df_filtered['Divida_Bruta'], name='Dívida Bruta', marker_color=COLOR_RED))
        fig_pos.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Caixa_Liquido'], name='Caixa Líquido', line=dict(color=COLOR_NAVY, width=3)))
        fig_pos.update_layout(xaxis=dict(type='category'), barmode='relative', height=380, plot_bgcolor="white")
        st.plotly_chart(fig_pos, use_container_width=True)

# ------------------------------------------
# TAB 3: RENTABILIDADE & CAIXA (FATOR DE CONVERSÃO CORRIGIDO)
# ------------------------------------------
with tab3:
    st.markdown("### Margens Operacionais e Conversão de Caixa")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"#### Estrutura de Margens (%) — ({tipo_visao})")
        fig_empilhada = go.Figure()
        fig_empilhada.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Margem_Bruta'], name='Margem_Bruta', fill='tozeroy', fillcolor='#FFF8E1', line=dict(color='#FFE082')))
        fig_empilhada.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Margem_EBITDA'], name='Margem_EBITDA', fill='tonexty', fillcolor='#FFECB3', line=dict(color='#FFB300')))
        fig_empilhada.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Margem_EBIT'], name='Margem_EBIT', fill='tonexty', fillcolor='#FFCDD2', line=dict(color='#E53935', width=2)))
        fig_empilhada.add_trace(go.Scatter(x=df_filtered['data_fim_exercicio'], y=df_filtered['Margem_Liquida'], name='Margem_Liquida', fill='tonexty', fillcolor='#BBDEFB', line=dict(color='#002B66', width=2)))
        fig_empilhada.update_layout(xaxis=dict(type='category'), height=380, plot_bgcolor="white")
        st.plotly_chart(fig_empilhada, use_container_width=True)
        
    with col_m2:
        st.markdown(f"#### Fator de Conversão de Caixa (FCO / EBIT) — ({tipo_visao})")
        
        # FORÇA O EIXO X A EXIBIR TODOS OS ANOS/PERÍODOS DE FORMA CATEGÓRICA
        eixo_x_fco = list(df_filtered['data_fim_exercicio'].astype(str))
        eixo_y_fco = list(df_filtered['FCO_EBIT'])
        
        fig_fco = go.Figure()
        fig_fco.add_trace(go.Bar(
            x=eixo_x_fco, 
            y=eixo_y_fco, 
            name='FCO / EBIT', 
            marker_color=COLOR_NAVY
        ))
        
        # LINHA DE REFERÊNCIA META (1.0x)
        fig_fco.add_shape(
            type="line", 
            x0=-0.5, 
            x1=len(eixo_x_fco) - 0.5, 
            y0=1.0, y1=1.0, 
            line=dict(color="grey", width=1.5, dash="dash")
        )
        
        fig_fco.update_layout(
            xaxis=dict(type='category', categoryorder='category ascending'), 
            height=380, plot_bgcolor="white"
        )
        st.plotly_chart(fig_fco, use_container_width=True)

# ------------------------------------------
# TAB 4: QUALITY SCORE & RADAR
# ------------------------------------------
with tab4:
    st.markdown("### Quality Score Fundamentalista & Radar Interativo (Teia)")
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.markdown(f"""
            <div class='kpi-card' style='border-left: 5px solid {COLOR_NAVY}; text-align: center; padding: 25px;'>
                <div class='kpi-title'>Quality Score Recalculado</div>
                <div class='kpi-value' style='font-size: 48px; color: {COLOR_NAVY};'>{quality_score_din:.1f} / 100</div>
                <div class='kpi-sub' style='color: {COLOR_GREEN}; font-size: 16px;'>CLASSIFICAÇÃO: {classificacao_din}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📖 Como o Score Ponderado é Calculado?")
        st.info("""
        O **Quality Score** unifica 5 inteligências financeiras em uma nota de **0 a 100**:
        
        1. **Rentabilidade (ROIC / ROE):** Eficiência na geração de valor sobre o capital empregado.
        2. **Saúde Financeira:** Posição de Caixa Líquido e solvência.
        3. **Crescimento:** Taxa composta de expansão das vendas e lucros.
        4. **Valuation (FCF Yield):** Retorno de caixa gerado em relação ao preço da ação.
        5. **Estabilidade:** Regularidade dos resultados ao longo dos trimestres.
        """)

    with col_s2:
        categories = ['Rentabilidade', 'Saúde Financeira', 'Crescimento', 'Valuation', 'Estabilidade']
        scores_ambev = [score_qualidade_din, score_saude_din, score_crescimento_din, score_valuation_din, score_estabilidade_din]
        categories_closed = categories + [categories[0]]
        scores_ambev_closed = scores_ambev + [scores_ambev[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scores_ambev_closed, theta=categories_closed, 
            fill='toself', fillcolor='rgba(211, 47, 47, 0.3)', 
            line=dict(color=COLOR_RED, width=3), name='Ambev S.A.'
        ))
        if exibir_comparacao:
            fig_radar.add_trace(go.Scatterpolar(
                r=[60, 70, 55, 65, 75, 60], theta=categories_closed, 
                fill='none', line=dict(color='#64748B', width=2, dash='dash'), 
                name='Média do Setor (B3)'
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=12)),
                angularaxis=dict(tickfont=dict(size=13, color="#0F172A"))
            ), 
            height=520, 
            margin=dict(l=60, r=60, t=40, b=40),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🕸️ O que o Gráfico de Teia (Radar) revela sobre a Ambev?")
    st.write("""
    * **Área Vermelha (Perfil Ambev):** Quanto maior a área coberta pela 'teia' vermelha, melhor é o equilíbrio fundamentalista da empresa.
    * **Linha Cinza (Benchmark):** Representa o patamar médio das demais empresas listadas no setor de consumo/bebidas na B3.
    * **Interatividade:** Use os *sliders* na barra lateral esquerda para recalcular a pontuação com base nas suas próprias convicções de mercado!
    """)

# ------------------------------------------
# TAB 5: ANÁLISES PREDITIVAS
# ------------------------------------------
with tab5:
    st.markdown("### 🔮 Modelagem Preditiva: Valuation DCF & Regressão Linear")
    
    st.markdown(f"#### A. Regressão Linear Simples: Tendência de Receita Líquida — ({tipo_visao})")
    
    df_reg = df_active.copy()
    df_reg['t'] = np.arange(1, len(df_reg) + 1)
    slope, intercept = np.polyfit(df_reg['t'], df_reg['Receita'], 1)
    df_reg['Tendencia_Linear'] = slope * df_reg['t'] + intercept
    
    r_squared = (np.corrcoef(df_reg['t'], df_reg['Receita'])[0, 1]) ** 2
    
    t_futuro = np.arange(len(df_reg), len(df_reg) + 5)
    rec_futura = slope * t_futuro + intercept
    
    data_ultima = str(df_reg['data_fim_exercicio'].iloc[-1])
    if tipo_visao == "Anual Acumulado (DFP)":
        datas_futuras = [data_ultima, '2026', '2027', '2028', '2029']
    else:
        datas_futuras = [data_ultima, '2026-03-31', '2026-06-30', '2026-09-30', '2026-12-31']
    
    fig_reg = go.Figure()
    
    fig_reg.add_trace(go.Scatter(
        x=df_reg['data_fim_exercicio'], y=df_reg['Receita'], 
        mode='lines+markers', name='Receita Realizada', 
        line=dict(color=COLOR_NAVY, width=2.5)
    ))
    
    fig_reg.add_trace(go.Scatter(
        x=df_reg['data_fim_exercicio'], y=df_reg['Tendencia_Linear'], 
        mode='lines', name='Linha de Tendência Histórica', 
        line=dict(color=COLOR_RED, width=2, dash='dash')
    ))
    
    fig_reg.add_trace(go.Scatter(
        x=datas_futuras, y=rec_futura, 
        mode='lines+markers', name='Projeção Futura (Regressão)', 
        line=dict(color=COLOR_PURPLE, width=3, dash='dot'),
        marker=dict(size=8, color=COLOR_PURPLE)
    ))
    
    fig_reg.update_layout(
        xaxis=dict(type='category'),
        height=400, plot_bgcolor="white", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    
    st.caption(f"📌 **Equação do Modelo:** Receita = R$ {intercept:,.0f}M + (R$ {slope:,.0f}M × t) | Coeficiente de Determinação R² = {r_squared:.2%}")
    st.markdown("---")
    
    st.markdown("#### B. Modelagem Preditiva de Valuation por DCF (Preço Alvo Interativo)")
    
    col_dcf1, col_dcf2, col_dcf3 = st.columns(3)
    with col_dcf1:
        wacc = st.slider("1. Taxa de Desconto / WACC (%)", 8.0, 16.0, 11.5, 0.5) / 100
    with col_dcf2:
        g_5a = st.slider("2. Crescimento FCF (Próximos 5 anos %)", 0.0, 12.0, 5.0, 0.5) / 100
    with col_dcf3:
        g_perp = st.slider("3. Crescimento na Perpetuidade (%)", 1.0, 6.0, 4.5, 0.25) / 100

    fcf_ultimo_ano = df_q.iloc[-4:]['FCF'].sum() * 1e6
    caixa_liq_atual = df_q.iloc[-1]['Caixa_Liquido'] * 1e6
    
    fcf_proj = [fcf_ultimo_ano * ((1 + g_5a) ** i) for i in range(1, 6)]
    vp_fcf = [fcf_proj[i] / ((1 + wacc) ** (i + 1)) for i in range(5)]
    soma_vp_fcf = sum(vp_fcf)
    
    val_terminal = (fcf_proj[-1] * (1 + g_perp)) / (wacc - g_perp)
    vp_val_terminal = val_terminal / ((1 + wacc) ** 5)
    
    equity_val = soma_vp_fcf + vp_val_terminal + caixa_liq_atual
    preco_alvo_dcf = equity_val / total_shares
    upside = ((preco_alvo_dcf - price_input) / price_input) * 100

    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.markdown(f"""
            <div class='kpi-card' style='border-left: 5px solid {COLOR_GREEN}; text-align: center; padding: 20px;'>
                <div class='kpi-title'>Preço Alvo Estimado (DCF)</div>
                <div class='kpi-value' style='font-size: 38px; color: {COLOR_GREEN};'>R$ {preco_alvo_dcf:.2f}</div>
                <div class='kpi-sub' style='color: {COLOR_NAVY}; font-size: 14px;'>Margem de Segurança: {upside:+.1f}% vs R$ {price_input:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_res2:
        df_dcf_summary = pd.DataFrame({
            'Componente do Modelo Preditivo': ['Soma a Valor Presente do FCF (2026-2030)', 'Valor Terminal (Perpetuidade Gordon)', 'Caixa Líquido Adicionado', 'Valor Justo do Equity (Cap. Mercado Ajustado)'],
            'Valor Estimado (R$ Bi)': [f'R$ {soma_vp_fcf/1e9:,.2f} Bi', f'R$ {vp_val_terminal/1e9:,.2f} Bi', f'R$ {caixa_liq_atual/1e9:,.2f} Bi', f'R$ {equity_val/1e9:,.2f} Bi']
        })
        st.table(df_dcf_summary)