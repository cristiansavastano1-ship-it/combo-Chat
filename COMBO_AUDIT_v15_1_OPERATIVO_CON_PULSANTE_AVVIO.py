import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as str_lit

# --- CONFIGURAZIONE PAGINA ---
str_lit.set_page_config(
    page_title="COMBO Master Pro Betting & Deep Analytics Suite",
    layout="wide",
    page_icon="⚽",
)

# --- DESIGN MODERNO AVANZATO (CUSTOM CSS & GLASSMORPHISM) ---
str_lit.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header principale
str_lit.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 25px; border-radius: 20px; border: 1px solid #312e81; text-align: center; margin-bottom: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);">
        <h1 style="color: #f8fafc; font-weight: 800; font-size: 2.0rem; margin: 0; letter-spacing: -0.5px;">⚽ COMBO MASTER PRO AI</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 8px;">Suite predittiva avanzata con Poisson, Monte Carlo e Value Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

LEAGUES = {
    "Serie A (Italia)": ("SA", "football-data"),
    "Premier League (Inghilterra)": ("PL", "football-data"),
    "La Liga (Spagna)": ("PD", "football-data"),
    "Bundesliga (Germania)": ("BL1", "football-data"),
    "Ligue 1 (Francia)": ("FL1", "football-data"),
    "Eredivisie (Olanda)": ("DED", "football-data"),
    "Champions League": ("CL", "football-data"),
}

# --- SELETTORE PRINCIPALE IN EVIDENZA (OTTIMIZZATO ANCHE PER MOBILE) ---
str_lit.markdown("### 🏆 Seleziona Campionato (Mobile & Desktop)")
campionato_scelto = str_lit.selectbox("Scegli il torneo da analizzare", list(LEAGUES.keys()), label_visibility="collapsed")
selezionati_dict = {campionato_scelto: LEAGUES[campionato_scelto]}

str_lit.sidebar.markdown("### ⚙ Configurazione API")
api_key = str_lit.sidebar.text_input("🔑 API Key (football-data.org)", value="", type="password")
str_lit.sidebar.markdown("---")
str_lit.sidebar.info("💡 Suggerimento: Da cellulare puoi cambiare campionato direttamente dal menu a tendina in alto nella pagina principale.")

tab_calendario, tab_classifica, tab_value, tab_grafici, tab_ai_schedine, tab_value_finder, tab_monte_carlo, tab_audit = str_lit.tabs([
    "📅 Calendario",
    "🏆 Classifica",
    "🔍 Value Bet",
    "📊 Grafici",
    "🤖 Schedine AI",
    "⚡ Value Finder",
    "🎲 Monte Carlo",
    "🛡️ Audit"
])

# --- FUNZIONI DI SUPPORTO & MODELLAZIONE ---
@str_lit.cache_data(ttl=3600)
def scarica_dati_club(chiave, league_code):
    if not chiave: return None
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": chiave}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

@str_lit.cache_data(ttl=3600)
def scarica_classifica_club(chiave, league_code):
    if not chiave: return None
    url = f"https://api.football-data.org/v4/competitions/{league_code}/standings"
    headers = {"X-Auth-Token": chiave}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

def poisson_prob(lmbda, k):
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def calcola_statistiche_avanzate_match(sq_casa, sq_trasf, statistiche_squadre):
    dati_c = statistiche_squadre.get(sq_casa, {"media_gf": 1.4, "media_gs": 1.1})
    dati_t = statistiche_squadre.get(sq_trasf, {"media_gf": 1.1, "media_gs": 1.3})
    
    xg_c = round(dati_c["media_gf"] * 0.95 + dati_t["media_gs"] * 0.05, 2)
    xg_t = round(dati_t["media_gf"] * 0.95 + dati_c["media_gs"] * 0.05, 2)
    
    tiri_c = round(xg_c * 9.2 + 1.5, 1)
    tiri_t = round(xg_t * 9.0 + 1.4, 1)
    
    porta_c = round(tiri_c * 0.35, 1)
    porta_t = round(tiri_t * 0.33, 1)
    
    falli_c = round(12.0 + (dati_c["media_gs"] * 0.5), 1)
    falli_t = round(12.5 + (dati_t["media_gs"] * 0.5), 1)
    
    offside_c = round(1.4 + (xg_c * 0.2), 1)
    offside_t = round(1.3 + (xg_t * 0.2), 1)
    
    return {
        "Gol Attesi (xG)": (xg_c, xg_t),
        "Tiri Attesi": (tiri_c, tiri_t),
        "Tiri in Porta Attesi": (porta_c, porta_t),
        "Falli Attesi": (falli_c, falli_t),
        "Fuorigioco Attesi": (offside_c, offside_t)
    }

# --- AUDIT WALK-FORWARD: PAREGGI (X) E GOAL / NO GOAL ---
def crea_audit_walk_forward(matches_list, min_storico=30, max_gol=10):
    """Audit walk-forward usando la stessa logica Poisson del pronostico operativo.

    Per ogni partita usa solo risultati precedenti della stessa competizione.
    Le lambda sono le medie gol fatti per squadra, coerenti con il modello live
    (media_gf della classifica); la matrice 0..10 viene normalizzata prima di
    calcolare 1X2 e Goal/No Goal. max_gol resta configurabile per i test.
    """
    partite = []
    for m in matches_list:
        status = str(m.get("status", "")).upper()
        score = (m.get("score") or {}).get("fullTime") or {}
        gc, gt = score.get("home"), score.get("away")
        data = m.get("utcDate")
        if status != "FINISHED" or gc is None or gt is None or not data:
            continue
        try:
            partite.append({
                "data": pd.to_datetime(data, utc=True),
                "competizione": m.get("Competizione", "Torneo"),
                "casa": m.get("homeTeam", {}).get("name", "Casa"),
                "trasferta": m.get("awayTeam", {}).get("name", "Trasferta"),
                "gol_casa": int(gc), "gol_trasferta": int(gt),
            })
        except (TypeError, ValueError):
            continue

    partite.sort(key=lambda x: x["data"])
    storico = []
    righe = []

    def poisson_vector(lam):
        # Stessa estensione della matrice dei pronostici live: risultati da 0 a 10 gol.
        lam = max(0.01, float(lam))
        vals = np.array([
            math.exp(-lam) * (lam ** k) / math.factorial(k)
            for k in range(max_gol + 1)
        ], dtype=float)
        return vals

    for partita in partite:
        # Escludiamo tutte le partite della stessa data: scelta conservativa anti-leakage.
        precedenti = [x for x in storico
                      if x["competizione"] == partita["competizione"]
                      and x["data"].date() < partita["data"].date()]
        if len(precedenti) >= min_storico:
            casa, trasf = partita["casa"], partita["trasferta"]
            # Media gol fatti per squadra in tutte le precedenti partite,
            # corrispondente a goalsFor / playedGames usato dal modello live.
            gf_c = [x["gol_casa"] for x in precedenti if x["casa"] == casa]
            gf_c += [x["gol_trasferta"] for x in precedenti if x["trasferta"] == casa]
            gf_t = [x["gol_casa"] for x in precedenti if x["casa"] == trasf]
            gf_t += [x["gol_trasferta"] for x in precedenti if x["trasferta"] == trasf]
            lam_c = float(np.mean(gf_c)) if gf_c else 1.4
            lam_t = float(np.mean(gf_t)) if gf_t else 1.1

            ph = poisson_vector(lam_c)
            pa = poisson_vector(lam_t)
            matrice = np.outer(ph, pa)
            totale = float(matrice.sum())
            if totale <= 0:
                storico.append(partita)
                continue
            matrice /= totale  # normalizzazione unica per tutti i mercati

            p1 = float(np.tril(matrice, -1).sum())
            px = float(np.trace(matrice))
            p2 = float(np.triu(matrice, 1).sum())
            p_goal = float(matrice[1:, 1:].sum())
            p_nogoal = 1.0 - p_goal
            # Controllo indipendente: con Poisson indipendenti, P(No Goal) =
            # 1 - P(casa segna almeno un gol) * P(trasferta segna almeno un gol).
            p_nogoal_formula = 1.0 - (1.0 - math.exp(-lam_c)) * (1.0 - math.exp(-lam_t))
            esito = "1" if partita["gol_casa"] > partita["gol_trasferta"] else ("X" if partita["gol_casa"] == partita["gol_trasferta"] else "2")
            reale_nogoal = int(partita["gol_casa"] == 0 or partita["gol_trasferta"] == 0)
            # Verifica semantica indipendente dall'etichetta: No Goal se almeno una squadra segna zero.
            reale_nogoal_check = int(min(partita["gol_casa"], partita["gol_trasferta"]) == 0)
            probs = {"1": p1, "X": px, "2": p2}
            previsto = max(probs, key=probs.get)
            posizione_x = 1 + sum(probs[k] > px for k in ("1", "2"))
            brier_1x2 = sum((probs[k] - (1.0 if esito == k else 0.0)) ** 2 for k in ("1", "X", "2"))
            p_esito = max(1e-12, probs[esito])
            p_ng_real = p_nogoal if reale_nogoal else p_goal
            righe.append({
                "Data": partita["data"].strftime("%Y-%m-%d"),
                "Casa": casa, "Trasferta": trasf,
                "P(1) %": 100*p1, "P(X) %": 100*px, "P(2) %": 100*p2,
                "Segno previsto": previsto, "Esito reale": esito,
                "X prevista come segno": int(previsto == "X"),
                "Posizione X": posizione_x,
                "P(No Goal) %": 100*p_nogoal, "No Goal favorito": int(p_nogoal > p_goal),
                "No Goal reale": reale_nogoal,
                "Gol casa reali": partita["gol_casa"], "Gol trasferta reali": partita["gol_trasferta"],
                "P(No Goal) formula esatta (%)": 100 * p_nogoal_formula,
                "Delta controllo No Goal (pp)": 100 * (p_nogoal - p_nogoal_formula),
                "Controllo etichetta No Goal": int(reale_nogoal == reale_nogoal_check),
                "Somma P(1X2) (%)": 100 * (p1 + px + p2),
                "Brier 1X2": brier_1x2, "Log Loss 1X2": -math.log(p_esito),
                "Brier Goal/No Goal": (p_nogoal - reale_nogoal) ** 2,
                "Log Loss Goal/No Goal": -math.log(max(1e-12, p_ng_real)),
                "Storico pre-partita": len(precedenti),
                "Lambda casa": lam_c, "Lambda trasferta": lam_t,
            })
        storico.append(partita)

    return pd.DataFrame(righe)


# --- ACQUISIZIONE DATI ---
statistiche_squadre = {}
matches_raw = []

for c_nome, (codice_lega, tipo_fonte) in selezionati_dict.items():
    if api_key:
        dati = scarica_dati_club(api_key, codice_lega)
        dati_classifica = scarica_classifica_club(api_key, codice_lega)
        if dati and "matches" in dati: 
            for m in dati["matches"]:
                m["Competizione"] = c_nome
                matches_raw.append(m)
        if dati_classifica and "standings" in dati_classifica:
            for s in dati_classifica["standings"]:
                for riga in s.get("table", []):
                    nome_sq = riga["team"]["name"]
                    giocate = max(riga["playedGames"], 1)
                    statistiche_squadre[nome_sq] = {
                        "media_gf": riga["goalsFor"] / giocate,
                        "media_gs": riga["goalsAgainst"] / giocate,
                        "punti": riga["points"],
                        "forma": "N/D",
                        "Competizione": c_nome
                    }

# --- TAB 1: CALENDARIO & STUDIO ---
with tab_calendario:
    if matches_raw:
        lista = []
        for m in matches_raw:
            g_c = m["score"]["fullTime"].get("home") if m.get("score") and m["score"].get("fullTime") else None
            g_t = m["score"]["fullTime"].get("away") if m.get("score") and m["score"].get("fullTime") else None
            lista.append({
                "Competizione": m.get("Competizione", "Torneo"),
                "giornata": m.get("matchday", 0), "casa": m["homeTeam"]["name"],
                "trasferta": m["awayTeam"]["name"], "data": m["utcDate"][:10],
                "ora": m["utcDate"][11:16], "gol_casa": g_c if g_c is not None else "-",
                "gol_trasf": g_t if g_t is not None else "-", "stato": m["status"],
            })
        df = pd.DataFrame(lista)
        
        c_f1, c_f2 = str_lit.columns(2)
        competizioni_disponibili = sorted(df["Competizione"].unique())
        comp_sel = c_f1.selectbox("🏆 Torneo", competizioni_disponibili, key="mobile_comp_select")
        
        df_comp = df[df["Competizione"] == comp_sel]
        giornate = sorted(df_comp["giornata"].unique())
        if giornate:
            giornata_sel = c_f2.selectbox("📅 Giornata", giornate, key="mobile_giornata_select")
            for idx, row in df_comp[df_comp["giornata"] == giornata_sel].iterrows():
                str_lit.markdown(
                    f"""
                    <div style="background-color: #111827; padding: 18px; border-radius: 12px; border: 1px solid #1f2937; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6;">{row['casa']}</span> 
                            <span style="color: #94a3b8; margin: 0 8px;">vs</span> 
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6;">{row['trasferta']}</span><br>
                            <span style="color: #64748b; font-size: 12px;">📅 {row['data']} ore {row['ora']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="background-color: #1e293b; color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 1.1rem;">{row['gol_casa']} - {row['gol_trasf']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if str_lit.button("📊 Analizza Match", key=f"btn_{idx}"):
                    str_lit.session_state["match_attivo"] = row
    else:
        str_lit.info("👈 Inserisci la chiave API nella barra laterale o seleziona un campionato valido.")

    if "match_attivo" in str_lit.session_state:
        m = str_lit.session_state["match_attivo"]
        sq_c, sq_t = m["casa"], m["trasferta"]
        lam_c = statistiche_squadre.get(sq_c, {}).get("media_gf", 1.4)
        lam_t = statistiche_squadre.get(sq_t, {}).get("media_gf", 1.1)

        p_c, p_p, p_t = 0.0, 0.0, 0.0
        prob_under_over = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0, 3.5: 0.0, 4.5: 0.0}
        prob_btts_yes = 0.0
        matrice_risultati = np.zeros((11, 11))

        for rc in range(11):
            for rt in range(11):
                prob = poisson_prob(lam_c, rc) * poisson_prob(lam_t, rt)
                matrice_risultati[rc, rt] = prob
                
                if rc > rt: p_c += prob
                elif rc == rt: p_p += prob
                else: p_t += prob
                
                tot_gol = rc + rt
                for soglia in prob_under_over:
                    if tot_gol <= soglia:
                        prob_under_over[soglia] += prob
                        
                if rc > 0 and rt > 0:
                    prob_btts_yes += prob

        tot = p_c + p_p + p_t
        # La matrice considera i punteggi 0..10: normalizziamo anche BTTS,
        # altrimenti la massa Poisson oltre 5 gol viene conteggiata erroneamente come No Goal.
        prob_btts_yes = prob_btts_yes / tot if tot > 0 else 0.0
        prob_btts_no = 1.0 - prob_btts_yes
        p_c, p_p, p_t = (p_c/tot)*100, (p_p/tot)*100, (p_t/tot)*100

        stats_match = calcola_statistiche_avanzate_match(sq_c, sq_t, statistiche_squadre)

        str_lit.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #312e81; border-radius: 20px; padding: 25px; margin-top: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 15px; margin-bottom: 20px;">
                    <span style="font-weight: 800; font-size: 1.2rem; color: #f8fafc;">📋 SCHEDA DINAMICA DELLA PARTITA</span>
                    <span style="color: #94a3b8; font-size: 0.95rem;">📅 {m.get('data', 'N/D')} - {m.get('ora', '')}</span>
                </div>
            """,
            unsafe_allow_html=True
        )

        col_sq, col_st = str_lit.columns([1.2, 1.8])
        with col_sq:
            str_lit.markdown(
                f"""
                <div style="text-align: center; padding: 30px 20px; background: rgba(15, 23, 42, 0.6); border-radius: 14px; border: 1px solid #1e293b;">
                    <h2 style="color: #38bdf8; margin: 0; font-size: 1.4rem;">{sq_c}</h2>
                    <p style="color: #64748b; margin: 5px 0 20px 0; font-size: 0.85rem;">CASA</p>
                    <h3 style="color: #a855f7; margin: 0; font-size: 1.2rem;">VS</h3>
                    <h2 style="color: #f43f5e; margin: 20px 0 0 0; font-size: 1.4rem;">{sq_t}</h2>
                    <p style="color: #64748b; margin: 5px 0 0 0; font-size: 0.85rem;">TRASFERTA</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_st:
            str_lit.markdown("<p style='text-align: center; font-weight: 700; color: #94a3b8; margin-bottom: 15px; letter-spacing: 0.5px;'>STATISTICHE PREVISTE (LIVE CALC)</p>", unsafe_allow_html=True)
            for label, (val_c, val_t) in stats_match.items():
                str_lit.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; background: #111827; padding: 12px 18px; border-radius: 10px; border: 1px solid #1f2937; margin-bottom: 10px;">
                        <span style="font-weight: 700; color: #38bdf8; font-size: 1.1rem; width: 60px; text-align: left;">{val_c}</span>
                        <span style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-align: center; flex-grow: 1;">{label}</span>
                        <span style="font-weight: 700; color: #f43f5e; font-size: 1.1rem; width: 60px; text-align: right;">{val_t}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        str_lit.markdown("<br>", unsafe_allow_html=True)
        str_lit.markdown(f"<h3>🔬 Analisi Avanzata Poisson & Mercati: {sq_c} vs {sq_t}</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = str_lit.columns(3)
        col1.metric("Segno 1 (Casa)", f"{p_c:.1f}%", f"Quota equa: {100/p_c:.2f}")
        col2.metric("Segno X (Pareggio)", f"{p_p:.1f}%", f"Quota equa: {100/p_p:.2f}")
        col3.metric("Segno 2 (Trasferta)", f"{p_t:.1f}%", f"Quota equa: {100/p_t:.2f}")
        
        str_lit.markdown("<br>", unsafe_allow_html=True)
        str_lit.markdown("#### ⚽ Analisi Under / Over & Goal / No Goal")
        
        uo_col1, uo_col2, uo_col3, uo_col4, uo_col5 = str_lit.columns(5)
        uo_col1.metric("Over 2.5", f"{(1 - prob_under_over[2.5])*100:.1f}%", f"Under: {prob_under_over[2.5]*100:.1f}%")
        uo_col2.metric("Over 1.5", f"{(1 - prob_under_over[1.5])*100:.1f}%", f"Under: {prob_under_over[1.5]*100:.1f}%")
        uo_col3.metric("Over 3.5", f"{(1 - prob_under_over[3.5])*100:.1f}%", f"Under: {prob_under_over[3.5]*100:.1f}%")
        uo_col4.metric("Goal (BTTS)", f"{prob_btts_yes*100:.1f}%", f"Quota: {100/(prob_btts_yes*100):.2f}" if prob_btts_yes > 0 else "N.D.")
        uo_col5.metric("No Goal", f"{prob_btts_no*100:.1f}%", f"Quota: {100/(prob_btts_no*100):.2f}" if prob_btts_no > 0 else "N.D.")

        str_lit.markdown("<br>", unsafe_allow_html=True)
        str_lit.markdown("#### 🎯 Top 3 Risultati Esatti più Probabili")
        
        lista_esatti = []
        for rc in range(11):
            for rt in range(11):
                lista_esatti.append(((rc, rt), matrice_risultati[rc, rt]))
        lista_esatti.sort(key=lambda x: x[1], reverse=True)
        
        c_res1, c_res2, c_res3 = str_lit.columns(3)
        with c_res1: str_lit.info(f"🥇 1° Esatto: **{lista_esatti[0][0][0]} - {lista_esatti[0][0][1]}** ({lista_esatti[0][1]*100:.1f}%)")
        with c_res2: str_lit.info(f"🥈 2° Esatto: **{lista_esatti[1][0][0]} - {lista_esatti[1][0][1]}** ({lista_esatti[1][1]*100:.1f}%)")
        with c_res3: str_lit.info(f"🥉 3° Esatto: **{lista_esatti[2][0][0]} - {lista_esatti[2][0][1]}** ({lista_esatti[2][1]*100:.1f}%)")
        
        str_lit.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: CLASSIFICA ---
with tab_classifica:
    str_lit.subheader("🏆 Classifica e Rendimento")
    if statistiche_squadre:
        df_cls = pd.DataFrame([{
            "Competizione": v.get("Competizione", "Torneo"),
            "Squadra": k, "Punti": v["punti"], 
            "Media GF": round(v["media_gf"], 2), 
            "Media GS": round(v["media_gs"], 2), 
            "DR": round(v["media_gf"] - v["media_gs"], 2)
        } for k, v in sorted(statistiche_squadre.items(), key=lambda x: x[1]["punti"], reverse=True)])
        str_lit.dataframe(df_cls, use_container_width=True, hide_index=True)
        str_lit.download_button("📥 Scarica CSV", df_cls.to_csv(index=False).encode("utf-8"), "classifica.csv", "text/csv")
    else:
        str_lit.warning("Dati non disponibili.")

# --- TAB 3: VALUE BET ---
with tab_value:
    str_lit.subheader("🔍 Calcolatore Value Bet & Kelly Criterion")
    c1, c2, c3 = str_lit.columns(3)
    p_stim = c1.slider("Probabilità Modello (%)", 1.0, 100.0, 45.0)
    q_book = c2.number_input("Quota Bookmaker", 1.01, 50.0, 2.30)
    bankroll = c3.number_input("Bankroll (€)", 10.0, 100000.0, 1000.0)
    
    q_equa = 100 / p_stim
    ev = ((p_stim / 100) * q_book) - 1
    b = q_book - 1
    p = p_stim / 100
    q = 1 - p
    kelly_fraction = max(0.0, ((b * p - q) / b)) * 0.25
    stake = bankroll * kelly_fraction

    m1, m2, m3 = str_lit.columns(3)
    m1.metric("Quota Equa", f"{q_equa:.2f}")
    m2.metric("Valore Atteso (EV)", f"{ev*100:+.2f}%")
    m3.metric("Stake (Kelly 25%)", f"€{stake:.2f}")

    if q_book > q_equa:
        str_lit.success("🔥 VALUE BET CERTIFICATA! Edge Positivo.")
    else:
        str_lit.error("❌ Nessun valore rilevato.")

# --- TAB 4: GRAFICI ---
with tab_grafici:
    str_lit.subheader("📊 Trend e Prestazioni")
    if statistiche_squadre:
        df_g = pd.DataFrame([{"Squadra": k, "Punti": v["punti"], "Media Gol Fatti": v["media_gf"]} for k, v in statistiche_squadre.items()])
        fig = px.bar(df_g, x="Squadra", y="Punti", color="Media Gol Fatti", template="plotly_dark", title="Punti e Potenziale Offensivo")
        str_lit.plotly_chart(fig, use_container_width=True)

def genera_dataset_valore(matches_list, stats_dict):
    righe = []
    for m in matches_list:
        h, a = m["homeTeam"]["name"], m["awayTeam"]["name"]
        if h in stats_dict and a in stats_dict:
            lc = stats_dict[h]["media_gf"]
            lt = stats_dict[a]["media_gf"]
            
            p_c, p_p, p_t = 0.0, 0.0, 0.0
            for rc in range(11):
                for rt in range(11):
                    prob = poisson_prob(lc, rc) * poisson_prob(lt, rt)
                    if rc > rt: p_c += prob
                    elif rc == rt: p_p += prob
                    else: p_t += prob
            
            tot_prob = p_c + p_p + p_t
            if tot_prob > 0:
                p_c, p_p, p_t = p_c / tot_prob, p_p / tot_prob, p_t / tot_prob
            
            esiti = [
                (f"1 ({h})", p_c),
                (f"X (Pareggio)", p_p),
                (f"2 ({a})", p_t)
            ]
            esiti.sort(key=lambda x: x[1], reverse=True)
            miglior_selezione, prob_scelta = esiti[0]
            
            q_book = round(1.03 / max(0.01, prob_scelta), 2)
            edge = round((prob_scelta * q_book - 1) * 100, 1)
            
            righe.append({
                "Competizione": m.get("Competizione", "Torneo"),
                "Giornata": m.get("matchday", 1),
                "Partita": f"{h} vs {a}", 
                "Selezione": miglior_selezione, 
                "Prob_Modello": round(prob_scelta * 100, 1), 
                "Quota_Book": q_book, 
                "Edge": edge
            })
            
    if not righe:
        return pd.DataFrame(columns=["Competizione", "Giornata", "Partita", "Selezione", "Prob_Modello", "Quota_Book", "Edge"])
    return pd.DataFrame(righe)

df_val = genera_dataset_valore(matches_raw, statistiche_squadre)

# --- TAB 5: SCHEDINE AI ---
with tab_ai_schedine:
    str_lit.subheader("🤖 Schedine Smart & Combo AI")
    if not df_val.empty and "Competizione" in df_val.columns:
        c_s1, c_s2 = str_lit.columns(2)
        comp_schedina = c_s1.selectbox("Torneo", sorted(df_val["Competizione"].unique()), key="mobile_schedina_torneo")
        df_comp_val = df_val[df_val["Competizione"] == comp_schedina]
        giornate_val = sorted(df_comp_val["Giornata"].unique())
        if giornate_val:
            giornata_schedina = c_s2.selectbox("Giornata", giornate_val, key="mobile_schedina_giornata")
            df_filtrato_giornata = df_comp_val[df_comp_val["Giornata"] == giornata_schedina]
            if not df_filtrato_giornata.empty:
                num_ev = str_lit.slider("Eventi in Multipla", 2, min(6, len(df_filtrato_giornata)), min(3, len(df_filtrato_giornata)))
                subset = df_filtrato_giornata.sort_values(by="Edge", ascending=False).head(num_ev)
                quota_tot = np.prod(subset['Quota_Book'].values)
                str_lit.metric("📈 Quota Totale Accumulatore", f"{quota_tot:.2f}")
                str_lit.dataframe(subset, use_container_width=True, hide_index=True)

# --- TAB 6: VALUE FINDER ---
with tab_value_finder:
    str_lit.subheader("⚡ Scanner Value Finder")
    if not df_val.empty:
        min_edge = str_lit.slider("Edge Minimo (%)", 0.0, 25.0, 3.0)
        str_lit.dataframe(df_val[df_val["Edge"] >= min_edge], use_container_width=True, hide_index=True)
    else:
        str_lit.info("Nessun incontro disponibile.")

# --- TAB 7: MONTE CARLO ---
with tab_monte_carlo:
    str_lit.subheader("🎲 Simulatore Monte Carlo Avanzato")
    if statistiche_squadre:
        nomi = sorted(list(statistiche_squadre.keys()))
        c1, c2, c3 = str_lit.columns(3)
        sq_c = c1.selectbox("Casa", nomi, index=0, key="mc_casa")
        sq_t = c2.selectbox("Trasferta", nomi, index=min(1, len(nomi)-1), key="mc_trasf")
        iterazioni = c3.slider("Iterazioni", 1000, 20000, 5000, step=1000, key="mc_iter")
        
        if str_lit.button("🚀 Esegui Simulazione", key="btn_mc"):
            lc = statistiche_squadre[sq_c]["media_gf"] * 1.05
            lt = statistiche_squadre[sq_t]["media_gf"] * 0.95
            
            gc_sim = np.random.poisson(lc, iterazioni)
            gt_sim = np.random.poisson(lt, iterazioni)
            
            v_c = np.sum(gc_sim > gt_sim) / (iterazioni / 100)
            v_p = np.sum(gc_sim == gt_sim) / (iterazioni / 100)
            v_t = np.sum(gc_sim < gt_sim) / (iterazioni / 100)
            
            tot_gol = gc_sim + gt_sim
            p_over15 = np.sum(tot_gol > 1.5) / (iterazioni / 100)
            p_over25 = np.sum(tot_gol > 2.5) / (iterazioni / 100)
            p_btts = np.sum((gc_sim > 0) & (gt_sim > 0)) / (iterazioni / 100)
            
            str_lit.markdown("#### 📊 Esiti 1X2 dalle Simulazioni")
            m1, m2, m3 = str_lit.columns(3)
            m1.metric("Vittoria Casa (1)", f"{v_c:.1f}%", f"Quota equa: {100/v_c:.2f}" if v_c > 0 else "N.D.")
            m2.metric("Pareggio (X)", f"{v_p:.1f}%", f"Quota equa: {100/v_p:.2f}" if v_p > 0 else "N.D.")
            m3.metric("Vittoria Trasferta (2)", f"{v_t:.1f}%", f"Quota equa: {100/v_t:.2f}" if v_t > 0 else "N.D.")
            
            str_lit.markdown("<br>", unsafe_allow_html=True)
            str_lit.markdown("#### ⚽ Mercati di Gol (Simulati)")
            uo1, uo2, uo3 = str_lit.columns(3)
            uo1.metric("Over 1.5 Gol", f"{p_over15:.1f}%")
            uo2.metric("Over 2.5 Gol", f"{p_over25:.1f}%")
            uo3.metric("Goal (BTTS)", f"{p_btts:.1f}%")
            
            str_lit.markdown("<br>", unsafe_allow_html=True)
            str_lit.markdown("#### 🎯 Top 3 Risultati Esatti più Frequenti")
            
            risultati_coppie = list(zip(gc_sim, gt_sim))
            conteggio_esatti = pd.Series(risultati_coppie).value_counts().head(3)
            
            r_col1, r_col2, r_col3 = str_lit.columns(3)
            colonne_res = [r_col1, r_col2, r_col3]
            
            for idx, ((rc, rt), count) in enumerate(conteggio_esatti.items()):
                perc = (count / iterazioni) * 100
                with colonne_res[idx]:
                    medaglia = ["🥇", "🥈", "🥉"][idx]
                    str_lit.info(f"{medaglia} **{rc} - {rt}** ({perc:.1f}% delle volte)")
            
            str_lit.markdown("<br>", unsafe_allow_html=True)
            df_dist = pd.DataFrame({"Gol Totali": tot_gol})
            fig_mc = px.histogram(df_dist, x="Gol Totali", nbins=int(max(tot_gol))+1, title="Distribuzione Frequenza Gol Totali (Iterazioni Monte Carlo)", template="plotly_dark")
            str_lit.plotly_chart(fig_mc, use_container_width=True)

# --- TAB 8: AUDIT ---
with tab_audit:
    str_lit.subheader("🛡️ Audit statistico: Pareggio X e Goal / No Goal")
    str_lit.caption(
        "Controllo walk-forward: per ogni partita storica le probabilità sono stimate usando soltanto "
        "risultati precedenti della stessa competizione. Nessuna percentuale viene forzata per aumentare X o No Goal."
    )
    df_audit = crea_audit_walk_forward(matches_raw)
    if df_audit.empty:
        str_lit.warning(
            "Non ci sono ancora abbastanza partite concluse per un audit walk-forward. "
            "Servono almeno 30 risultati precedenti nella competizione prima di valutare una partita."
        )
    else:
        # V9: audit walk-forward esteso a 0..10, allineato alla matrice del pronostico live.
        str_lit.markdown("#### 🔎 V9 — Audit allineato alla matrice Poisson 0–10")
        str_lit.caption(
            "Confronta la probabilità No Goal della matrice 0–10 normalizzata con la formula Poisson analitica; "
            "controlla inoltre l'etichetta reale e la somma delle probabilità 1X2. Non modifica le previsioni."
        )
        max_delta_ng = pd.to_numeric(df_audit["Delta controllo No Goal (pp)"], errors="coerce").abs().max()
        etichette_ok = int(pd.to_numeric(df_audit["Controllo etichetta No Goal"], errors="coerce").fillna(0).sum())
        somma_1x2_max_err = (pd.to_numeric(df_audit["Somma P(1X2) (%)"], errors="coerce") - 100).abs().max()
        k1, k2, k3, k4 = str_lit.columns(4)
        k1.metric("Partite controllate", f"{len(df_audit)}")
        k2.metric("Etichette No Goal coerenti", f"{etichette_ok}/{len(df_audit)}")
        k3.metric("Max differenza formula NG", f"{max_delta_ng:.4f} pp")
        k4.metric("Max errore somma 1X2", f"{somma_1x2_max_err:.6f} pp")
        if etichette_ok == len(df_audit) and somma_1x2_max_err < 0.01:
            str_lit.success("Controlli di coerenza superati: etichette No Goal e somma 1X2 corrette nel campione.")
        else:
            str_lit.error("Attenzione: almeno un controllo di coerenza non è superato. Verificare le righe dettagliate.")
        with str_lit.expander("Dettaglio tecnico per partita", expanded=False):
            cols_check = ["Data", "Casa", "Trasferta", "Gol casa reali", "Gol trasferta reali", "P(No Goal) %", "P(No Goal) formula esatta (%)", "Delta controllo No Goal (pp)", "No Goal reale", "Controllo etichetta No Goal", "Somma P(1X2) (%)", "Storico pre-partita"]
            str_lit.dataframe(df_audit[cols_check].round(4), use_container_width=True, hide_index=True)
        n = len(df_audit)
        freq_x_reale = (df_audit["Esito reale"] == "X").mean() * 100
        prob_x_media = df_audit["P(X) %"].mean()
        freq_x_prevista = df_audit["X prevista come segno"].mean() * 100
        freq_ng_reale = df_audit["No Goal reale"].mean() * 100
        prob_ng_media = df_audit["P(No Goal) %"].mean()
        freq_ng_favorito = df_audit["No Goal favorito"].mean() * 100

        a1, a2, a3 = str_lit.columns(3)
        a1.metric("Partite valutate", f"{n}", "solo walk-forward")
        a2.metric("Pareggi reali", f"{freq_x_reale:.1f}%", f"P(X) media {prob_x_media:.1f}% · scarto {prob_x_media-freq_x_reale:+.1f} pp")
        a3.metric("Segno X più probabile", f"{freq_x_prevista:.1f}%", "frequenza delle previsioni X")

        b1, b2, b3 = str_lit.columns(3)
        b1.metric("No Goal reali", f"{freq_ng_reale:.1f}%")
        b2.metric("P(No Goal) media", f"{prob_ng_media:.1f}%", f"scarto {prob_ng_media-freq_ng_reale:+.1f} pp")
        b3.metric("No Goal favorito", f"{freq_ng_favorito:.1f}%", "partite in cui P(No Goal) > P(Goal)")

        c1, c2, c3, c4 = str_lit.columns(4)
        c1.metric("Brier 1X2", f"{df_audit['Brier 1X2'].mean():.4f}", "più basso è meglio")
        c2.metric("Log Loss 1X2", f"{df_audit['Log Loss 1X2'].mean():.4f}", "più basso è meglio")
        c3.metric("Brier Goal/No Goal", f"{df_audit['Brier Goal/No Goal'].mean():.4f}", "più basso è meglio")
        c4.metric("Log Loss Goal/No Goal", f"{df_audit['Log Loss Goal/No Goal'].mean():.4f}", "più basso è meglio")

        str_lit.markdown("#### 📈 Probabilità previste vs frequenze reali")
        confronto = pd.DataFrame({
            "Indicatore": ["Pareggio X", "No Goal"],
            "Probabilità media modello (%)": [prob_x_media, prob_ng_media],
            "Frequenza reale (%)": [freq_x_reale, freq_ng_reale],
        })
        fig_audit = px.bar(
            confronto, x="Indicatore", y=["Probabilità media modello (%)", "Frequenza reale (%)"],
            barmode="group", template="plotly_dark", title="Calibrazione empirica — modello contro risultati reali"
        )
        str_lit.plotly_chart(fig_audit, use_container_width=True)

        # Diagnostica aggiuntiva: frequenza reale dentro fasce di probabilità.
        str_lit.markdown("#### 🔬 Calibrazione per fasce di probabilità")
        fasce = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.000001]
        etichette = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]

        df_x_bin = df_audit[["P(X) %", "Esito reale"]].copy()
        df_x_bin["X reale"] = (df_x_bin["Esito reale"] == "X").astype(int)
        df_x_bin["Fascia probabilità"] = pd.cut(df_x_bin["P(X) %"].clip(0, 100), bins=fasce, labels=etichette, right=True, include_lowest=True)
        cal_x = df_x_bin.groupby("Fascia probabilità", observed=False).agg(
            Partite=("X reale", "size"), **{"Probabilità media (%)": ("P(X) %", "mean"), "Frequenza reale (%)": ("X reale", lambda x: 100*x.mean())}
        ).reset_index()
        cal_x = cal_x[cal_x["Partite"] > 0].copy()
        cal_x["Scarto (pp)"] = cal_x["Probabilità media (%)"] - cal_x["Frequenza reale (%)"]
        str_lit.caption("Pareggio X: la frequenza reale in tabella è la quota di pareggi osservati dentro ciascuna fascia P(X).")
        str_lit.dataframe(cal_x.round(2), use_container_width=True, hide_index=True)


        df_ng_bin = df_audit[["P(No Goal) %", "No Goal reale"]].copy()
        df_ng_bin["Fascia probabilità"] = pd.cut(df_ng_bin["P(No Goal) %"].clip(0, 100), bins=fasce, labels=etichette, right=True, include_lowest=True)
        cal_ng = df_ng_bin.groupby("Fascia probabilità", observed=False).agg(
            Partite=("No Goal reale", "size"), **{"Probabilità media (%)": ("P(No Goal) %", "mean"), "Frequenza reale (%)": ("No Goal reale", lambda x: 100*x.mean())}
        ).reset_index()
        cal_ng = cal_ng[cal_ng["Partite"] > 0].copy()
        cal_ng["Scarto (pp)"] = cal_ng["Probabilità media (%)"] - cal_ng["Frequenza reale (%)"]
        str_lit.caption("Goal / No Goal: frequenza reale = partite in cui almeno una delle due squadre non ha segnato.")
        str_lit.dataframe(cal_ng.round(2), use_container_width=True, hide_index=True)

        str_lit.markdown("#### 🧭 Perché il segno X viene scelto poco? ")
        posizione_x = df_audit["Posizione X"].value_counts().reindex([1, 2, 3], fill_value=0)
        x1, x2, x3 = str_lit.columns(3)
        x1.metric("X al 1° posto", f"{posizione_x.loc[1]}", f"{posizione_x.loc[1]/n*100:.1f}% delle partite")
        x2.metric("X al 2° posto", f"{posizione_x.loc[2]}", f"{posizione_x.loc[2]/n*100:.1f}% delle partite")
        x3.metric("X al 3° posto", f"{posizione_x.loc[3]}", f"{posizione_x.loc[3]/n*100:.1f}% delle partite")

        # V4: diagnostica della X per equilibrio tra probabilità di vittoria casa/trasferta.
        # Solo analisi: non modifica probabilità operative, lambda o pronostici.
        str_lit.markdown("#### ⚖️ Diagnostica X in base all'equilibrio tra P(1) e P(2)")
        str_lit.caption(
            "Le partite sono raggruppate in base alla differenza assoluta tra P(1) e P(2). "
            "È una diagnostica retrospettiva: non modifica il modello."
        )
        df_eq = df_audit.copy()
        df_eq["Differenza P(1)-P(2) (pp)"] = (df_eq["P(1) %"] - df_eq["P(2) %"]).abs()
        df_eq["X reale (0/1)"] = (df_eq["Esito reale"] == "X").astype(int)
        df_eq["P(X) (0-1)"] = (df_eq["P(X) %"] / 100.0).clip(0.0, 1.0)
        df_eq["Brier X"] = (df_eq["P(X) (0-1)"] - df_eq["X reale (0/1)"]) ** 2
        df_eq["Gruppo equilibrio"] = pd.cut(
            df_eq["Differenza P(1)-P(2) (pp)"],
            bins=[-0.000001, 10, 25, float("inf")],
            labels=["Equilibrate (≤10 pp)", "Favorita moderata (>10–25 pp)", "Favorita netta (>25 pp)"],
            right=True,
            include_lowest=True,
        )
        riepilogo_eq = df_eq.groupby("Gruppo equilibrio", observed=False).agg(
            Partite=("X reale (0/1)", "size"),
            **{
                "Differenza media P(1)-P(2) (pp)": ("Differenza P(1)-P(2) (pp)", "mean"),
                "P(X) media (%)": ("P(X) %", "mean"),
                "Pareggi reali (%)": ("X reale (0/1)", lambda x: 100 * x.mean()),
                "Brier X": ("Brier X", "mean"),
            },
        ).reset_index()
        riepilogo_eq["Scarto P(X) - pareggi reali (pp)"] = (
            riepilogo_eq["P(X) media (%)"] - riepilogo_eq["Pareggi reali (%)"]
        )
        riepilogo_eq["X al 1° posto (%)"] = riepilogo_eq["Gruppo equilibrio"].map(
            df_eq.groupby("Gruppo equilibrio", observed=False)["Posizione X"].apply(lambda x: 100 * (x == 1).mean())
        )
        str_lit.dataframe(riepilogo_eq.round(2), use_container_width=True, hide_index=True)
        str_lit.caption(
            "Brier X = errore quadratico tra P(X) e l'esito pareggio (0/1); più basso è meglio. "
            "I gruppi con poche partite vanno interpretati con cautela."
        )

        # V6: diagnostica No Goal per fasce di lambda totale attesa, con dettagli aggiuntivi.
        # Solo analisi retrospettiva: non cambia le probabilità o i pronostici operativi.
        str_lit.markdown("#### 🥅 Diagnostica No Goal per gol attesi")
        str_lit.caption(
            "Raggruppa le partite in base alla somma Lambda casa + Lambda trasferta. "
            "Serve a capire se la sovrastima No Goal si concentra in partite con pochi, medi o molti gol attesi; "
            "non applica correzioni al modello."
        )
        colonne_lambda = {"Lambda casa", "Lambda trasferta", "P(No Goal) %", "No Goal reale", "Brier Goal/No Goal"}
        if colonne_lambda.issubset(df_audit.columns):
            df_ng_tipo = df_audit.copy()
            df_ng_tipo["Gol attesi totali (lambda) "] = (
                pd.to_numeric(df_ng_tipo["Lambda casa"], errors="coerce")
                + pd.to_numeric(df_ng_tipo["Lambda trasferta"], errors="coerce")
            )
            df_ng_tipo["P(No Goal) (0-1)"] = (pd.to_numeric(df_ng_tipo["P(No Goal) %"], errors="coerce") / 100).clip(0, 1)
            df_ng_tipo["No Goal reale (0/1)"] = pd.to_numeric(df_ng_tipo["No Goal reale"], errors="coerce")
            df_ng_tipo["Gruppo gol attesi"] = pd.cut(
                df_ng_tipo["Gol attesi totali (lambda) "],
                bins=[-float("inf"), 2.0, 2.8, float("inf")],
                labels=["Bassi (≤2.0)", "Medi (>2.0–2.8)", "Alti (>2.8)"],
                right=True,
                include_lowest=True,
            )
            df_ng_tipo = df_ng_tipo.dropna(subset=["Gruppo gol attesi", "No Goal reale (0/1)", "P(No Goal) (0-1)"])
            if not df_ng_tipo.empty:
                df_ng_tipo["No Goal favorito (0/1)"] = (
                    pd.to_numeric(df_ng_tipo["P(No Goal) %"], errors="coerce") > 50.0
                ).astype(int)
                riepilogo_ng_tipo = df_ng_tipo.groupby("Gruppo gol attesi", observed=False).agg(
                    Partite=("No Goal reale (0/1)", "size"),
                    **{
                        "Lambda casa media": ("Lambda casa", "mean"),
                        "Lambda trasferta media": ("Lambda trasferta", "mean"),
                        "Lambda totale media": ("Gol attesi totali (lambda) ", "mean"),
                        "P(No Goal) media (%)": ("P(No Goal) %", "mean"),
                        "No Goal reali (%)": ("No Goal reale (0/1)", lambda x: 100 * x.mean()),
                        "No Goal favorito (%)": ("No Goal favorito (0/1)", lambda x: 100 * x.mean()),
                        "Brier No Goal": ("Brier Goal/No Goal", "mean"),
                    },
                ).reset_index()
                riepilogo_ng_tipo["Scarto (pp)"] = (
                    riepilogo_ng_tipo["P(No Goal) media (%)"] - riepilogo_ng_tipo["No Goal reali (%)"]
                )
                riepilogo_ng_tipo = riepilogo_ng_tipo[[
                    "Gruppo gol attesi", "Partite", "Lambda casa media", "Lambda trasferta media",
                    "Lambda totale media", "P(No Goal) media (%)", "No Goal reali (%)",
                    "No Goal favorito (%)", "Scarto (pp)", "Brier No Goal"
                ]]
                str_lit.dataframe(riepilogo_ng_tipo.round(2), use_container_width=True, hide_index=True)
                str_lit.caption(
                    "Brier No Goal è riportato dal calcolo walk-forward già presente nell'audit. "
                    "Le fasce con poche partite sono descrittive e non bastano da sole a giustificare una modifica."
                )
            else:
                str_lit.info("Non ci sono righe valide sufficienti per la diagnostica No Goal per lambda.")
        else:
            str_lit.warning("Diagnostica No Goal per lambda non disponibile: mancano una o più colonne attese nell'audit.")

        scarto_x = prob_x_media - freq_x_reale
        scarto_ng = prob_ng_media - freq_ng_reale
        if n < 100:
            str_lit.warning(f"Campione ancora limitato ({n} partite): considera questi risultati preliminari, non una prova definitiva di calibrazione.")
        if abs(scarto_x) >= 7:
            str_lit.info("Pareggi X: lo scarto tra probabilità media e frequenza reale è ampio. Prima di modificare il modello, confrontare anche le fasce di probabilità e le singole competizioni.")
        else:
            str_lit.success("Pareggi X: sul campione disponibile lo scarto aggregato non supera 7 punti percentuali. Controllare comunque il dettaglio per fascia e campionato.")
        if abs(scarto_ng) >= 7:
            str_lit.info("No Goal: lo scarto tra probabilità media e frequenza reale è ampio. Non applicare correzioni fisse; verificare prima la calibrazione per fasce di probabilità.")
        else:
            str_lit.success("No Goal: sul campione disponibile lo scarto aggregato non supera 7 punti percentuali. Controllare comunque il dettaglio per fascia e campionato.")

        with str_lit.expander("🔎 Dettaglio partite usate nell'audit"):
            str_lit.dataframe(df_audit, use_container_width=True, hide_index=True)
            str_lit.download_button(
                "📥 Scarica audit walk-forward CSV",
                df_audit.to_csv(index=False).encode("utf-8-sig"),
                "audit_walk_forward_x_nogoal.csv", "text/csv"
            )

