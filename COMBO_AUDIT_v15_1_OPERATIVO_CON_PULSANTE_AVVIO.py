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
def crea_audit_walk_forward(matches_list, min_storico=30, max_gol=8):
    """Valuta probabilità storiche usando esclusivamente partite precedenti alla data analizzata."""
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

    def media(valori, fallback):
        return float(np.mean(valori)) if valori else float(fallback)

    def media_ridotta(valori, fallback, peso_prior=5.0):
        # Shrinkage verso la media del campionato per evitare medie estreme con pochi dati.
        n = len(valori)
        return (sum(valori) + peso_prior * fallback) / (n + peso_prior) if n else float(fallback)

    def poisson_vector(lam):
        lam = max(0.05, min(float(lam), 5.0))
        vals = np.array([math.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(max_gol + 1)], dtype=float)
        s = vals.sum()
        return vals / s if s > 0 else np.eye(1, max_gol + 1, 0).ravel()

    for partita in partite:
        # Non usare mai la partita corrente né risultati successivi per stimare le probabilità.
        precedenti = [x for x in storico if x["competizione"] == partita["competizione"] and x["data"].date() < partita["data"].date()]
        if len(precedenti) >= min_storico:
            media_casa_lega = media([x["gol_casa"] for x in precedenti], 1.40)
            media_trasf_lega = media([x["gol_trasferta"] for x in precedenti], 1.10)

            casa = partita["casa"]
            trasf = partita["trasferta"]
            casa_casa = [x for x in precedenti if x["casa"] == casa]
            casa_trasf = [x for x in precedenti if x["trasferta"] == casa]
            trasf_casa = [x for x in precedenti if x["casa"] == trasf]
            trasf_trasf = [x for x in precedenti if x["trasferta"] == trasf]

            gf_c = media_ridotta([x["gol_casa"] for x in casa_casa], media_casa_lega)
            gs_c = media_ridotta([x["gol_trasferta"] for x in casa_casa], media_trasf_lega)
            gf_t = media_ridotta([x["gol_trasferta"] for x in trasf_trasf], media_trasf_lega)
            gs_t = media_ridotta([x["gol_casa"] for x in trasf_trasf], media_casa_lega)

            # Modello Poisson semplice e trasparente: attacco/defesa casa e trasferta.
            lam_c = gf_c * gs_t / max(0.25, media_casa_lega)
            lam_t = gf_t * gs_c / max(0.25, media_trasf_lega)
            ph = poisson_vector(lam_c)
            pa = poisson_vector(lam_t)
            matrice = np.outer(ph, pa)
            matrice /= max(matrice.sum(), 1e-12)
            p1 = float(np.tril(matrice, -1).sum())
            px = float(np.trace(matrice))
            p2 = float(np.triu(matrice, 1).sum())
            p_goal = float(matrice[1:, 1:].sum())
            p_nogoal = 1.0 - p_goal
            esito = "1" if partita["gol_casa"] > partita["gol_trasferta"] else ("X" if partita["gol_casa"] == partita["gol_trasferta"] else "2")
            reale_nogoal = int(partita["gol_casa"] == 0 or partita["gol_trasferta"] == 0)
            probs = {"1": p1, "X": px, "2": p2}
            previsto = max(probs, key=probs.get)
            brier_1x2 = sum((probs[k] - (1.0 if esito == k else 0.0)) ** 2 for k in ("1", "X", "2"))
            p_esito = max(1e-12, probs[esito])
            p_ng_real = p_nogoal if reale_nogoal else p_goal
            righe.append({
                "Data": partita["data"].strftime("%Y-%m-%d"),
                "Casa": casa, "Trasferta": trasf,
                "P(1) %": 100*p1, "P(X) %": 100*px, "P(2) %": 100*p2,
                "Segno previsto": previsto, "Esito reale": esito,
                "X prevista come segno": int(previsto == "X"),
                "P(No Goal) %": 100*p_nogoal, "No Goal favorito": int(p_nogoal > p_goal),
                "No Goal reale": reale_nogoal,
                "Brier 1X2": brier_1x2, "Log Loss 1X2": -math.log(p_esito),
                "Brier Goal/No Goal": (p_nogoal - reale_nogoal) ** 2,
                "Log Loss Goal/No Goal": -math.log(max(1e-12, p_ng_real)),
                "Storico pre-partita": len(precedenti),
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
        matrice_risultati = np.zeros((6, 6))

        for rc in range(6):
            for rt in range(6):
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
        p_c, p_p, p_t = (p_c/tot)*100, (p_p/tot)*100, (p_t/tot)*100
        prob_btts_no = 1.0 - prob_btts_yes

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
        for rc in range(6):
            for rt in range(6):
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
            for rc in range(6):
                for rt in range(6):
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

