import math
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as str_lit

# --- CONFIGURAZIONE PAGINA ---
str_lit.set_page_config(
    page_title="COMBO Master Pro Betting & Deep Analytics Suite — Diagnostica V16",
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

@str_lit.cache_data(ttl=21600, show_spinner=False)
def carica_storico_audit_gratuito():
    """Carica fino a 3 stagioni da football-data.co.uk per l'audit multi-campionato.
    Questa fonte è separata dai dati operativi/API della app e non modifica i pronostici live.
    """
    leghe_csv = {
        "Serie A (Italia)": "I1",
        "Premier League (Inghilterra)": "E0",
        "La Liga (Spagna)": "SP1",
        "Bundesliga (Germania)": "D1",
        "Ligue 1 (Francia)": "F1",
        "Eredivisie (Olanda)": "N1",
    }
    anno_corrente = datetime.now().year
    # La stagione calcistica inizia normalmente ad agosto.
    anno_inizio = anno_corrente if datetime.now().month >= 8 else anno_corrente - 1
    stagioni = [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(anno_inizio, anno_inizio - 3, -1)]
    risultati = []
    diagnostica = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; COMBO-Audit/14.0)"}
    for nome_lega, codice in leghe_csv.items():
        righe_lega = 0
        errori = []
        for stagione in stagioni:
            url = f"https://football-data.co.uk/mmz4281/{stagione}/{codice}.csv"
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code != 200 or not response.content:
                    errori.append(f"{stagione}: HTTP {response.status_code}")
                    continue
                from io import BytesIO
                df = pd.read_csv(BytesIO(response.content), encoding="latin1", on_bad_lines="skip")
                colonne = {str(c).strip(): c for c in df.columns}
                richieste = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
                if not all(c in colonne for c in richieste):
                    errori.append(f"{stagione}: colonne richieste mancanti")
                    continue
                for _, r in df.iterrows():
                    if pd.isna(r[colonne["FTHG"]]) or pd.isna(r[colonne["FTAG"]]) or pd.isna(r[colonne["Date"]]):
                        continue
                    try:
                        data = pd.to_datetime(str(r[colonne["Date"]]).strip(), dayfirst=True, errors="coerce")
                        gc, gt = int(r[colonne["FTHG"]]), int(r[colonne["FTAG"]])
                        casa, trasf = str(r[colonne["HomeTeam"]]).strip(), str(r[colonne["AwayTeam"]]).strip()
                        if pd.isna(data) or not casa or not trasf or gc < 0 or gt < 0:
                            continue
                        risultati.append({
                            "status": "FINISHED",
                            "utcDate": data.strftime("%Y-%m-%dT12:00:00Z"),
                            "Competizione": nome_lega,
                            "homeTeam": {"name": casa},
                            "awayTeam": {"name": trasf},
                            "score": {"fullTime": {"home": gc, "away": gt}},
                            "Stagione audit": stagione,
                        })
                        righe_lega += 1
                    except (TypeError, ValueError, OverflowError):
                        continue
            except Exception as exc:
                errori.append(f"{stagione}: {type(exc).__name__}")
        diagnostica.append({"Competizione": nome_lega, "Partite storiche caricate": righe_lega,
                            "Stagioni richieste": ", ".join(stagioni),
                            "Note caricamento": "; ".join(errori) if errori else "OK"})
    return risultati, pd.DataFrame(diagnostica)


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
            migliore_vittoria = max(p1, p2)
            # Scarto positivo: X supera entrambe le vittorie; negativo: almeno una vittoria la supera.
            scarto_x_vs_vittoria = 100.0 * (px - migliore_vittoria)
            # Risultato esatto più probabile della matrice normalizzata (solo diagnostica).
            score_c, score_t = np.unravel_index(int(np.argmax(matrice)), matrice.shape)
            prob_score_top = float(matrice[score_c, score_t])
            brier_1x2 = sum((probs[k] - (1.0 if esito == k else 0.0)) ** 2 for k in ("1", "X", "2"))
            p_esito = max(1e-12, probs[esito])
            p_ng_real = p_nogoal if reale_nogoal else p_goal

            # V12: baseline empirica walk-forward, stimata SOLO dalle partite precedenti
            # della stessa competizione. Laplace smoothing evita probabilità pari a zero.
            n_pre = len(precedenti)
            conteggi_esiti = {
                "1": sum(x["gol_casa"] > x["gol_trasferta"] for x in precedenti),
                "X": sum(x["gol_casa"] == x["gol_trasferta"] for x in precedenti),
                "2": sum(x["gol_casa"] < x["gol_trasferta"] for x in precedenti),
            }
            p_base = {k: (conteggi_esiti[k] + 1.0) / (n_pre + 3.0) for k in ("1", "X", "2")}
            brier_base_1x2 = sum((p_base[k] - (1.0 if esito == k else 0.0)) ** 2 for k in ("1", "X", "2"))
            logloss_base_1x2 = -math.log(max(1e-12, p_base[esito]))
            ng_pre = sum(x["gol_casa"] == 0 or x["gol_trasferta"] == 0 for x in precedenti)
            p_base_ng = (ng_pre + 1.0) / (n_pre + 2.0)
            p_base_ng_esito = p_base_ng if reale_nogoal else (1.0 - p_base_ng)
            brier_base_ng = (p_base_ng - reale_nogoal) ** 2
            logloss_base_ng = -math.log(max(1e-12, p_base_ng_esito))

            righe.append({
                "Data": partita["data"].strftime("%Y-%m-%d"),
                "Anno": int(partita["data"].year),
                "Competizione": partita["competizione"],
                "Casa": casa, "Trasferta": trasf,
                "P(1) %": 100*p1, "P(X) %": 100*px, "P(2) %": 100*p2,
                "Segno previsto": previsto, "Esito reale": esito,
                "X prevista come segno": int(previsto == "X"),
                "Posizione X": posizione_x,
                "Migliore tra P(1) e P(2) (%)": 100 * migliore_vittoria,
                "Scarto P(X) - migliore vittoria (pp)": scarto_x_vs_vittoria,
                "Risultato esatto più probabile": f"{score_c}-{score_t}",
                "Probabilità risultato esatto top (%)": 100 * prob_score_top,
                "Punteggio esatto top è pareggio": int(score_c == score_t),
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
                "Baseline P(1) %": 100 * p_base["1"],
                "Baseline P(X) %": 100 * p_base["X"],
                "Baseline P(2) %": 100 * p_base["2"],
                "Brier 1X2 baseline": brier_base_1x2,
                "Log Loss 1X2 baseline": logloss_base_1x2,
                "Baseline P(No Goal) %": 100 * p_base_ng,
                "Brier Goal/No Goal baseline": brier_base_ng,
                "Log Loss Goal/No Goal baseline": logloss_base_ng,
                "Storico pre-partita": len(precedenti),
                "Baseline storico pre-partita": len(precedenti),
                "Controllo causalità baseline": int(all(x["data"].date() < partita["data"].date() for x in precedenti)),
                "Lambda casa": lam_c, "Lambda trasferta": lam_t,
            })
        storico.append(partita)

    return pd.DataFrame(righe)




def _prob_clip(p):
    return float(np.clip(float(p), 1e-6, 1.0 - 1e-6))


def _fit_platt_prior(prob, outcome, min_train=100, max_train=300, steps=80):
    """Fit Platt logistico regolarizzato usando esclusivamente osservazioni passate."""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    if len(y) < min_train or len(np.unique(y)) < 2:
        return None
    p, y = p[-max_train:], y[-max_train:]
    x = np.log(np.clip(p, 1e-5, 1-1e-5) / np.clip(1-p, 1e-5, 1))
    # Newton-Raphson su intercetta e coefficiente, con lieve ridge sul coefficiente.
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(steps):
        z = np.clip(X @ beta, -25, 25)
        q = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(q * (1-q), 1e-6, None)
        grad = X.T @ (q-y) + np.array([0.01 * beta[0], 0.02 * beta[1]])
        hess = X.T @ (X * w[:, None]) + np.array([[0.01, 0.0], [0.0, 0.02]])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        beta -= step
        beta[1] = float(np.clip(beta[1], -5.0, 5.0))
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return beta


def _calcola_metriche_binaria(prob, y):
    p = np.clip(np.asarray(prob, dtype=float), 1e-12, 1-1e-12)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p-y)**2)), float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


@str_lit.cache_data(ttl=21600, show_spinner=False)
def crea_test_v15_walk_forward(df_audit, min_train=100, max_train=300, max_gol=10):
    """Test diagnostici out-of-sample: correzione lambda e calibrazione X/No Goal.
    Ogni parametro per la partita target è stimato solo da righe precedenti della stessa competizione.
    Nessun valore prodotto da questa funzione viene usato nelle previsioni operative.
    """
    if df_audit.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d = df_audit.copy()
    d['_data_dt'] = pd.to_datetime(d['Data'], errors='coerce', utc=True)
    d = d.sort_values(['_data_dt', 'Competizione']).reset_index(drop=True)
    d['P(X) raw'] = pd.to_numeric(d['P(X) %'], errors='coerce') / 100.0
    d['P(NG) raw'] = pd.to_numeric(d['P(No Goal) %'], errors='coerce') / 100.0
    d['X y'] = (d['Esito reale'] == 'X').astype(int)
    d['NG y'] = pd.to_numeric(d['No Goal reale'], errors='coerce').fillna(0).astype(int)
    d['y1'] = (d['Esito reale'] == '1').astype(int)
    d['y2'] = (d['Esito reale'] == '2').astype(int)
    for col in ['Lambda casa', 'Lambda trasferta', 'Gol casa reali', 'Gol trasferta reali', 'P(1) %', 'P(2) %']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d['P(X) calibrata WF'] = np.nan
    d['P(NG) calibrata WF'] = np.nan
    d['Lambda casa corretta WF'] = np.nan
    d['Lambda trasferta corretta WF'] = np.nan
    d['P(1) lambda corretta WF'] = np.nan
    d['P(X) lambda corretta WF'] = np.nan
    d['P(2) lambda corretta WF'] = np.nan
    d['P(NG) lambda corretta WF'] = np.nan
    d['Training usato V15'] = 0
    d['Fattore casa WF'] = np.nan
    d['Fattore trasferta WF'] = np.nan

    # Elaborazione per competizione; target-day escluso dal training.
    for comp, idxs in d.groupby('Competizione', dropna=False).groups.items():
        idxs = list(idxs)
        prior_idxs = []
        for idx in idxs:
            target_date = d.at[idx, '_data_dt']
            # Elimina tutte le partite dello stesso giorno dal training.
            prior_idxs = [j for j in prior_idxs if d.at[j, '_data_dt'].date() < target_date.date()]
            train = d.loc[prior_idxs].tail(max_train)
            if len(train) >= min_train:
                # Calibrazione binaria X e No Goal, fit solo su precedenti della stessa competizione.
                bx = _fit_platt_prior(train['P(X) raw'], train['X y'], min_train, max_train)
                bng = _fit_platt_prior(train['P(NG) raw'], train['NG y'], min_train, max_train)
                if bx is not None:
                    p = _prob_clip(d.at[idx, 'P(X) raw'])
                    x = np.log(p/(1-p))
                    d.at[idx, 'P(X) calibrata WF'] = 1/(1+np.exp(-np.clip(bx[0]+bx[1]*x, -25, 25)))
                if bng is not None:
                    p = _prob_clip(d.at[idx, 'P(NG) raw'])
                    x = np.log(p/(1-p))
                    d.at[idx, 'P(NG) calibrata WF'] = 1/(1+np.exp(-np.clip(bng[0]+bng[1]*x, -25, 25)))

                # Fattori di correzione lambda, stimati dal rapporto gol reali / lambda
                # nelle sole previsioni già valutate. Shrinkage verso 1 e limiti anti-estremi.
                lc = train['Lambda casa'].to_numpy(dtype=float)
                la = train['Lambda trasferta'].to_numpy(dtype=float)
                gh = train['Gol casa reali'].to_numpy(dtype=float)
                ga = train['Gol trasferta reali'].to_numpy(dtype=float)
                valid_h = np.isfinite(lc) & np.isfinite(gh) & (lc > 0)
                valid_a = np.isfinite(la) & np.isfinite(ga) & (la > 0)
                fh = float(np.clip((np.mean(gh[valid_h]) / np.mean(lc[valid_h])) if valid_h.any() else 1.0, 0.80, 1.25))
                fa = float(np.clip((np.mean(ga[valid_a]) / np.mean(la[valid_a])) if valid_a.any() else 1.0, 0.80, 1.25))
                # Shrinkage: con massimo 300 partite evita correzioni troppo aggressive.
                shrink = len(train) / (len(train) + 150.0)
                fh = 1.0 + (fh - 1.0) * shrink
                fa = 1.0 + (fa - 1.0) * shrink
                lc_new = max(0.01, float(d.at[idx, 'Lambda casa']) * fh)
                la_new = max(0.01, float(d.at[idx, 'Lambda trasferta']) * fa)
                d.at[idx, 'Lambda casa corretta WF'] = lc_new
                d.at[idx, 'Lambda trasferta corretta WF'] = la_new
                d.at[idx, 'Fattore casa WF'] = fh
                d.at[idx, 'Fattore trasferta WF'] = fa
                # Matrice Poisson 0..max_gol con la stessa normalizzazione dell'audit originale.
                vh = np.array([math.exp(-lc_new) * lc_new**k / math.factorial(k) for k in range(max_gol+1)])
                va = np.array([math.exp(-la_new) * la_new**k / math.factorial(k) for k in range(max_gol+1)])
                mat = np.outer(vh, va)
                mat /= max(float(mat.sum()), 1e-15)
                d.at[idx, 'P(1) lambda corretta WF'] = float(np.tril(mat, -1).sum())
                d.at[idx, 'P(X) lambda corretta WF'] = float(np.trace(mat))
                d.at[idx, 'P(2) lambda corretta WF'] = float(np.triu(mat, 1).sum())
                d.at[idx, 'P(NG) lambda corretta WF'] = float(1.0 - mat[1:, 1:].sum())
                d.at[idx, 'Training usato V15'] = len(train)
            # Aggiungi la riga target allo storico solo dopo avere calcolato la previsione.
            prior_idxs.append(idx)

    # Tabella aggregata delle metriche: confronta sulle stesse righe per ciascuna coppia.
    righe_metriche = []
    yx = d['X y'].to_numpy(dtype=float)
    yng = d['NG y'].to_numpy(dtype=float)
    px_raw = d['P(X) raw'].to_numpy(dtype=float)
    png_raw = d['P(NG) raw'].to_numpy(dtype=float)
    for label, col, target, base in [
        ('X — Brier', 'P(X) calibrata WF', 'X y', 'P(X) raw'),
        ('No Goal — Brier', 'P(NG) calibrata WF', 'NG y', 'P(NG) raw'),
    ]:
        mask = d[col].notna() & d[base].notna()
        if int(mask.sum()) >= 1:
            y = d.loc[mask, target].to_numpy(dtype=float)
            raw = d.loc[mask, base].to_numpy(dtype=float)
            cal = d.loc[mask, col].to_numpy(dtype=float)
            righe_metriche.append({'Test': label, 'N test out-of-sample': int(mask.sum()), 'Metrica raw': float(np.mean((raw-y)**2)), 'Metrica WF': float(np.mean((cal-y)**2)), 'Delta WF - raw': float(np.mean((cal-y)**2)-np.mean((raw-y)**2)), 'LogLoss raw': _calcola_metriche_binaria(raw,y)[1], 'LogLoss WF': _calcola_metriche_binaria(cal,y)[1]})
    # Lambda correction: goals MAE and probability scores on identical rows.
    mask_l = d['Lambda casa corretta WF'].notna() & d['Lambda trasferta corretta WF'].notna()
    if int(mask_l.sum()) >= 1:
        z = d.loc[mask_l]
        raw_h = z['Lambda casa'].to_numpy(dtype=float); raw_a = z['Lambda trasferta'].to_numpy(dtype=float)
        adj_h = z['Lambda casa corretta WF'].to_numpy(dtype=float); adj_a = z['Lambda trasferta corretta WF'].to_numpy(dtype=float)
        gh = z['Gol casa reali'].to_numpy(dtype=float); ga = z['Gol trasferta reali'].to_numpy(dtype=float)
        y1 = z['y1'].to_numpy(dtype=float); yxv = z['X y'].to_numpy(dtype=float); y2 = z['y2'].to_numpy(dtype=float)
        raw_probs = z[['P(1) %','P(X) %','P(2) %']].to_numpy(dtype=float)/100
        adj_probs = z[['P(1) lambda corretta WF','P(X) lambda corretta WF','P(2) lambda corretta WF']].to_numpy(dtype=float)
        raw_brier_1x2 = float(np.mean(np.sum((raw_probs - np.column_stack([y1,yxv,y2]))**2, axis=1)))
        adj_brier_1x2 = float(np.mean(np.sum((adj_probs - np.column_stack([y1,yxv,y2]))**2, axis=1)))
        yng_l = z['NG y'].to_numpy(dtype=float)
        raw_ng = z['P(NG) raw'].to_numpy(dtype=float)
        adj_ng = z['P(NG) lambda corretta WF'].to_numpy(dtype=float)
        righe_metriche.extend([
            {'Test':'Lambda — MAE gol casa', 'N test out-of-sample':int(mask_l.sum()), 'Metrica raw':float(np.mean(np.abs(raw_h-gh))), 'Metrica WF':float(np.mean(np.abs(adj_h-gh))), 'Delta WF - raw':float(np.mean(np.abs(adj_h-gh))-np.mean(np.abs(raw_h-gh))), 'LogLoss raw':np.nan, 'LogLoss WF':np.nan},
            {'Test':'Lambda — MAE gol trasferta', 'N test out-of-sample':int(mask_l.sum()), 'Metrica raw':float(np.mean(np.abs(raw_a-ga))), 'Metrica WF':float(np.mean(np.abs(adj_a-ga))), 'Delta WF - raw':float(np.mean(np.abs(adj_a-ga))-np.mean(np.abs(raw_a-ga))), 'LogLoss raw':np.nan, 'LogLoss WF':np.nan},
            {'Test':'Lambda — Brier 1X2', 'N test out-of-sample':int(mask_l.sum()), 'Metrica raw':raw_brier_1x2, 'Metrica WF':adj_brier_1x2, 'Delta WF - raw':adj_brier_1x2-raw_brier_1x2, 'LogLoss raw':np.nan, 'LogLoss WF':np.nan},
            {'Test':'Lambda — Brier No Goal', 'N test out-of-sample':int(mask_l.sum()), 'Metrica raw':float(np.mean((raw_ng-yng_l)**2)), 'Metrica WF':float(np.mean((adj_ng-yng_l)**2)), 'Delta WF - raw':float(np.mean((adj_ng-yng_l)**2)-np.mean((raw_ng-yng_l)**2)), 'LogLoss raw':_calcola_metriche_binaria(raw_ng,yng_l)[1], 'LogLoss WF':_calcola_metriche_binaria(adj_ng,yng_l)[1]},
        ])
    # Riepilogo per competizione: raw vs correzioni sulle stesse righe out-of-sample.
    righe_comp = []
    for comp, z in d.groupby('Competizione', dropna=False):
        row = {
            'Competizione': comp,
            'Partite valutate': int(len(z)),
            'Prima data': z['_data_dt'].min(),
            'Ultima data': z['_data_dt'].max(),
            'Training minimo usato': int(z['Training usato V15'].min()),
            'Training massimo usato': int(z['Training usato V15'].max()),
        }
        for raw_col, cal_col, target_col, label_raw, label_cal in [
            ('P(X) raw', 'P(X) calibrata WF', 'X y', 'Brier X raw', 'Brier X calibrato WF'),
            ('P(NG) raw', 'P(NG) calibrata WF', 'NG y', 'Brier No Goal raw', 'Brier No Goal calibrato WF'),
        ]:
            zcal = z[z[cal_col].notna()]
            row[label_raw] = float(np.mean((zcal[raw_col].to_numpy(dtype=float) - zcal[target_col].to_numpy(dtype=float))**2)) if not zcal.empty else np.nan
            row[label_cal] = float(np.mean((zcal[cal_col].to_numpy(dtype=float) - zcal[target_col].to_numpy(dtype=float))**2)) if not zcal.empty else np.nan
        zz = z[z['Lambda casa corretta WF'].notna() & z['Lambda trasferta corretta WF'].notna()]
        if not zz.empty:
            targets = np.column_stack([(zz['Esito reale'] == '1').astype(int), (zz['Esito reale'] == 'X').astype(int), (zz['Esito reale'] == '2').astype(int)])
            probs_raw = zz[['P(1) %','P(X) %','P(2) %']].to_numpy(dtype=float)/100.0
            probs_adj = zz[['P(1) lambda corretta WF','P(X) lambda corretta WF','P(2) lambda corretta WF']].to_numpy(dtype=float)
            row['Brier 1X2 raw'] = float(np.mean(np.sum((probs_raw-targets)**2, axis=1)))
            row['Brier 1X2 lambda WF'] = float(np.mean(np.sum((probs_adj-targets)**2, axis=1)))
            row['MAE lambda casa raw'] = float(np.mean(np.abs(zz['Lambda casa']-zz['Gol casa reali'])))
            row['MAE lambda casa WF'] = float(np.mean(np.abs(zz['Lambda casa corretta WF']-zz['Gol casa reali'])))
            row['MAE lambda trasferta raw'] = float(np.mean(np.abs(zz['Lambda trasferta']-zz['Gol trasferta reali'])))
            row['MAE lambda trasferta WF'] = float(np.mean(np.abs(zz['Lambda trasferta corretta WF']-zz['Gol trasferta reali'])))
        else:
            for col in ['Brier 1X2 raw','Brier 1X2 lambda WF','MAE lambda casa raw','MAE lambda casa WF','MAE lambda trasferta raw','MAE lambda trasferta WF']:
                row[col] = np.nan
        righe_comp.append(row)
    return d, pd.DataFrame(righe_metriche), pd.DataFrame(righe_comp)


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
    str_lit.markdown("#### 🌍 V15 — Audit storico multi-campionato con dati gratuiti")
    str_lit.caption(
        "L'audit diagnostico carica separatamente fino a tre stagioni per sei campionati da football-data.co.uk. "
        "Non usa la sola competizione selezionata nel menu e non modifica i dati o le previsioni operative. "
        "Per ogni partita, le stime usano soltanto risultati precedenti della stessa competizione."
    )
    with str_lit.spinner("Caricamento dello storico gratuito per tutti i campionati..."):
        matches_audit, df_caricamento = carica_storico_audit_gratuito()
    str_lit.markdown("##### Stato caricamento storico")
    str_lit.dataframe(df_caricamento, use_container_width=True, hide_index=True)
    if matches_audit:
        df_audit = crea_audit_walk_forward(matches_audit)
    else:
        # Fallback trasparente alla fonte già usata dalla V13, senza cambiare il modello.
        df_audit = crea_audit_walk_forward(matches_raw)
        str_lit.warning("Fonte storica gratuita non raggiungibile: mostrato il campione API della V13, se disponibile.")
    if df_audit.empty:
        str_lit.warning(
            "Non ci sono ancora abbastanza partite concluse per un audit walk-forward. "
            "Servono almeno 30 risultati precedenti nella competizione prima di valutare una partita."
        )
    else:
        # V15: pacchetto diagnostico aggiuntivo; nessuna modifica alle previsioni operative.
        str_lit.markdown("#### 🧪 V17 — Calibrazione No Goal per fasce (solo diagnostica)")
        str_lit.caption(
            "In un'unica esecuzione verifica: riconciliazione caricamento/valutazione, correzione walk-forward delle lambda casa/trasferta, "
            "calibrazione separata di X e No Goal. Ogni correzione usa solo partite precedenti della stessa competizione; le previsioni operative restano invariate."
        )
        with str_lit.spinner("Esecuzione test walk-forward V17; può richiedere qualche secondo..."):
            df_v15_test, df_v15_metriche, df_v15_comp = crea_test_v15_walk_forward(df_audit)
        # Riconciliazione: partite valutate per competizione contro partite caricate dalla fonte gratuita.
        if matches_audit:
            caricate_comp = pd.DataFrame(matches_audit)
            caricate_comp = caricate_comp.groupby("Competizione").size().reset_index(name="Partite caricate")
            valutate_comp = df_audit.groupby("Competizione").size().reset_index(name="Partite valutate")
            df_riconciliazione = caricate_comp.merge(valutate_comp, on="Competizione", how="outer").fillna(0)
            df_riconciliazione["Partite non valutate / differenza"] = df_riconciliazione["Partite caricate"] - df_riconciliazione["Partite valutate"]
            str_lit.markdown("##### 1. Riconciliazione partite caricate e valutate")
            str_lit.dataframe(df_riconciliazione.round(0), use_container_width=True, hide_index=True)
            str_lit.caption("La differenza include soprattutto lo storico minimo richiesto prima della prima valutazione; eventuali scarti diversi vanno investigati.")
        if not df_v15_metriche.empty:
            str_lit.markdown("##### 2. Confronto raw vs correzioni walk-forward")
            str_lit.dataframe(df_v15_metriche.round(5), use_container_width=True, hide_index=True)
            str_lit.caption("Delta WF - raw negativo = versione testata migliore. Per le righe lambda, la metrica è MAE dei gol attesi; per le probabilità è Brier Score. Log Loss è mostrato dove applicabile.")
        else:
            str_lit.info("Campione insufficiente per i test calibrati: servono almeno 100 osservazioni precedenti della stessa competizione.")
        if not df_v15_comp.empty:
            str_lit.markdown("##### 3. Risultati diagnostici per competizione")
            str_lit.dataframe(df_v15_comp.round(3), use_container_width=True, hide_index=True)

        # V16: verifica della stabilità della calibrazione No Goal per periodo e competizione.
        # Confronto paired: raw e WF sono calcolati sulle stesse partite in cui WF è disponibile.
        if not df_v15_test.empty:
            colonne_ng_v16 = ["P(NG) raw", "P(NG) calibrata WF", "NG y", "Competizione", "Data"]
            if all(c in df_v15_test.columns for c in colonne_ng_v16):
                df_ng_v16 = df_v15_test[colonne_ng_v16].copy()
                for col in ["P(NG) raw", "P(NG) calibrata WF", "NG y"]:
                    df_ng_v16[col] = pd.to_numeric(df_ng_v16[col], errors="coerce")
                df_ng_v16["Data"] = pd.to_datetime(df_ng_v16["Data"], errors="coerce", utc=True)
                df_ng_v16 = df_ng_v16.dropna(subset=["P(NG) raw", "P(NG) calibrata WF", "NG y", "Competizione", "Data"])
                df_ng_v16["Anno test"] = df_ng_v16["Data"].dt.year

                def _riepilogo_ng_v16(gruppi, nome_gruppo):
                    risultati = []
                    for chiave, gruppo in df_ng_v16.groupby(gruppi, dropna=False):
                        if not isinstance(chiave, tuple):
                            chiave = (chiave,)
                        y = gruppo["NG y"].to_numpy(dtype=float)
                        raw = gruppo["P(NG) raw"].to_numpy(dtype=float)
                        wf = gruppo["P(NG) calibrata WF"].to_numpy(dtype=float)
                        brier_raw, ll_raw = _calcola_metriche_binaria(raw, y)
                        brier_wf, ll_wf = _calcola_metriche_binaria(wf, y)
                        r = {nome_gruppo: " / ".join(str(v) for v in chiave), "Partite test WF": int(len(gruppo)),
                             "Brier NG raw": brier_raw, "Brier NG WF": brier_wf,
                             "Delta Brier WF - raw": brier_wf - brier_raw,
                             "LogLoss NG raw": ll_raw, "LogLoss NG WF": ll_wf,
                             "Delta LogLoss WF - raw": ll_wf - ll_raw}
                        if len(gruppo) < 100:
                            r["Affidabilità campione"] = "Basso n (<100)"
                        else:
                            r["Affidabilità campione"] = "Almeno 100"
                        r["Entrambe migliorano"] = "Sì" if (brier_wf < brier_raw and ll_wf < ll_raw) else "No"
                        risultati.append(r)
                    return pd.DataFrame(risultati)

                str_lit.markdown("##### 4. V16 — Stabilità No Goal per anno")
                str_lit.caption("Confronto RAW/WF sulle stesse partite out-of-sample per cui esiste una calibrazione WF. Delta negativo è migliore; gli anni con meno di 100 partite sono indicativi.")
                df_ng_anno_v16 = _riepilogo_ng_v16(["Anno test"], "Anno")
                if not df_ng_anno_v16.empty:
                    str_lit.dataframe(df_ng_anno_v16.round(5), use_container_width=True, hide_index=True)
                else:
                    str_lit.info("Non ci sono abbastanza previsioni calibrate WF per il riepilogo annuale No Goal.")

                str_lit.markdown("##### 5. V16 — Stabilità No Goal per competizione")
                df_ng_comp_v16 = _riepilogo_ng_v16(["Competizione"], "Competizione")
                if not df_ng_comp_v16.empty:
                    str_lit.dataframe(df_ng_comp_v16.round(5), use_container_width=True, hide_index=True)
                else:
                    str_lit.info("Non ci sono abbastanza previsioni calibrate WF per il riepilogo per competizione.")


                # V17: calibrazione per fasce RAW e WF, più confronto delle metriche per fascia.
                # Solo diagnostica: non modifica né sostituisce le previsioni operative.
                str_lit.markdown("##### 6. V17 — Calibrazione RAW per fasce di probabilità")
                str_lit.caption("Le fasce sono definite dalla probabilità RAW pre-correzione. La frequenza reale è la percentuale di partite con No Goal. Fasce con pochi casi vanno interpretate con cautela.")
                fasce_v17 = [0.0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.000001]
                etichette_v17 = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
                df_ng_v17 = df_ng_v16.copy()
                for c in ["P(NG) raw", "P(NG) calibrata WF", "NG y"]:
                    df_ng_v17[c] = pd.to_numeric(df_ng_v17[c], errors="coerce")
                df_ng_v17 = df_ng_v17.dropna(subset=["P(NG) raw", "P(NG) calibrata WF", "NG y"])
                df_ng_v17["Fascia RAW"] = pd.cut(df_ng_v17["P(NG) raw"].clip(0, 1), bins=fasce_v17, labels=etichette_v17, right=True, include_lowest=True)
                df_ng_v17["Fascia WF"] = pd.cut(df_ng_v17["P(NG) calibrata WF"].clip(0, 1), bins=fasce_v17, labels=etichette_v17, right=True, include_lowest=True)

                def _calibrazione_v17(df, col_prob, col_fascia):
                    rows = []
                    for fascia in etichette_v17:
                        g = df.loc[df[col_fascia] == fascia]
                        if g.empty:
                            continue
                        rows.append({
                            "Fascia probabilità": fascia,
                            "Partite": int(len(g)),
                            "Probabilità media (%)": float(g[col_prob].mean() * 100),
                            "No Goal reali (%)": float(g["NG y"].mean() * 100),
                            "Scarto previsione - reale (pp)": float((g[col_prob].mean() - g["NG y"].mean()) * 100),
                            "Affidabilità campione": "Basso n (<100)" if len(g) < 100 else "Almeno 100"
                        })
                    return pd.DataFrame(rows)

                df_v17_raw_cal = _calibrazione_v17(df_ng_v17, "P(NG) raw", "Fascia RAW")
                if not df_v17_raw_cal.empty:
                    str_lit.dataframe(df_v17_raw_cal.round(3), use_container_width=True, hide_index=True)
                    str_lit.download_button("📥 Scarica calibrazione RAW per fasce V17", df_v17_raw_cal.to_csv(index=False).encode("utf-8-sig"), "v17_nogoal_calibrazione_raw_fasce.csv", "text/csv")
                else:
                    str_lit.info("Nessun dato valido per la calibrazione RAW per fasce.")

                str_lit.markdown("##### 7. V17 — Calibrazione walk-forward per fasce di probabilità")
                str_lit.caption("Qui le fasce sono definite dalla probabilità WF corretta; il confronto mostra se la probabilità corretta corrisponde meglio alla frequenza reale osservata.")
                df_v17_wf_cal = _calibrazione_v17(df_ng_v17, "P(NG) calibrata WF", "Fascia WF")
                if not df_v17_wf_cal.empty:
                    str_lit.dataframe(df_v17_wf_cal.round(3), use_container_width=True, hide_index=True)
                    str_lit.download_button("📥 Scarica calibrazione WF per fasce V17", df_v17_wf_cal.to_csv(index=False).encode("utf-8-sig"), "v17_nogoal_calibrazione_wf_fasce.csv", "text/csv")
                else:
                    str_lit.info("Nessun dato valido per la calibrazione WF per fasce.")

                str_lit.markdown("##### 8. V17 — Confronto RAW vs WF per fascia RAW")
                str_lit.caption("Per un confronto appaiato e leggibile, le partite sono raggruppate in base alla probabilità RAW. Delta WF − RAW negativo significa miglioramento della metrica.")
                righe_confronto_v17 = []
                for fascia in etichette_v17:
                    g = df_ng_v17.loc[df_ng_v17["Fascia RAW"] == fascia]
                    if g.empty:
                        continue
                    y = g["NG y"].to_numpy(dtype=float)
                    raw = g["P(NG) raw"].to_numpy(dtype=float)
                    wf = g["P(NG) calibrata WF"].to_numpy(dtype=float)
                    brier_raw, ll_raw = _calcola_metriche_binaria(raw, y)
                    brier_wf, ll_wf = _calcola_metriche_binaria(wf, y)
                    righe_confronto_v17.append({
                        "Fascia probabilità RAW": fascia,
                        "Partite": int(len(g)),
                        "Brier RAW": brier_raw,
                        "Brier WF": brier_wf,
                        "Delta Brier WF - RAW": brier_wf - brier_raw,
                        "LogLoss RAW": ll_raw,
                        "LogLoss WF": ll_wf,
                        "Delta LogLoss WF - RAW": ll_wf - ll_raw,
                        "Entrambe migliorano": "Sì" if brier_wf < brier_raw and ll_wf < ll_raw else "No",
                        "Affidabilità campione": "Basso n (<100)" if len(g) < 100 else "Almeno 100"
                    })
                df_v17_confronto = pd.DataFrame(righe_confronto_v17)
                if not df_v17_confronto.empty:
                    str_lit.dataframe(df_v17_confronto.round(5), use_container_width=True, hide_index=True)
                    str_lit.download_button("📥 Scarica confronto RAW-WF per fasce V17", df_v17_confronto.to_csv(index=False).encode("utf-8-sig"), "v17_nogoal_confronto_raw_wf_fasce.csv", "text/csv")
                else:
                    str_lit.info("Nessun dato valido per il confronto RAW vs WF per fascia.")
                str_lit.info("V17 è esclusivamente diagnostica: nessuna correzione viene applicata alle previsioni operative.")
            else:
                str_lit.warning("Diagnostica V16 No Goal non disponibile: mancano colonne necessarie nel risultato walk-forward.")

        if not df_v15_test.empty:
            with str_lit.expander("Dettaglio delle correzioni walk-forward (solo test, non operative)", expanded=False):
                cols_v15 = [c for c in ["Data", "Competizione", "Casa", "Trasferta", "Training usato V15", "Fattore casa WF", "Fattore trasferta WF", "P(X) %", "P(X) calibrata WF", "P(No Goal) %", "P(NG) calibrata WF", "P(NG) lambda corretta WF"] if c in df_v15_test.columns]
                df_v15_display = df_v15_test[cols_v15].copy()
                for col_pct in ["P(X) calibrata WF", "P(NG) calibrata WF", "P(1) lambda corretta WF", "P(X) lambda corretta WF", "P(2) lambda corretta WF", "P(NG) lambda corretta WF"]:
                    if col_pct in df_v15_display.columns:
                        df_v15_display[col_pct] = pd.to_numeric(df_v15_display[col_pct], errors="coerce") * 100.0
                str_lit.dataframe(df_v15_display.round(3), use_container_width=True, hide_index=True)
                df_v15_download = df_v15_test.copy()
                for col_pct in ["P(X) calibrata WF", "P(NG) calibrata WF", "P(1) lambda corretta WF", "P(X) lambda corretta WF", "P(2) lambda corretta WF", "P(NG) lambda corretta WF"]:
                    if col_pct in df_v15_download.columns:
                        df_v15_download[col_pct] = pd.to_numeric(df_v15_download[col_pct], errors="coerce") * 100.0
                str_lit.download_button("📥 Scarica confronto diagnostico V16 CSV", df_v15_download.to_csv(index=False).encode("utf-8-sig"), "audit_diagnostica_v16_walk_forward.csv", "text/csv")

        # V12: confronto con baseline empirica walk-forward; il modello operativo resta invariato.
        str_lit.markdown("#### 🔎 V15 — Audit esteso, campione comune e stabilità temporale")
        str_lit.caption(
            "Mantiene i controlli precedenti e amplia il campione storico a più competizioni. Verifica il campione comune Poisson/baseline, la causalità "
            "della baseline e la stabilità per competizione e anno. La baseline usa solo risultati "
            "precedenti della stessa competizione con smoothing Laplace. Nessuna modifica alle previsioni operative."
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

        # V10: quanto la probabilità X si avvicina o supera la migliore tra 1 e 2.
        str_lit.markdown("#### 🧩 Quanto è competitiva la X rispetto a 1 e 2?")
        str_lit.caption(
            "Scarto = P(X) meno la maggiore tra P(1) e P(2). Valore negativo: una vittoria è favorita; "
            "zero o positivo: la X è almeno pari alla migliore vittoria. Nessuna probabilità viene cambiata."
        )
        df_x_gap = df_audit.copy()
        df_x_gap["Esito X reale (0/1)"] = (df_x_gap["Esito reale"] == "X").astype(int)
        df_x_gap["Fascia scarto X-vittoria"] = pd.cut(
            df_x_gap["Scarto P(X) - migliore vittoria (pp)"],
            bins=[-float("inf"), -15, -10, -5, 0, float("inf")],
            labels=["X indietro >15 pp", "X indietro 10–15 pp", "X indietro 5–10 pp", "X indietro 0–5 pp", "X pari o superiore"],
            right=False,
        )
        riepilogo_x_gap = df_x_gap.groupby("Fascia scarto X-vittoria", observed=False).agg(
            Partite=("Esito X reale (0/1)", "size"),
            **{
                "Scarto medio X-vittoria (pp)": ("Scarto P(X) - migliore vittoria (pp)", "mean"),
                "P(X) media (%)": ("P(X) %", "mean"),
                "Pareggi reali (%)": ("Esito X reale (0/1)", lambda x: 100 * x.mean()),
                "X al 1° posto (%)": ("X prevista come segno", lambda x: 100 * x.mean()),
            },
        ).reset_index()
        str_lit.dataframe(riepilogo_x_gap.round(2), use_container_width=True, hide_index=True)

        str_lit.markdown("#### 🎯 Risultato esatto più probabile — controllo descrittivo")
        str_lit.caption(
            "Il punteggio esatto top è il singolo risultato con probabilità maggiore nella matrice 0–10. "
            "Non è una previsione aggiuntiva né una modifica al segno 1X2."
        )
        score_top_rate = 100 * df_audit["Punteggio esatto top è pareggio"].mean()
        score_top_draw_real = df_audit.loc[df_audit["Punteggio esatto top è pareggio"].eq(1), "Esito reale"].eq("X").mean() * 100 if df_audit["Punteggio esatto top è pareggio"].eq(1).any() else 0.0
        z1, z2 = str_lit.columns(2)
        z1.metric("Punteggio esatto top in parità", f"{score_top_rate:.1f}%", "quota partite con score top X-X")
        z2.metric("Pareggi reali quando score top è X-X", f"{score_top_draw_real:.1f}%", "descrittivo; campione potenzialmente piccolo")
        cols_score = ["Data", "Casa", "Trasferta", "P(1) %", "P(X) %", "P(2) %", "Scarto P(X) - migliore vittoria (pp)", "Segno previsto", "Risultato esatto più probabile", "Probabilità risultato esatto top (%)", "Esito reale"]
        with str_lit.expander("Dettaglio diagnostico X e punteggio esatto"):
            str_lit.dataframe(df_audit[cols_score].sort_values("Scarto P(X) - migliore vittoria (pp)", ascending=False).round(3), use_container_width=True, hide_index=True)

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

        # V11: separa l'errore di stima dei gol attesi dalla sola probabilità No Goal.
        # Diagnostica retrospettiva; non modifica lambda, matrice o previsioni operative.
        str_lit.markdown("#### 🧪 V11 — Gol attesi vs gol reali")
        str_lit.caption(
            "Confronta le lambda pre-partita con i gol effettivamente segnati, sia per la squadra di casa "
            "sia per quella in trasferta. La frequenza reale di zero gol viene confrontata con exp(-lambda). "
            "È un controllo diagnostico, non una correzione del modello."
        )
        df_lam = df_audit.copy()
        for col in ["Lambda casa", "Lambda trasferta", "Gol casa reali", "Gol trasferta reali"]:
            df_lam[col] = pd.to_numeric(df_lam[col], errors="coerce")
        df_lam = df_lam.dropna(subset=["Lambda casa", "Lambda trasferta", "Gol casa reali", "Gol trasferta reali"])
        if not df_lam.empty:
            # Tabella riepilogativa complessiva: atteso vs osservato.
            riepilogo_lam = pd.DataFrame([
                {
                    "Lato": "Casa",
                    "Partite": len(df_lam),
                    "Lambda media": df_lam["Lambda casa"].mean(),
                    "Gol reali medi": df_lam["Gol casa reali"].mean(),
                    "Errore medio (reali - attesi)": (df_lam["Gol casa reali"] - df_lam["Lambda casa"]).mean(),
                    "Zero gol attesi (%)": 100 * np.exp(-df_lam["Lambda casa"]).mean(),
                    "Zero gol reali (%)": 100 * (df_lam["Gol casa reali"] == 0).mean(),
                },
                {
                    "Lato": "Trasferta",
                    "Partite": len(df_lam),
                    "Lambda media": df_lam["Lambda trasferta"].mean(),
                    "Gol reali medi": df_lam["Gol trasferta reali"].mean(),
                    "Errore medio (reali - attesi)": (df_lam["Gol trasferta reali"] - df_lam["Lambda trasferta"]).mean(),
                    "Zero gol attesi (%)": 100 * np.exp(-df_lam["Lambda trasferta"]).mean(),
                    "Zero gol reali (%)": 100 * (df_lam["Gol trasferta reali"] == 0).mean(),
                },
            ])
            str_lit.dataframe(riepilogo_lam.round(3), use_container_width=True, hide_index=True)

            # Raggruppamento per lambda, così si vede se gli errori si concentrano sui valori bassi.
            for lato, col_lam, col_gol in [
                ("Casa", "Lambda casa", "Gol casa reali"),
                ("Trasferta", "Lambda trasferta", "Gol trasferta reali"),
            ]:
                d = df_lam[[col_lam, col_gol]].copy()
                d["Fascia lambda"] = pd.cut(
                    d[col_lam], bins=[-float("inf"), 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, float("inf")],
                    labels=["≤0.75", "0.75–1.00", "1.00–1.25", "1.25–1.50", "1.50–1.75", "1.75–2.00", ">2.00"],
                    right=True, include_lowest=True,
                )
                d["Zero gol reale (0/1)"] = (d[col_gol] == 0).astype(int)
                d = d.dropna(subset=["Fascia lambda"])
                riepilogo_lato = d.groupby("Fascia lambda", observed=False).agg(
                    Partite=(col_gol, "size"),
                    **{
                        "Lambda media": (col_lam, "mean"),
                        "Gol reali medi": (col_gol, "mean"),
                        "Zero gol attesi (%)": (col_lam, lambda x: 100 * np.exp(-x).mean()),
                        "Zero gol reali (%)": ("Zero gol reale (0/1)", lambda x: 100 * x.mean()),
                    },
                ).reset_index()
                riepilogo_lato["Errore medio (reali - attesi)"] = riepilogo_lato["Gol reali medi"] - riepilogo_lato["Lambda media"]
                with str_lit.expander(f"Dettaglio lambda — {lato}", expanded=False):
                    str_lit.dataframe(riepilogo_lato.round(3), use_container_width=True, hide_index=True)

        # Calibrazione X per competizione: utile per distinguere un problema globale da uno localizzato.
        str_lit.markdown("#### 🏟️ V14 — Pareggio X per competizione")
        if "Competizione" in df_audit.columns:
            # Il nome competizione viene riportato nella riga walk-forward.
            riepilogo_comp = df_audit.copy()
            riepilogo_comp["X reale (0/1)"] = (riepilogo_comp["Esito reale"] == "X").astype(int)
            riepilogo_comp = riepilogo_comp.groupby("Competizione", dropna=False).agg(
                Partite=("Esito reale", "size"),
                **{
                    "P(X) media (%)": ("P(X) %", "mean"),
                    "Pareggi reali (%)": ("X reale (0/1)", lambda x: 100 * x.mean()),
                    "X prevista 1° (%)": ("X prevista come segno", lambda x: 100 * x.mean()),
                },
            ).reset_index()
            riepilogo_comp["Scarto P(X) - pareggi reali (pp)"] = riepilogo_comp["P(X) media (%)"] - riepilogo_comp["Pareggi reali (%)"]
            str_lit.dataframe(riepilogo_comp.round(2), use_container_width=True, hide_index=True)
            str_lit.caption("Le competizioni con poche osservazioni sono indicative soltanto; non modificare il modello sulla base di un campione piccolo.")

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

        # V12: confronto aggregato con baseline naïve, sugli stessi identici eventi walk-forward.
        str_lit.markdown("#### 📏 V15 — Poisson contro baseline: confronto sullo stesso campione")
        metriche_confronto = [
            ("1X2 — Brier", "Brier 1X2", "Brier 1X2 baseline"),
            ("1X2 — Log Loss", "Log Loss 1X2", "Log Loss 1X2 baseline"),
            ("Goal/No Goal — Brier", "Brier Goal/No Goal", "Brier Goal/No Goal baseline"),
            ("Goal/No Goal — Log Loss", "Log Loss Goal/No Goal", "Log Loss Goal/No Goal baseline"),
        ]
        righe_confronto = []
        for nome, col_modello, col_base in metriche_confronto:
            if col_modello in df_audit.columns and col_base in df_audit.columns:
                val_modello = pd.to_numeric(df_audit[col_modello], errors="coerce").mean()
                val_base = pd.to_numeric(df_audit[col_base], errors="coerce").mean()
                righe_confronto.append({
                    "Metrica": nome,
                    "Poisson": val_modello,
                    "Baseline storica": val_base,
                    "Delta Poisson - baseline": val_modello - val_base,
                    "Migliore": "Poisson" if val_modello < val_base else ("Baseline" if val_base < val_modello else "Pari"),
                })
        if righe_confronto:
            str_lit.dataframe(pd.DataFrame(righe_confronto).round(5), use_container_width=True, hide_index=True)
            str_lit.caption(
                "Per Brier e Log Loss valori più bassi sono migliori. La baseline usa soltanto risultati "
                "precedenti della stessa competizione, con smoothing Laplace; il confronto non implica "
                "che la baseline sia un modello predittivo avanzato."
            )
        else:
            str_lit.info("Confronto baseline non disponibile: mancano le colonne necessarie.")

        # V13: controlli espliciti di campione comune e causalità della baseline.
        str_lit.markdown("#### 🧪 Verifica del campione e della causalità")
        n_comune = int(df_audit[["Brier 1X2", "Brier 1X2 baseline", "Log Loss 1X2", "Log Loss 1X2 baseline",
                                 "Brier Goal/No Goal", "Brier Goal/No Goal baseline",
                                 "Log Loss Goal/No Goal", "Log Loss Goal/No Goal baseline"]].notna().all(axis=1).sum())
        causalita_ok = int(pd.to_numeric(df_audit.get("Controllo causalità baseline", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        k1, k2, k3 = str_lit.columns(3)
        k1.metric("Partite con metriche complete comuni", f"{n_comune}/{len(df_audit)}")
        k2.metric("Baseline senza dati futuri", f"{causalita_ok}/{len(df_audit)}")
        k3.metric("Campionati nel campione", f"{df_audit['Competizione'].nunique()}")
        if n_comune == len(df_audit) and causalita_ok == len(df_audit):
            str_lit.success("Campione comune confermato: Poisson e baseline sono valutati sulle stesse partite; la baseline usa solo date precedenti.")
        else:
            str_lit.warning("Campione incompleto o controllo temporale non superato: interpretare il confronto con cautela.")
        str_lit.info(
            "Nota importante: il walk-forward può valutare soltanto le partite storiche effettivamente presenti in matches_raw. "
            "Se la fonte carica solo una stagione o un sottoinsieme di risultati, questa versione non inventa partite mancanti: "
            "il numero mostrato sopra è il campione realmente disponibile."
        )

        # V13: confronto separato per competizione, evitando di nascondere differenze tra leghe.
        str_lit.markdown("#### 🏆 Risultati per competizione")
        colonne_metriche = ["Brier 1X2", "Brier 1X2 baseline", "Log Loss 1X2", "Log Loss 1X2 baseline",
                            "Brier Goal/No Goal", "Brier Goal/No Goal baseline",
                            "Log Loss Goal/No Goal", "Log Loss Goal/No Goal baseline"]
        df_comp = df_audit.dropna(subset=colonne_metriche).groupby("Competizione", dropna=False).agg(
            Partite=("Esito reale", "size"),
            **{"P(X) media (%)": ("P(X) %", "mean"), "Pareggi reali (%)": ("Esito reale", lambda x: 100 * (x == "X").mean()),
               "No Goal medio (%)": ("P(No Goal) %", "mean"), "No Goal reali (%)": ("No Goal reale", lambda x: 100 * x.mean()),
               "Brier 1X2 Poisson": ("Brier 1X2", "mean"), "Brier 1X2 baseline": ("Brier 1X2 baseline", "mean"),
               "Brier NG Poisson": ("Brier Goal/No Goal", "mean"), "Brier NG baseline": ("Brier Goal/No Goal baseline", "mean")}
        ).reset_index()
        if not df_comp.empty:
            str_lit.dataframe(df_comp.round(4), use_container_width=True, hide_index=True)
        else:
            str_lit.caption("Non ci sono dati sufficienti per il dettaglio per competizione.")

        # V13: stabilità temporale descrittiva per anno; campioni piccoli sono segnalati.
        str_lit.markdown("#### 📆 Stabilità temporale per anno")
        df_anno = df_audit.dropna(subset=colonne_metriche).groupby("Anno", dropna=False).agg(
            Partite=("Esito reale", "size"),
            **{"Pareggi reali (%)": ("Esito reale", lambda x: 100 * (x == "X").mean()),
               "P(X) media (%)": ("P(X) %", "mean"), "No Goal reali (%)": ("No Goal reale", lambda x: 100 * x.mean()),
               "P(No Goal) media (%)": ("P(No Goal) %", "mean"),
               "Brier 1X2 Poisson": ("Brier 1X2", "mean"), "Brier 1X2 baseline": ("Brier 1X2 baseline", "mean"),
               "Brier NG Poisson": ("Brier Goal/No Goal", "mean"), "Brier NG baseline": ("Brier Goal/No Goal baseline", "mean")}
        ).reset_index()
        if not df_anno.empty:
            str_lit.dataframe(df_anno.round(4), use_container_width=True, hide_index=True)
            if (df_anno["Partite"] < 100).any():
                str_lit.caption("Attenzione: gli anni con meno di 100 partite sono indicativi e non consentono conclusioni robuste.")
        else:
            str_lit.caption("Non ci sono dati sufficienti per il dettaglio temporale.")

        with str_lit.expander("🔎 Dettaglio partite usate nell'audit"):
            str_lit.dataframe(df_audit, use_container_width=True, hide_index=True)
            str_lit.download_button(
                "📥 Scarica audit walk-forward CSV",
                df_audit.to_csv(index=False).encode("utf-8-sig"),
                "audit_walk_forward_x_nogoal.csv", "text/csv"
            )

