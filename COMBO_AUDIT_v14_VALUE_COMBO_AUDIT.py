import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import time
import os
import pickle
from datetime import datetime, date
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="COMBO — V13.6 Operativa", page_icon="⚽", layout="centered")

# =====================================================================
# 🔎 AUDIT VERSION v5 — derivata dall'app originale, ma separata.
# NON usare questa versione per sostituire l'app originale: serve solo a
# misurare il modello in modo più rigoroso prima di decidere modifiche.
# Principi dell'audit:
#   1) nessuna calibrazione post-hoc globale nel backtest OOS;
#   2) griglia risultati più ampia (0..12);
#   3) Brier score + log loss oltre alla semplice accuratezza;
#   4) curve di calibrazione per fasce di probabilità;
#   5) confronto esplicito modello/mercato, senza dichiarare vincitori.
# =====================================================================
AUDIT_SCORE_MAX = 12
EPS_PROB = 1e-9
V4_MAX_BUCKETS = 8

CAMPIONATI_DOMESTICI = {
    "Italia - Serie A": {"id_fd": "I1"},
    "Inghilterra - Premier League": {"id_fd": "E0"},
    "Spagna - La Liga": {"id_fd": "SP1"},
    "Germania - Bundesliga": {"id_fd": "D1"},
    "Francia - Ligue 1": {"id_fd": "F1"},
}

CAMPIONATI_COPPE = {
    "🌍 UEFA Champions League": {"code": "CL", "gratis_confermato": True},
    "🌍 UEFA Europa League": {"code": "EL", "gratis_confermato": False},
    "🌍 UEFA Conference League": {"code": "UECL", "gratis_confermato": False},
}

HEADERS_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

K_SHRINKAGE = 10  # stesso principio già validato nell'App Risultati Fissi


# =====================================================================
# 🔧 FIX #4 — MATCHING SQUADRE "MORBIDO"
# Prima: confronto testuale esatto. Con le coppe europee (nomi da
# football-data.org, es. "Manchester United FC") contro i nomi domestici
# (es. "Man United") il match esatto falliva spesso in silenzio, facendo
# scattare il generatore di dati finti (ora rimosso, vedi FIX #1).
# =====================================================================
def normalizza_nome_squadra(nome):
    nome = str(nome)
    for suffisso in [" FC", " CF", " AFC", " AC", " SC", " CFC"]:
        nome = nome.replace(suffisso, "")
    return nome.strip().lower()


def nomi_corrispondono(a, b):
    na, nb = normalizza_nome_squadra(a), normalizza_nome_squadra(b)
    return na == nb or na in nb or nb in na


def codici_stagione(oggi=None):
    oggi = oggi or date.today()
    anno_inizio_corrente = oggi.year if oggi.month >= 7 else oggi.year - 1
    anno_inizio_precedente = anno_inizio_corrente - 1
    fmt = lambda a: f"{a % 100:02d}{(a + 1) % 100:02d}"
    return fmt(anno_inizio_corrente), fmt(anno_inizio_precedente)


def tau_dixon_coles(gc, gt, lc, lt, rho):
    if gc == 0 and gt == 0: return 1 - (lc * lt * rho)
    elif gc == 0 and gt == 1: return 1 + (lc * rho)
    elif gc == 1 and gt == 0: return 1 + (lt * rho)
    elif gc == 1 and gt == 1: return 1 - rho
    return 1.0


def media_ewma(serie, span):
    serie = serie.dropna()
    if len(serie) == 0: return None
    return serie.ewm(span=span, min_periods=1).mean().iloc[-1]


def media_pesata_decadimento(df, colonna, data_riferimento, emivita):
    if colonna not in df.columns: return None
    sub = df[[colonna, 'Date_parsed']].dropna()
    if len(sub) == 0: return None
    giorni = (data_riferimento - sub['Date_parsed']).dt.days.clip(lower=0)
    pesi = 0.5 ** (giorni / emivita)
    tot = pesi.sum()
    return sub[colonna].mean() if tot <= 0 else (sub[colonna] * pesi).sum() / tot


# =====================================================================
# 🔧 Download robusto: header da browser + retry + fallback www/no-www
# (stessa correzione già validata nell'App Risultati Fissi, dopo il
# disservizio di football-data.co.uk causato dal blocco geografico su
# "www" per il traffico non-UK)
# =====================================================================
def scarica_csv_robusto(url, tentativi=3, attesa_secondi=2):
    """FIX #11 — BOM (Byte Order Mark): fixtures.csv inizia con un carattere
    invisibile Unicode che, se non gestito, si attacca al nome della prima
    colonna ("Div" diventa "\\ufeffDiv"), facendo fallire in silenzio ogni
    controllo tipo "if 'Div' in colonne" — nessuna fixture futura veniva mai
    trovata, su nessun campionato, per questo motivo esatto. 'utf-8-sig' lo
    rimuove in fase di decodifica; non fa danno se il file non ha il BOM."""
    varianti_url = [url]
    if "://www." in url:
        varianti_url.append(url.replace("://www.", "://"))
    elif "://" in url:
        varianti_url.append(url.replace("://", "://www."))

    ultimo_errore = None
    for tentativo in range(tentativi):
        for url_prova in varianti_url:
            try:
                resp = requests.get(url_prova, headers=HEADERS_BROWSER, timeout=15)
                resp.raise_for_status()
                testo = resp.content.decode('utf-8-sig', errors='replace')
                df = pd.read_csv(io.StringIO(testo))
                df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]  # rete di sicurezza extra
                return df, None
            except Exception as e:
                ultimo_errore = str(e)
        if tentativo < tentativi - 1:
            time.sleep(attesa_secondi)
    return None, ultimo_errore


@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_campionato(id_fd):
    """FIX B — aggiunto il tag 'Stagione' (precedente/corrente), necessario
    per poter validare il value bet SOLO sui risultati reali più recenti,
    non su quelli già usati per calibrare le medie di lega."""
    codice_corrente, codice_precedente = codici_stagione()
    frames = []
    for codice, label in [(codice_precedente, 'precedente'), (codice_corrente, 'corrente')]:
        url = f"https://football-data.co.uk/mmz4281/{codice}/{id_fd}.csv"
        df, _ = scarica_csv_robusto(url)
        if df is not None:
            df.columns = df.columns.str.strip()
            df['Stagione'] = label
            frames.append(df)
    if frames:
        dati = pd.concat(frames, ignore_index=True, sort=False)
        dati['Date_parsed'] = pd.to_datetime(dati['Date'], errors='coerce', dayfirst=True)
        return dati.dropna(subset=['Date_parsed']).sort_values('Date_parsed').reset_index(drop=True)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def carica_tutti_i_campionati():
    tutti_dati = []
    for c_info in CAMPIONATI_DOMESTICI.values():
        df = carica_dati_campionato(c_info["id_fd"])
        if df is not None:
            tutti_dati.append(df)
    if tutti_dati:
        return pd.concat(tutti_dati, ignore_index=True, sort=False)
    return pd.DataFrame()


def estrai_partite_squadra_intelligente(squadra, df_coppa, df_globale):
    """Matching morbido (FIX #4): prima prova nel dataset della competizione
    stessa, poi nel dataset globale domestico come fallback."""
    if df_coppa is not None and not df_coppa.empty:
        maschera = df_coppa['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra)) | \
                   df_coppa['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
        f_coppa = df_coppa[maschera]
        if len(f_coppa) > 0:
            return f_coppa

    if df_globale is not None and not df_globale.empty:
        maschera = df_globale['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra)) | \
                   df_globale['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
        f_glob = df_globale[maschera]
        if len(f_glob) > 0:
            return f_glob

    return pd.DataFrame()


# =====================================================================
# 🔧 FIX CRITICO #12 — ATTRIBUZIONE CORRETTA DI GOL/TIRI/CORNER
# BUG PRECEDENTE: le partite di una squadra venivano raccolte sia in casa
# sia in trasferta, ma poi il codice leggeva sempre la colonna "di casa"
# (FTHG) come "gol fatti". Per le partite giocate in TRASFERTA, FTHG sono
# i gol dell'AVVERSARIO — quindi circa metà dei dati di attacco erano in
# realtà dati di difesa e viceversa. Effetto: una squadra che segna 3 e
# subisce 0 risultava 1.5/1.5, cioè perfettamente nella media — il modello
# perdeva quasi del tutto la capacità di distinguere squadre forti e deboli,
# e finiva sotto la semplice baseline "vince sempre la squadra di casa".
# Qui i valori vengono attribuiti guardando, partita per partita, se la
# squadra giocava in casa o fuori.
# =====================================================================
def serie_squadra(df, squadra, tipo):
    """tipo: 'gol_fatti', 'gol_subiti', 'tiri_fatti', 'corner_fatti'.
    Ritorna una Series con i valori attribuiti correttamente alla squadra,
    indipendentemente dal fatto che giocasse in casa o in trasferta."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    colonne_casa = {'gol_fatti': 'FTHG', 'gol_subiti': 'FTAG', 'tiri_fatti': 'HST', 'corner_fatti': 'HC'}
    colonne_trasf = {'gol_fatti': 'FTAG', 'gol_subiti': 'FTHG', 'tiri_fatti': 'AST', 'corner_fatti': 'AC'}
    col_c, col_t = colonne_casa[tipo], colonne_trasf[tipo]

    if col_c not in df.columns or col_t not in df.columns:
        return pd.Series(dtype=float)

    gioca_in_casa = df['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
    valori = df[col_c].where(gioca_in_casa, df[col_t])
    return valori.dropna()


def estrai_scontri_diretti(squadra_casa, squadra_trasferta, df_coppa, df_globale):
    frames_tot = []
    if df_coppa is not None and not df_coppa.empty:
        frames_tot.append(df_coppa)
    if df_globale is not None and not df_globale.empty:
        frames_tot.append(df_globale)
    if not frames_tot:
        return pd.DataFrame()

    df_uni = pd.concat(frames_tot, ignore_index=True, sort=False)
    if 'FTHG' not in df_uni.columns or 'FTAG' not in df_uni.columns:
        return pd.DataFrame()

    maschera_diretto = df_uni['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra_casa)) & \
                        df_uni['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra_trasferta))
    maschera_inverso = df_uni['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra_trasferta)) & \
                        df_uni['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra_casa))

    h2h = df_uni[(df_uni['FTHG'].notna()) & (df_uni['FTAG'].notna()) & (maschera_diretto | maschera_inverso)].copy()

    if 'Date_parsed' in h2h.columns:
        h2h = h2h.sort_values('Date_parsed', ascending=False)
    return h2h.head(5)


# =====================================================================
# 🔧 FIX #1 — VIA IL GENERATORE DI DATI FINTI, DENTRO VERO SHRINKAGE
# Prima: se una squadra non aveva partite trovate, il codice INVENTAVA un
# profilo attacco/difesa calcolato dalla somma dei codici ASCII del nome
# squadra — un numero pseudo-casuale spacciato per statistica.
# Ora: quando i dati specifici sono pochi o assenti, la stima converge
# gradualmente verso la media di lega (shrinkage, stesso principio già
# validato nell'App Risultati Fissi) invece di inventare un profilo finto.
# Il chiamante riceve n_casa/n_trasf per mostrare un avviso onesto quando
# il campione specifico è scarso.
# =====================================================================
def calcola_modello_completo(giocate_coppa, squadra_casa, squadra_trasferta, rho, ewma_span,
                              emivita, df_globale, data_riferimento=None):
    giocate_validi = giocate_coppa.dropna(subset=['FTHG', 'FTAG']) if giocate_coppa is not None else pd.DataFrame()
    if data_riferimento is None:
        data_riferimento = giocate_validi['Date_parsed'].max() if not giocate_validi.empty else pd.Timestamp(date.today())

    # Baseline di lega/competizione. Se il campione della competizione è
    # troppo piccolo (tipico per le coppe a inizio stagione), lo arricchiamo
    # con il dataset domestico globale per una stima più stabile.
    base_per_media = giocate_validi
    if len(giocate_validi) < 20 and df_globale is not None and not df_globale.empty:
        base_per_media = pd.concat([giocate_validi, df_globale.dropna(subset=['FTHG', 'FTAG'])], ignore_index=True, sort=False)

    m_gol_casa = media_pesata_decadimento(base_per_media, 'FTHG', data_riferimento, emivita) or 1.65
    m_gol_trasf = media_pesata_decadimento(base_per_media, 'FTAG', data_riferimento, emivita) or 1.25
    m_tiri_casa_lega = media_pesata_decadimento(base_per_media, 'HST', data_riferimento, emivita) or 4.8
    m_tiri_trasf_lega = media_pesata_decadimento(base_per_media, 'AST', data_riferimento, emivita) or 4.1
    m_corner_casa_lega = media_pesata_decadimento(base_per_media, 'HC', data_riferimento, emivita) or 5.4
    m_corner_trasf_lega = media_pesata_decadimento(base_per_media, 'AC', data_riferimento, emivita) or 4.6

    forma_casa = estrai_partite_squadra_intelligente(squadra_casa, giocate_coppa, df_globale)
    forma_trasf = estrai_partite_squadra_intelligente(squadra_trasferta, giocate_coppa, df_globale)
    n_casa, n_trasf = len(forma_casa), len(forma_trasf)

    # FIX CRITICO #12 — i valori vengono attribuiti alla squadra giusta
    # guardando, partita per partita, se giocava in casa o in trasferta.
    gf_casa_rec = media_ewma(serie_squadra(forma_casa, squadra_casa, 'gol_fatti'), ewma_span) if n_casa else None
    gs_casa_rec = media_ewma(serie_squadra(forma_casa, squadra_casa, 'gol_subiti'), ewma_span) if n_casa else None
    gf_trasf_rec = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'gol_fatti'), ewma_span) if n_trasf else None
    gs_trasf_rec = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'gol_subiti'), ewma_span) if n_trasf else None

    tiri_casa = media_ewma(serie_squadra(forma_casa, squadra_casa, 'tiri_fatti'), ewma_span) if n_casa else None
    corner_casa = media_ewma(serie_squadra(forma_casa, squadra_casa, 'corner_fatti'), ewma_span) if n_casa else None
    tiri_trasf = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'tiri_fatti'), ewma_span) if n_trasf else None
    corner_trasf = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'corner_fatti'), ewma_span) if n_trasf else None

    # Shrinkage: peso -> 0 quando n_casa/n_trasf sono pochi o zero, quindi la
    # stima converge verso il rapporto neutro 1.0 (= "come la media") invece
    # di un profilo inventato. Peso -> 1 quando il campione è ampio, quindi ci
    # si fida del dato specifico della squadra.
    peso_casa = n_casa / (n_casa + K_SHRINKAGE)
    peso_trasf = n_trasf / (n_trasf + K_SHRINKAGE)

    # FIX CRITICO #13 — riferimento corretto per i rapporti attacco/difesa.
    # I dati di una squadra mescolano partite in casa e in trasferta, quindi
    # vanno confrontati con la media di lega COMPLESSIVA (casa+trasferta), non
    # con quella specifica di casa o di trasferta. Prima si confrontava il dato
    # misto della squadra di casa con la media "solo casa" (più alta) e quello
    # della squadra ospite con la media "solo trasferta" (più bassa): risultato,
    # le squadre di casa risultavano sistematicamente sottovalutate e le ospiti
    # sopravvalutate — per questo il modello prevedeva più vittorie in trasferta
    # che in casa, il contrario di come funziona davvero il calcio.
    # Il vantaggio del fattore campo resta comunque applicato, più sotto, dal
    # fatto che lambda_casa usa m_gol_casa e lambda_trasferta usa m_gol_trasf.
    m_gol_complessiva = (m_gol_casa + m_gol_trasf) / 2.0

    rapp_attacco_casa = (gf_casa_rec / max(0.1, m_gol_complessiva)) if gf_casa_rec is not None else 1.0
    rapp_difesa_casa = (gs_casa_rec / max(0.1, m_gol_complessiva)) if gs_casa_rec is not None else 1.0
    rapp_attacco_trasf = (gf_trasf_rec / max(0.1, m_gol_complessiva)) if gf_trasf_rec is not None else 1.0
    rapp_difesa_trasf = (gs_trasf_rec / max(0.1, m_gol_complessiva)) if gs_trasf_rec is not None else 1.0

    attacco_casa = peso_casa * rapp_attacco_casa + (1 - peso_casa) * 1.0
    difesa_casa = peso_casa * rapp_difesa_casa + (1 - peso_casa) * 1.0
    attacco_trasf = peso_trasf * rapp_attacco_trasf + (1 - peso_trasf) * 1.0
    difesa_trasf = peso_trasf * rapp_difesa_trasf + (1 - peso_trasf) * 1.0

    # Stessa logica del FIX #13 anche per tiri e angoli: i dati della squadra
    # sono misti casa+trasferta, quindi il riferimento verso cui convergono
    # (shrinkage) dev'essere la media complessiva, non quella specifica.
    m_tiri_complessiva = (m_tiri_casa_lega + m_tiri_trasf_lega) / 2.0
    m_corner_complessiva = (m_corner_casa_lega + m_corner_trasf_lega) / 2.0

    tiri_casa_finale = peso_casa * tiri_casa + (1 - peso_casa) * m_tiri_complessiva if tiri_casa is not None else m_tiri_casa_lega
    corner_casa_finale = peso_casa * corner_casa + (1 - peso_casa) * m_corner_complessiva if corner_casa is not None else m_corner_casa_lega
    tiri_trasf_finale = peso_trasf * tiri_trasf + (1 - peso_trasf) * m_tiri_complessiva if tiri_trasf is not None else m_tiri_trasf_lega
    corner_trasf_finale = peso_trasf * corner_trasf + (1 - peso_trasf) * m_corner_complessiva if corner_trasf is not None else m_corner_trasf_lega

    lam_c = max(0.2, attacco_casa * difesa_trasf * m_gol_casa)
    lam_t = max(0.2, attacco_trasf * difesa_casa * m_gol_trasf)

    prob_1, prob_x, prob_2 = 0.0, 0.0, 0.0
    prob_goal, prob_nogoal = 0.0, 0.0
    limiti_under = [1.5, 2.5, 3.5]
    prob_under = {l: 0.0 for l in limiti_under}
    multigol_casa = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    multigol_trasf = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    combo_stats = {"1 + Goal": 0.0, "1 + Over 2.5": 0.0, "X + Under 2.5": 0.0, "2 + Goal": 0.0}
    griglia_risultati = []  # FIX #5 — griglia completa per il motore combo libero

    tot_p = 0.0
    for gc in range(AUDIT_SCORE_MAX + 1):
        for gt in range(AUDIT_SCORE_MAX + 1):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            tot_p += p
            segno = 'X' if gc == gt else ('1' if gc > gt else '2')
            griglia_risultati.append({"gc": gc, "gt": gt, "p": p, "segno": segno})

            if segno == '1': prob_1 += p
            elif segno == 'X': prob_x += p
            else: prob_2 += p

            if gc > 0 and gt > 0: prob_goal += p
            else: prob_nogoal += p

            for l in limiti_under:
                if gc + gt < l: prob_under[l] += p

            for mg_key, (mi_c, ma_c) in [("0-1", (0,1)), ("0-2", (0,2)), ("1-2", (1,2)), ("1-3", (1,3)), ("2-3", (2,3)), ("2-4", (2,4))]:
                if mi_c <= gc <= ma_c: multigol_casa[mg_key] += p
                if mi_c <= gt <= ma_c: multigol_trasf[mg_key] += p

            if segno == '1' and gc > 0 and gt > 0: combo_stats["1 + Goal"] += p
            if segno == '1' and (gc + gt) > 2.5: combo_stats["1 + Over 2.5"] += p
            if segno == 'X' and (gc + gt) < 2.5: combo_stats["X + Under 2.5"] += p
            if segno == '2' and gc > 0 and gt > 0: combo_stats["2 + Goal"] += p

    if tot_p > 0:
        f = 100.0 / tot_p
        prob_1, prob_x, prob_2 = prob_1*f, prob_x*f, prob_2*f
        prob_goal, prob_nogoal = prob_goal*f, prob_nogoal*f
        prob_under = {l: v*f for l, v in prob_under.items()}
        multigol_casa = {k: v*f for k, v in multigol_casa.items()}
        multigol_trasf = {k: v*f for k, v in multigol_trasf.items()}
        combo_stats = {k: v*f for k, v in combo_stats.items()}
        for r in griglia_risultati:
            r["p"] *= f

    return {
        "prob_1": prob_1, "prob_X": prob_x, "prob_2": prob_2,
        "prob_goal": prob_goal, "prob_nogoal": prob_nogoal,
        "prob_under": prob_under, "multigol_casa": multigol_casa, "multigol_trasf": multigol_trasf,
        "combo": combo_stats, "griglia": griglia_risultati,
        "angoli_stimati": f"{corner_casa_finale + corner_trasf_finale:.1f}",
        "tiri_stimati": f"{tiri_casa_finale + tiri_trasf_finale:.1f}",
        "n_casa": n_casa, "n_trasf": n_trasf,
    }


# =====================================================================
# 🔧 FIX #5 — MOTORE COMBO LIBERO
# Calcola la probabilità congiunta di qualunque combinazione di segno +
# soglia gol (Over/Under) + Gol/No Gol, leggendo direttamente dalla griglia
# di risultati già calcolata — corretto per costruzione (somma le celle
# della griglia Poisson che soddisfano TUTTE le condizioni insieme, non
# moltiplica probabilità come se fossero eventi indipendenti, cosa che
# gol/segno NON sono).
# =====================================================================
def calcola_combo_libera(griglia, segno=None, soglia_gol=None, tipo_soglia=None, gol_nogol=None):
    tot = 0.0
    for r in griglia:
        gc, gt, p = r["gc"], r["gt"], r["p"]
        ok = True
        if segno and r["segno"] != segno:
            ok = False
        if ok and soglia_gol is not None and tipo_soglia:
            tot_g = gc + gt
            if tipo_soglia == "Over" and not (tot_g > soglia_gol): ok = False
            if tipo_soglia == "Under" and not (tot_g < soglia_gol): ok = False
        if ok and gol_nogol:
            entrambe_segnano = gc > 0 and gt > 0
            if gol_nogol == "Goal" and not entrambe_segnano: ok = False
            if gol_nogol == "NoGoal" and entrambe_segnano: ok = False
        if ok:
            tot += p
    return tot


@st.cache_data(ttl=1800, show_spinner=False)
def carica_fixture_future(id_fd):
    df, _ = scarica_csv_robusto("https://football-data.co.uk/fixtures.csv")
    if df is not None:
        fx = df.copy()
        fx.columns = fx.columns.str.strip()
        if 'Div' in fx.columns:
            fx = fx[fx['Div'] == id_fd].copy()
            fx['Date_parsed'] = pd.to_datetime(fx['Date'], errors='coerce', dayfirst=True)
            oggi = pd.Timestamp(date.today())
            return fx[fx['Date_parsed'] >= oggi].sort_values('Date_parsed').reset_index(drop=True)
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_api_europee(codice_competizione, api_key):
    headers = {"X-Auth-Token": api_key}
    url_matches = f"https://api.football-data.org/v4/competitions/{codice_competizione}/matches"
    try:
        resp = requests.get(url_matches, headers=headers, timeout=15)
        if resp.status_code == 403:
            return "ERRORE_403"
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        rows = []
        for m in matches:
            status = m['status']
            fthg, ftag = None, None
            if status == 'FINISHED':
                fthg = m['score']['fullTime']['home']
                ftag = m['score']['fullTime']['away']
            rows.append({
                'Date': m['utcDate'][:10], 'HomeTeam': m['homeTeam']['name'], 'AwayTeam': m['awayTeam']['name'],
                'FTHG': fthg, 'FTAG': ftag, 'Status': status,
            })
        df = pd.DataFrame(rows)
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.sort_values('Date_parsed').reset_index(drop=True)
    except Exception as e:
        return str(e)


# =====================================================================
# 🔧 FIX #3 — VALUE BET CORRETTO (overround + media multi-bookmaker)
# Prima: EV = probabilità_modello × quota_grezza di UN SOLO bookmaker
# (Bet365), senza depurare la quota dal margine del bookmaker — lo stesso
# bug dell'overround già corretto nell'App Risultati Fissi. Ora: media di
# tutti i bookmaker disponibili nel file, quota "equa" depurata dal margine.
# =====================================================================
def classifica_colonne_quote(colonne):
    apertura_h = [c for c in colonne if c.endswith('H') and not c.endswith('CH') and c not in ['FTHG', 'HTHG', 'PTHG']]
    apertura_d = [c for c in colonne if c.endswith('D') and not c.endswith('CD') and c not in ['FTHG', 'FTAG', 'HTHG', 'HTAG']]
    apertura_a = [c for c in colonne if c.endswith('A') and not c.endswith('CA') and c not in ['FTAG', 'HTAG', 'PTAG']]
    return apertura_h, apertura_d, apertura_a


def quote_mercato_normalizzate(riga, colonne_h, colonne_d, colonne_a):
    vh = [riga[c] for c in colonne_h if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    vd = [riga[c] for c in colonne_d if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    va = [riga[c] for c in colonne_a if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    if not (vh and vd and va): return None
    qh, qd, qa = sum(vh)/len(vh), sum(vd)/len(vd), sum(va)/len(va)
    pih, pid, pia = 1/qh, 1/qd, 1/qa
    over = pih + pid + pia
    return {"q_casa_equa": over/pih, "q_x_equa": over/pid, "q_trasf_equa": over/pia,
            "overround": over, "n_bookmakers": len(vh)}


# =====================================================================
# 🔧 PUNTO C — STIMA APPROSSIMATA DELLA QUOTA COMBO
# Nessun bookmaker pubblica una quota per combo libere tipo "1 + Over 2.5 +
# Goal" nei file gratuiti — solo per i mercati singoli (1X2, Over/Under
# 2.5). Qui stimiamo una quota "come se" i mercati fossero indipendenti
# (moltiplicando le quote eque dei singoli mercati disponibili) — è
# un'APPROSSIMAZIONE, non un dato di mercato reale: segno e gol totali
# nella stessa partita sono in una certa misura correlati, quindi il numero
# vero si discosterà da questo. Il componente Gol/No Gol non ha una quota
# disponibile nel file, quindi non entra nella stima — etichettato chiaramente.
# =====================================================================
def classifica_colonne_over_under(colonne, soglia="2.5"):
    over_cols = [c for c in colonne if c.endswith(f'>{soglia}')]
    under_cols = [c for c in colonne if c.endswith(f'<{soglia}')]
    return over_cols, under_cols


def quote_over_under_normalizzate(riga, colonne_over, colonne_under):
    v_over = [riga[c] for c in colonne_over if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    v_under = [riga[c] for c in colonne_under if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    if not (v_over and v_under): return None
    q_over, q_under = sum(v_over)/len(v_over), sum(v_under)/len(v_under)
    pi_over, pi_under = 1/q_over, 1/q_under
    overround = pi_over + pi_under
    return {"q_over_equa": overround/pi_over, "q_under_equa": overround/pi_under, "n_bookmakers": len(v_over)}


def stima_quota_combo_approssimata(segno, soglia_gol, tipo_soglia, quote_1x2, quote_ou_25):
    """Ritorna (quota_stimata_o_None, lista_componenti_non_prezzate)."""
    fattori = []
    non_prezzate = []
    if segno:
        if quote_1x2:
            mappa = {"1": quote_1x2["q_casa_equa"], "X": quote_1x2["q_x_equa"], "2": quote_1x2["q_trasf_equa"]}
            fattori.append(mappa[segno])
        else:
            non_prezzate.append(f"segno {segno}")
    if soglia_gol is not None and tipo_soglia:
        if soglia_gol == 2.5 and quote_ou_25:
            fattori.append(quote_ou_25["q_over_equa"] if tipo_soglia == "Over" else quote_ou_25["q_under_equa"])
        else:
            non_prezzate.append(f"{tipo_soglia} {soglia_gol}")
    quota = None
    if fattori:
        quota = 1.0
        for f in fattori:
            quota *= f
    return quota, non_prezzate


# =====================================================================
# 🔧 BACKTEST SENZA FILTRO — accuratezza pura del modello
# Diverso dal value bet (che filtra per EV/soglia): qui valutiamo semplicemente
# "quante volte la previsione principale del modello (il segno più probabile)
# ha indovinato il risultato vero?", su TUTTE le partite disponibili, senza
# nessun filtro — la metrica più diretta e senza sorprese sulla bontà di base
# del modello, utile per mandarmi i numeri e controllarli insieme.
# =====================================================================
def esegui_backtest_senza_filtro(dati_completi, rho, ewma_span, emivita, usa_oos, id_fd=None,
                                  usa_calibrazione=False, richiedi_accordo_mercato=False):
    """richiedi_accordo_mercato (IDEA #1): valuta la previsione principale SOLO
    sulle partite dove il modello è d'accordo col favorito del mercato (stesso
    segno con la quota più bassa) — un filtro di fiducia, non una correzione
    della stima come il blending già scartato. Riduce il campione ma, se
    l'idea è valida, dovrebbe alzare l'accuratezza sul sottoinsieme rimasto."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    ha_stagione = 'Stagione' in tutte.columns

    if usa_oos and ha_stagione:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))

    if not indici:
        return None

    calib_info = carica_calibratore(id_fd, "1x2") if (usa_calibrazione and id_fd) else None
    calibratore = calib_info["calibratore"] if calib_info else None
    colonne_h, colonne_d, colonne_a = classifica_colonne_quote(tutte.columns)

    n_partite, n_corrette = 0, 0
    n_scartate_disaccordo, n_scartate_no_quote = 0, 0
    per_segno = {"1": {"previste": 0, "corrette": 0}, "X": {"previste": 0, "corrette": 0}, "2": {"previste": 0, "corrette": 0}}

    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue
        if calibratore is not None:
            m = applica_calibrazione_1x2(m, calibratore)

        probabilita = {"1": m['prob_1'], "X": m['prob_X'], "2": m['prob_2']}
        previsione_principale = max(probabilita, key=probabilita.get)

        if richiedi_accordo_mercato:
            quote = quote_mercato_normalizzate(partita, colonne_h, colonne_d, colonne_a)
            if quote is None:
                n_scartate_no_quote += 1
                continue
            quote_per_segno = {"1": quote["q_casa_equa"], "X": quote["q_x_equa"], "2": quote["q_trasf_equa"]}
            favorito_mercato = min(quote_per_segno, key=quote_per_segno.get)  # quota più bassa = favorito
            if previsione_principale != favorito_mercato:
                n_scartate_disaccordo += 1
                continue

        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        n_partite += 1
        per_segno[previsione_principale]["previste"] += 1
        if previsione_principale == esito:
            n_corrette += 1
            per_segno[previsione_principale]["corrette"] += 1

    return {"n_partite": n_partite, "n_corrette": n_corrette, "per_segno": per_segno,
            "n_scartate_disaccordo": n_scartate_disaccordo, "n_scartate_no_quote": n_scartate_no_quote}


# =====================================================================
# 🔧 PUNTO B — CALIBRAZIONE POST-HOC (1X2 + le 12 combo automatiche)
# Stessa tecnica isotonic regression già validata nell'App Risultati Fissi.
# Un correttore per 1X2 (corregge "Esito Finale" e il Value Bet), più uno
# separato per ciascuna delle 12 combinazioni automatiche (segno × Gol/No
# Gol × Over/Under 2.5) — la verità per allenarle è già nei risultati reali
# che scarichiamo (nessuna fonte dati nuova serve). Le combo libere con
# soglie diverse da 2.5 restano SENZA calibrazione: te lo segnaliamo,
# non lo nascondiamo.
# =====================================================================
LE_12_COMBO = [(s, g, t) for s in ["1", "X", "2"] for g in ["Goal", "NoGoal"] for t in ["Over", "Under"]]



# =====================================================================
# V13 — PLATT SOLO O/U 2.5
# Walk-forward: il fit usa esclusivamente partite precedenti alla partita
# da analizzare. Non modifica 1X2, Goal/No Goal, Multigol o la griglia.
# =====================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def _v13_training_ou_platt(dati, rho, ewma_span, emivita, data_riferimento, n_train=100):
    """Costruisce il training O/U 2.5 senza look-ahead."""
    if dati is None or len(dati) == 0 or 'FTHG' not in dati.columns or 'FTAG' not in dati.columns:
        return pd.DataFrame()
    df = dati.copy()
    if 'Date_parsed' not in df.columns:
        return pd.DataFrame()
    df = df[df['FTHG'].notna() & df['FTAG'].notna() & df['Date_parsed'].notna()].copy()
    if data_riferimento is not None and pd.notna(data_riferimento):
        df = df[df['Date_parsed'] < data_riferimento].copy()
    df = df.sort_values('Date_parsed').tail(max(int(n_train) + 40, int(n_train))).copy()
    rows = []
    for _, r in df.iterrows():
        d = r['Date_parsed']
        prior = df[df['Date_parsed'] < d].copy()
        if len(prior) < 20:
            continue
        try:
            m = calcola_modello_completo(
                prior,
                r['HomeTeam'], r['AwayTeam'],
                rho, ewma_span, emivita, pd.DataFrame(),
                data_riferimento=d
            )
            if not m or 'prob_under' not in m or 2.5 not in m['prob_under']:
                continue
            p_over = 1.0 - float(m['prob_under'][2.5]) / 100.0
            y_over = int(float(r['FTHG']) + float(r['FTAG']) > 2.5)
            if np.isfinite(p_over):
                rows.append({'p_raw': float(np.clip(p_over, 0.001, 0.999)), 'y_over': y_over, 'Date': d})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Date').tail(int(n_train)).reset_index(drop=True)


def _v13_logit(p):
    """Logit numerically stable per una probabilita 0..1."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _v13_fit_platt_ou(df_train):
    """Fit Platt standard e restituisce (modello, pendenza, motivo).

    La calibrazione viene applicata solo quando la pendenza e' positiva:
    una calibrazione valida deve preservare l'ordinamento delle probabilita'.
    """
    if df_train is None or len(df_train) < 50:
        return None, np.nan, "campione insufficiente"
    y = df_train['y_over'].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None, np.nan, "un solo esito presente nel training"
    p = df_train['p_raw'].astype(float).to_numpy()
    x = np.array([_v13_logit(v) for v in p], dtype=float).reshape(-1, 1)
    model = LogisticRegression(solver='lbfgs', C=1e6, max_iter=1000)
    model.fit(x, y)
    coef = float(model.coef_[0, 0])
    if not np.isfinite(coef) or coef <= 0.0:
        return None, coef, "fit non monotono: pendenza non positiva"
    return model, coef, "applicabile"


def _v13_predict_platt_ou(p_raw_over, calibratore):
    if calibratore is None:
        return float(np.clip(p_raw_over, 0.001, 0.999))
    x = [[_v13_logit(p_raw_over)]]
    return float(np.clip(calibratore.predict_proba(x)[0, 1], 0.001, 0.999))


def _v13_apply_platt_ou(modello, calibratore):
    """Applica Platt esclusivamente alla probabilita O/U 2.5 del match."""
    if modello is None or calibratore is None:
        return modello
    p_raw_over = 1.0 - float(modello['prob_under'][2.5]) / 100.0
    p_platt_over = _v13_predict_platt_ou(p_raw_over, calibratore)
    modello['prob_under'][2.5] = (1.0 - p_platt_over) * 100.0
    return modello

def _path_calibratore(id_fd, chiave="1x2"):
    return f"calibratore_combo_{chiave}_{id_fd}.pkl"


def _salva_calibratore(id_fd, chiave, calibratore, n_oss):
    try:
        with open(_path_calibratore(id_fd, chiave), "wb") as f:
            pickle.dump({"calibratore": calibratore, "n_osservazioni": n_oss,
                         "timestamp": datetime.now().isoformat(timespec="minutes")}, f)
    except Exception:
        pass


def carica_calibratore(id_fd, chiave="1x2"):
    path = _path_calibratore(id_fd, chiave)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _allena_isotonic(osservazioni):
    if len(osservazioni) < 100:
        return None
    x = np.array([p for p, _ in osservazioni])
    y = np.array([1.0 if av else 0.0 for _, av in osservazioni])
    calibratore = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    calibratore.fit(x, y)
    return calibratore


def applica_calibrazione_1x2(modello, calibratore):
    if calibratore is None or modello is None:
        return modello
    p1 = float(calibratore.predict([modello['prob_1']/100])[0])
    px = float(calibratore.predict([modello['prob_X']/100])[0])
    p2 = float(calibratore.predict([modello['prob_2']/100])[0])
    tot = p1 + px + p2
    if tot <= 0:
        return modello
    m = dict(modello)
    m['prob_1'], m['prob_X'], m['prob_2'] = p1/tot*100, px/tot*100, p2/tot*100
    return m


def allena_tutte_le_calibrazioni(dati_completi, rho, ewma_span, emivita, id_fd):
    """Un'unica passata sui dati storici: per ogni partita calcola il modello
    UNA volta, poi costruisce le osservazioni (predetto, avverato) per 1X2 e
    per tutte e 12 le combo insieme — efficiente, una sola passata."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    oss_1x2 = []
    oss_combo = {c: [] for c in LE_12_COMBO}

    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue

        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        oss_1x2.append((m['prob_1']/100, esito == '1'))
        oss_1x2.append((m['prob_X']/100, esito == 'X'))
        oss_1x2.append((m['prob_2']/100, esito == '2'))

        tot_gol_reale = partita['FTHG'] + partita['FTAG']
        entrambe_reale = partita['FTHG'] > 0 and partita['FTAG'] > 0
        for (segno_c, gg_c, tipo_c) in LE_12_COMBO:
            prob_c = calcola_combo_libera(m['griglia'], segno=segno_c, soglia_gol=2.5, tipo_soglia=tipo_c, gol_nogol=gg_c)
            avverato_c = (segno_c == esito) and \
                         ((gg_c == "Goal") == entrambe_reale) and \
                         ((tipo_c == "Over") == (tot_gol_reale > 2.5))
            oss_combo[(segno_c, gg_c, tipo_c)].append((prob_c/100, avverato_c))

    risultati = {"1x2": {"n": len(oss_1x2), "ok": False}}
    cal_1x2 = _allena_isotonic(oss_1x2)
    if cal_1x2:
        _salva_calibratore(id_fd, "1x2", cal_1x2, len(oss_1x2))
        risultati["1x2"]["ok"] = True

    for chiave_c, oss_c in oss_combo.items():
        nome_chiave = f"{chiave_c[0]}_{chiave_c[1]}_{chiave_c[2]}"
        cal_c = _allena_isotonic(oss_c)
        risultati[nome_chiave] = {"n": len(oss_c), "ok": cal_c is not None}
        if cal_c:
            _salva_calibratore(id_fd, nome_chiave, cal_c, len(oss_c))

    return risultati


# =====================================================================
# 🔧 BACKTEST COMBO — stessa logica del backtest 1X2 senza filtro, applicata
# alla combo automatica più probabile di ogni partita (tra le 12 possibili).
# Usa la calibrazione per-combo se disponibile e attiva, e la stessa verità
# (risultato reale) già usata per allenarle — nessuna fonte dati nuova.
# =====================================================================
def esegui_backtest_combo(dati_completi, rho, ewma_span, emivita, usa_oos, id_fd=None,
                           usa_calibrazione=False, richiedi_accordo_mercato=False, richiedi_accordo_ou=False):
    """richiedi_accordo_mercato: valuta solo le combo il cui SEGNO è d'accordo
    col favorito del mercato (idea #1, già validata su 1X2: +5/+10 punti).
    richiedi_accordo_ou (idea #1b, DA VERIFICARE): in aggiunta, richiede che
    anche la componente OVER/UNDER sia d'accordo con la quota di mercato
    Over/Under 2.5 — copre una seconda delle tre dimensioni di una combo
    (il filtro sul solo segno ne copre una su tre, la terza — Gol/No Gol —
    resta comunque non filtrabile: non esiste una quota di mercato gratuita
    per quel mercato nei file che usiamo)."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    ha_stagione = 'Stagione' in tutte.columns

    if usa_oos and ha_stagione:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))

    if not indici:
        return None

    colonne_h, colonne_d, colonne_a = classifica_colonne_quote(tutte.columns)
    colonne_over, colonne_under = classifica_colonne_over_under(tutte.columns, "2.5")
    n_partite, n_corrette = 0, 0
    n_scartate_disaccordo, n_scartate_disaccordo_ou, n_scartate_no_quote = 0, 0, 0
    per_combo = {f"{s}_{g}_{t}": {"previste": 0, "corrette": 0} for (s, g, t) in LE_12_COMBO}

    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue

        probabilita_combo = {}
        for (s, g, t) in LE_12_COMBO:
            nome_chiave = f"{s}_{g}_{t}"
            prob_grezza = calcola_combo_libera(m['griglia'], segno=s, soglia_gol=2.5, tipo_soglia=t, gol_nogol=g)
            prob_finale = prob_grezza
            if usa_calibrazione and id_fd:
                calib_c = carica_calibratore(id_fd, nome_chiave)
                if calib_c:
                    prob_finale = float(calib_c["calibratore"].predict([prob_grezza/100])[0]) * 100
            probabilita_combo[nome_chiave] = prob_finale

        combo_prevista = max(probabilita_combo, key=probabilita_combo.get)
        s_p, g_p, t_p = combo_prevista.split("_")

        if richiedi_accordo_mercato or richiedi_accordo_ou:
            quote = quote_mercato_normalizzate(partita, colonne_h, colonne_d, colonne_a)
            if quote is None:
                n_scartate_no_quote += 1
                continue

        if richiedi_accordo_mercato:
            quote_per_segno = {"1": quote["q_casa_equa"], "X": quote["q_x_equa"], "2": quote["q_trasf_equa"]}
            favorito_mercato = min(quote_per_segno, key=quote_per_segno.get)
            if s_p != favorito_mercato:
                n_scartate_disaccordo += 1
                continue

        if richiedi_accordo_ou:
            quote_ou = quote_over_under_normalizzate(partita, colonne_over, colonne_under)
            if quote_ou is None:
                n_scartate_no_quote += 1
                continue
            favorito_ou_mercato = "Over" if quote_ou["q_over_equa"] < quote_ou["q_under_equa"] else "Under"
            if t_p != favorito_ou_mercato:
                n_scartate_disaccordo_ou += 1
                continue

        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        tot_gol_reale = partita['FTHG'] + partita['FTAG']
        entrambe_reale = partita['FTHG'] > 0 and partita['FTAG'] > 0
        avverata = (s_p == esito) and ((g_p == "Goal") == entrambe_reale) and ((t_p == "Over") == (tot_gol_reale > 2.5))

        n_partite += 1
        per_combo[combo_prevista]["previste"] += 1
        if avverata:
            n_corrette += 1
            per_combo[combo_prevista]["corrette"] += 1

    return {"n_partite": n_partite, "n_corrette": n_corrette, "per_combo": per_combo,
            "n_scartate_disaccordo": n_scartate_disaccordo, "n_scartate_disaccordo_ou": n_scartate_disaccordo_ou,
            "n_scartate_no_quote": n_scartate_no_quote}


# =====================================================================
# 🧪 METRICHE AUDIT
# =====================================================================
def _clip_prob(p):
    return float(np.clip(p, EPS_PROB, 1.0 - EPS_PROB))

def calcola_metriche_1x2(risultati):
    """Calcola accuracy, Brier multiclass e log loss dalle osservazioni OOS."""
    if not risultati:
        return None
    n = len(risultati)
    corrette = 0
    brier = 0.0
    logloss = 0.0
    bins = {k: {"n": 0, "somma_p": 0.0, "positivi": 0} for k in ["0-20", "20-40", "40-60", "60-80", "80-100"]}
    for r in risultati:
        probs = r["probs"]
        esito = r["esito"]
        pred = max(probs, key=probs.get)
        if pred == esito:
            corrette += 1
        y = {"1": 0.0, "X": 0.0, "2": 0.0}
        y[esito] = 1.0
        brier += sum((probs[k] - y[k]) ** 2 for k in y)
        logloss -= np.log(_clip_prob(probs[esito]))
        pmax = max(probs.values()) * 100.0
        if pmax < 20: b = "0-20"
        elif pmax < 40: b = "20-40"
        elif pmax < 60: b = "40-60"
        elif pmax < 80: b = "60-80"
        else: b = "80-100"
        bins[b]["n"] += 1
        bins[b]["somma_p"] += pmax
        bins[b]["positivi"] += int(pred == esito)
    return {
        "n": n,
        "accuracy": corrette / n * 100.0,
        "brier": brier / n,
        "log_loss": logloss / n,
        "bins": bins,
    }

def esegui_audit_1x2(dati_completi, rho, ewma_span, emivita, stagione_corrente=True):
    """Backtest rigoroso: per ogni partita usa esclusivamente partite precedenti.
    NON carica calibratori salvati: questa funzione misura il modello grezzo.
    """
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    if stagione_corrente and 'Stagione' in tutte.columns:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))
    osservazioni = []
    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span, emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None:
            continue
        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        osservazioni.append({
            "probs": {"1": m['prob_1']/100.0, "X": m['prob_X']/100.0, "2": m['prob_2']/100.0},
            "esito": esito,
            "data": partita.get('Date_parsed'),
            "casa": partita['HomeTeam'],
            "trasferta": partita['AwayTeam'],
        })
    return osservazioni



# =====================================================================
# 💰 V3 — BACKTEST ECONOMICO OOS (ROI / EV)
# Valuta il modello con quote storiche PRE-PARTITA disponibili nei CSV.
# Importante: per ROI usiamo la MEDIA DELLE QUOTE GREZZE dei bookmaker
# disponibili, non la quota "equa" depurata dal margine. È quindi una
# proxy storica del prezzo medio disponibile, non la prova di esecuzione
# presso un bookmaker specifico.
# Il modello viene sempre calcolato usando esclusivamente il passato.
# Non vengono usati calibratori salvati.
# Le combo 1X2 + O/U + Goal/NoGoal NON hanno una quota storica combo nei
# file gratuiti: non inventiamo un ROI combo. Le combo restano valutate
# separatamente per accuratezza, come nella V2.
# =====================================================================
def quote_medie_grezze_1x2(riga, colonne_h, colonne_d, colonne_a):
    def media_valori(cols):
        vals = []
        for c in cols:
            v = riga.get(c)
            if pd.notna(v) and isinstance(v, (int, float)) and v > 1.0:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None
    qh, qd, qa = media_valori(colonne_h), media_valori(colonne_d), media_valori(colonne_a)
    if qh is None or qd is None or qa is None:
        return None
    return {"1": qh, "X": qd, "2": qa,
            "n_bookmakers": min(sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_h),
                                  sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_d),
                                  sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_a))}


def quote_medie_grezze_ou(riga, colonne_over, colonne_under):
    def media_valori(cols):
        vals = []
        for c in cols:
            v = riga.get(c)
            if pd.notna(v) and isinstance(v, (int, float)) and v > 1.0:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None
    qo, qu = media_valori(colonne_over), media_valori(colonne_under)
    if qo is None or qu is None:
        return None
    return {"Over": qo, "Under": qu,
            "n_bookmakers": min(sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_over),
                                  sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_under))}


def _stat_roi(n_bet, wins, profit, ev_sum, odds_sum, drawdown):
    if n_bet <= 0:
        return {"scommesse": 0, "vinte": 0, "strike_rate": None, "profitto": 0.0,
                "roi": None, "ev_medio": None, "quota_media": None, "max_drawdown": 0.0}
    return {"scommesse": n_bet, "vinte": wins, "strike_rate": wins / n_bet * 100.0,
            "profitto": profit, "roi": profit / n_bet * 100.0,
            "ev_medio": ev_sum / n_bet * 100.0, "quota_media": odds_sum / n_bet,
            "max_drawdown": drawdown}


def _aggiorna_pnl(equity, profit, peak):
    equity += profit
    peak = max(peak, equity)
    return equity, peak, peak - equity


def esegui_backtest_roi_1x2(dati_completi, rho, ewma_span, emivita, usa_oos=True,
                            soglia_ev=0.0, richiedi_accordo_mercato=False):
    """ROI OOS sul segno 1X2. Una puntata flat da 1 unita per selezione.
    La selezione è il segno più probabile del modello raw. Si scommette solo
    se EV = p_modello * quota_media_grezza - 1 >= soglia_ev.
    """
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    if usa_oos and 'Stagione' in tutte.columns:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))
    colonne_h, colonne_d, colonne_a = classifica_colonne_quote(tutte.columns)
    bets = []
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue
        q = quote_medie_grezze_1x2(partita, colonne_h, colonne_d, colonne_a)
        if q is None: continue
        probs = {"1": m['prob_1']/100.0, "X": m['prob_X']/100.0, "2": m['prob_2']/100.0}
        scelta = max(probs, key=probs.get)
        if richiedi_accordo_mercato:
            favorito_mercato = min(q, key=q.get)
            if scelta != favorito_mercato:
                continue
        quota = q[scelta]
        ev = probs[scelta] * quota - 1.0
        if ev < soglia_ev:
            continue
        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        win = scelta == esito
        profit = quota - 1.0 if win else -1.0
        equity, peak, dd = _aggiorna_pnl(equity, profit, peak)
        max_dd = max(max_dd, dd)
        bets.append({"data": partita.get('Date_parsed'), "casa": partita['HomeTeam'], "trasferta": partita['AwayTeam'],
                     "mercato": "1X2", "scelta": scelta, "prob_modello": probs[scelta], "quota": quota,
                     "ev": ev, "esito": esito, "vinta": win, "profitto": profit})
    n = len(bets)
    return _stat_roi(n, sum(b["vinta"] for b in bets), sum(b["profitto"] for b in bets),
                     sum(b["ev"] for b in bets), sum(b["quota"] for b in bets), max_dd), pd.DataFrame(bets)


def esegui_backtest_roi_ou(dati_completi, rho, ewma_span, emivita, usa_oos=True,
                           soglia_ev=0.0, richiedi_accordo_mercato=False):
    """ROI OOS su Over/Under 2.5 con quota media grezza pre-partita."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    if usa_oos and 'Stagione' in tutte.columns:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))
    colonne_over, colonne_under = classifica_colonne_over_under(tutte.columns, "2.5")
    bets = []
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue
        q = quote_medie_grezze_ou(partita, colonne_over, colonne_under)
        if q is None: continue
        probs = {"Over": (100.0 - m['prob_under'][2.5]) / 100.0,
                 "Under": m['prob_under'][2.5] / 100.0}
        scelta = max(probs, key=probs.get)
        if richiedi_accordo_mercato:
            favorito_mercato = min(q, key=q.get)
            if scelta != favorito_mercato:
                continue
        quota = q[scelta]
        ev = probs[scelta] * quota - 1.0
        if ev < soglia_ev:
            continue
        tot_gol = partita['FTHG'] + partita['FTAG']
        esito = "Over" if tot_gol > 2.5 else "Under"
        win = scelta == esito
        profit = quota - 1.0 if win else -1.0
        equity, peak, dd = _aggiorna_pnl(equity, profit, peak)
        max_dd = max(max_dd, dd)
        bets.append({"data": partita.get('Date_parsed'), "casa": partita['HomeTeam'], "trasferta": partita['AwayTeam'],
                     "mercato": "O/U 2.5", "scelta": scelta, "prob_modello": probs[scelta], "quota": quota,
                     "ev": ev, "esito": esito, "vinta": win, "profitto": profit})
    n = len(bets)
    return _stat_roi(n, sum(b["vinta"] for b in bets), sum(b["profitto"] for b in bets),
                     sum(b["ev"] for b in bets), sum(b["quota"] for b in bets), max_dd), pd.DataFrame(bets)


def mostra_roi(ris, titolo):
    st.write(f"**{titolo}**")
    if ris["scommesse"] == 0:
        st.warning("Nessuna scommessa qualificata con questi criteri.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scommesse", ris["scommesse"])
    c2.metric("Strike rate", f"{ris['strike_rate']:.1f}%")
    c3.metric("ROI", f"{ris['roi']:+.1f}%")
    c4.metric("Profitto (1u)", f"{ris['profitto']:+.2f}")
    st.caption(f"Quota media: {ris['quota_media']:.2f} — EV medio teorico: {ris['ev_medio']:+.1f}% — Max drawdown: {ris['max_drawdown']:.2f} unità")


def mostra_sezione_roi_v3(dati, campionato, id_fd, rho_val, ewma_span_val, emivita_val):
    st.divider()
    st.write("**💰 V3 — Backtest economico OOS: ROI / EV**")
    st.caption("Testa il modello grezzo su quote storiche pre-partita. Una puntata = 1 unità. "
               "La quota è la media grezza dei bookmaker disponibili nel CSV: è una proxy storica, non una garanzia di esecuzione.")
    st.warning("⚠️ Questo test NON usa la quota combo stimata. Le combo libere non hanno una quota storica combo nel dataset gratuito, quindi qui misuriamo 1X2 e O/U 2.5 separatamente.")
    col1, col2 = st.columns(2)
    with col1:
        periodo = st.selectbox("Periodo ROI", ["Stagione corrente OOS", "Tutto lo storico"], key=f"roi_periodo_{id_fd}")
    with col2:
        soglia_ev_pct = st.selectbox("EV minimo", [0, 5, 10, 15, 20], index=0, key=f"roi_ev_{id_fd}")
    accordo_segno = st.checkbox("Solo se modello e mercato concordano sul segno", value=False, key=f"roi_segno_{id_fd}")
    accordo_ou = st.checkbox("Solo se modello e mercato concordano su Over/Under 2.5", value=False, key=f"roi_ou_{id_fd}")
    usa_oos = periodo == "Stagione corrente OOS"
    soglia = soglia_ev_pct / 100.0
    if st.button("💰 Esegui backtest ROI / EV", key=f"roi_run_{id_fd}"):
        with st.spinner("Calcolo ROI/EV OOS in corso..."):
            if accordo_ou:
                # Il filtro O/U vale per entrambi i mercati; viene applicato
                # separatamente nella funzione del mercato corrispondente.
                pass
            r1, df1 = esegui_backtest_roi_1x2(dati, rho_val, ewma_span_val, emivita_val, usa_oos, soglia, accordo_segno)
            r2, df2 = esegui_backtest_roi_ou(dati, rho_val, ewma_span_val, emivita_val, usa_oos, soglia, accordo_ou)
            mostra_roi(r1, "1X2 — selezione più probabile del modello")
            mostra_roi(r2, "Over/Under 2.5 — selezione più probabile del modello")
            if not df1.empty or not df2.empty:
                df_out = pd.concat([df1, df2], ignore_index=True)
                st.dataframe(df_out[['data','casa','trasferta','mercato','scelta','prob_modello','quota','ev','esito','vinta','profitto']],
                             use_container_width=True, hide_index=True)
                st.download_button("⬇️ Scarica dettaglio ROI come CSV", data=df_out.to_csv(index=False).encode('utf-8'),
                                   file_name=f"roi_ev_{id_fd}.csv", mime="text/csv", key=f"roi_dl_{id_fd}")
            st.caption("Interpretazione: ROI > 0 significa profitto storico sul campione e sulle quote usate dal test. "
                       "Non implica che il risultato si ripeta nel futuro. EV è il valore teorico calcolato dalla probabilità del modello e dalla quota storica usata.")


# =====================================================================
# 🔎 V4 — AUDIT QUOTE / EV
# Confronta quattro oggetti distinti, senza confonderli:
#   1) quota media grezza dei bookmaker (proxy del prezzo storico);
#   2) probabilità implicita dalla quota media grezza;
#   3) probabilità implicita normalizzata bookmaker-per-bookmaker;
#   4) probabilità del modello e relativo EV teorico.
# Questo audit NON modifica il modello e NON usa calibratori salvati.
# =====================================================================
def _float_quote(v):
    return float(v) if pd.notna(v) and isinstance(v, (int, float, np.number)) and float(v) > 1.0 else None


def _bookmaker_prefix(col, suffixes):
    for suf in suffixes:
        if str(col).endswith(suf):
            return str(col)[:-len(suf)]
    return None


def _audit_quote_1x2(riga, colonne_h, colonne_d, colonne_a):
    gruppi = {}
    for c in colonne_h:
        pre = _bookmaker_prefix(c, ['H'])
        q = _float_quote(riga.get(c))
        if pre and q is not None: gruppi.setdefault(pre, {})['1'] = q
    for c in colonne_d:
        pre = _bookmaker_prefix(c, ['D'])
        q = _float_quote(riga.get(c))
        if pre and q is not None: gruppi.setdefault(pre, {})['X'] = q
    for c in colonne_a:
        pre = _bookmaker_prefix(c, ['A'])
        q = _float_quote(riga.get(c))
        if pre and q is not None: gruppi.setdefault(pre, {})['2'] = q

    completi = [g for g in gruppi.values() if all(k in g for k in ('1','X','2'))]
    if not completi:
        return None

    q_media = {k: float(np.mean([g[k] for g in completi])) for k in ('1','X','2')}
    implied_raw = {k: 1.0/q_media[k] for k in q_media}
    over_raw = sum(implied_raw.values())
    p_norm_da_media_quote = {k: implied_raw[k]/over_raw for k in implied_raw}

    norm_per_book = []
    overrounds = []
    for g in completi:
        inv = {k: 1.0/g[k] for k in ('1','X','2')}
        ov = sum(inv.values())
        overrounds.append(ov)
        norm_per_book.append({k: inv[k]/ov for k in inv})
    p_market = {k: float(np.mean([x[k] for x in norm_per_book])) for k in ('1','X','2')}

    return {
        'quota_media': q_media,
        'p_implicita_quota_media': implied_raw,
        'overround_quota_media': over_raw,
        'p_norm_da_quota_media': p_norm_da_media_quote,
        'p_market_norm_media_bookmaker': p_market,
        'overround_medio_bookmaker': float(np.mean(overrounds)),
        'n_bookmakers': len(completi),
    }


def _audit_quote_ou(riga, colonne_over, colonne_under):
    gruppi = {}
    for c in colonne_over:
        pre = _bookmaker_prefix(c, ['>2.5'])
        q = _float_quote(riga.get(c))
        if pre and q is not None: gruppi.setdefault(pre, {})['Over'] = q
    for c in colonne_under:
        pre = _bookmaker_prefix(c, ['<2.5'])
        q = _float_quote(riga.get(c))
        if pre and q is not None: gruppi.setdefault(pre, {})['Under'] = q
    completi = [g for g in gruppi.values() if 'Over' in g and 'Under' in g]
    if not completi:
        return None
    q_media = {k: float(np.mean([g[k] for g in completi])) for k in ('Over','Under')}
    implied_raw = {k: 1.0/q_media[k] for k in q_media}
    over_raw = sum(implied_raw.values())
    p_norm_da_media_quote = {k: implied_raw[k]/over_raw for k in implied_raw}
    norm_per_book, overrounds = [], []
    for g in completi:
        inv = {k: 1.0/g[k] for k in ('Over','Under')}
        ov = sum(inv.values())
        overrounds.append(ov)
        norm_per_book.append({k: inv[k]/ov for k in inv})
    p_market = {k: float(np.mean([x[k] for x in norm_per_book])) for k in ('Over','Under')}
    return {
        'quota_media': q_media,
        'p_implicita_quota_media': implied_raw,
        'overround_quota_media': over_raw,
        'p_norm_da_quota_media': p_norm_da_media_quote,
        'p_market_norm_media_bookmaker': p_market,
        'overround_medio_bookmaker': float(np.mean(overrounds)),
        'n_bookmakers': len(completi),
    }


def _audit_quote_statistiche(df):
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for lo, hi, label in [(0, .4, '0-40%'), (.4, .5, '40-50%'), (.5, .6, '50-60%'),
                          (.6, .7, '60-70%'), (.7, .8, '70-80%'), (.8, 1.01, '80-100%')]:
        x = df[(df['prob_modello'] >= lo) & (df['prob_modello'] < hi)]
        if x.empty: continue
        rows.append({
            'Fascia prob. modello': label,
            'Bet': len(x),
            'Prob. modello media': x['prob_modello'].mean()*100,
            'Prob. mercato fair media': x['p_mercato_fair'].mean()*100,
            'Frequenza reale': x['vinta'].mean()*100,
            'EV teorico medio': x['ev_raw'].mean()*100,
            'ROI': x['profitto'].sum()/len(x)*100,
        })
    return pd.DataFrame(rows)


def esegui_audit_quote_1x2(dati_completi, rho, ewma_span, emivita):
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    colonne_h, colonne_d, colonne_a = classifica_colonne_quote(tutte.columns)
    rows = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span, emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue
        q = _audit_quote_1x2(partita, colonne_h, colonne_d, colonne_a)
        if q is None: continue
        probs = {'1': m['prob_1']/100, 'X': m['prob_X']/100, '2': m['prob_2']/100}
        scelta = max(probs, key=probs.get)
        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        quota = q['quota_media'][scelta]
        p_imp = q['p_implicita_quota_media'][scelta]
        p_fair = q['p_market_norm_media_bookmaker'][scelta]
        ev_raw = probs[scelta]*quota - 1
        win = scelta == esito
        profitto = quota-1 if win else -1
        rows.append({
            'data': partita.get('Date_parsed'), 'casa': partita['HomeTeam'], 'trasferta': partita['AwayTeam'],
            'scelta': scelta, 'prob_modello': probs[scelta], 'quota': quota,
            'p_mercato_implicita_raw': p_imp, 'p_mercato_fair': p_fair,
            'edge_vs_fair': probs[scelta]-p_fair, 'overround_medio': q['overround_medio_bookmaker']-1,
            'ev_raw': ev_raw, 'esito': esito, 'vinta': win, 'profitto': profitto,
            'n_bookmakers': q['n_bookmakers'],
        })
    return pd.DataFrame(rows)


def esegui_audit_quote_ou(dati_completi, rho, ewma_span, emivita):
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    colonne_over, colonne_under = classifica_colonne_over_under(tutte.columns, '2.5')
    rows = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span, emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue
        q = _audit_quote_ou(partita, colonne_over, colonne_under)
        if q is None: continue
        probs = {'Over': (100-m['prob_under'][2.5])/100, 'Under': m['prob_under'][2.5]/100}
        scelta = max(probs, key=probs.get)
        tot = partita['FTHG'] + partita['FTAG']
        esito = 'Over' if tot > 2.5 else 'Under'
        quota = q['quota_media'][scelta]
        p_imp = q['p_implicita_quota_media'][scelta]
        p_fair = q['p_market_norm_media_bookmaker'][scelta]
        ev_raw = probs[scelta]*quota - 1
        win = scelta == esito
        profitto = quota-1 if win else -1
        rows.append({
            'data': partita.get('Date_parsed'), 'casa': partita['HomeTeam'], 'trasferta': partita['AwayTeam'],
            'scelta': scelta, 'prob_modello': probs[scelta], 'quota': quota,
            'p_mercato_implicita_raw': p_imp, 'p_mercato_fair': p_fair,
            'edge_vs_fair': probs[scelta]-p_fair, 'overround_medio': q['overround_medio_bookmaker']-1,
            'ev_raw': ev_raw, 'esito': esito, 'vinta': win, 'profitto': profitto,
            'n_bookmakers': q['n_bookmakers'],
        })
    return pd.DataFrame(rows)


def mostra_audit_quote_v4(dati, campionato, id_fd, rho_val, ewma_span_val, emivita_val):
    st.divider()
    st.write('**🔎 V4 — Audit quote, probabilità di mercato ed EV**')
    st.caption('Questa sezione non modifica il modello. Ricostruisce il prezzo medio storico e separa quota grezza, probabilità implicita, probabilità di mercato normalizzata ed EV teorico.')
    if st.button('🔎 Esegui audit quote / EV', key=f'v4_quote_{id_fd}'):
        with st.spinner('Audit quote/EV in corso...'):
            df1 = esegui_audit_quote_1x2(dati, rho_val, ewma_span_val, emivita_val)
            df2 = esegui_audit_quote_ou(dati, rho_val, ewma_span_val, emivita_val)
        for titolo, df in [('1X2 — selezione più probabile', df1), ('O/U 2.5 — selezione più probabile', df2)]:
            st.markdown(f'### {titolo}')
            if df.empty:
                st.warning('Nessun dato sufficiente per l’audit.')
                continue
            roi = df['profitto'].sum()/len(df)*100
            evm = df['ev_raw'].mean()*100
            edge = df['edge_vs_fair'].mean()*100
            st.write(f'Bet: **{len(df)}** · Strike: **{df["vinta"].mean()*100:.1f}%** · ROI: **{roi:+.1f}%** · EV teorico medio: **{evm:+.1f}%** · Edge medio vs mercato fair: **{edge:+.1f} punti**')
            st.dataframe(_audit_quote_statistiche(df).round(3), use_container_width=True, hide_index=True)
            st.caption('Mercato fair = media delle probabilità implicite normalizzate bookmaker-per-bookmaker. EV teorico usa invece la quota media grezza, cioè il prezzo storico della puntata.')
            st.dataframe(df[['data','casa','trasferta','scelta','prob_modello','quota','p_mercato_implicita_raw','p_mercato_fair','edge_vs_fair','overround_medio','ev_raw','esito','vinta','profitto','n_bookmakers']].round(4), use_container_width=True, hide_index=True)
        if not df1.empty or not df2.empty:
            out = pd.concat([df1.assign(mercato='1X2'), df2.assign(mercato='O/U 2.5')], ignore_index=True)
            st.download_button('⬇️ Scarica audit quote V4 CSV', data=out.to_csv(index=False).encode('utf-8'), file_name=f'audit_quote_v4_{id_fd}.csv', mime='text/csv', key=f'v4_dl_{id_fd}')


# =====================================================================
# 🧪 V2 — CALIBRAZIONE FUORI CAMPIONE
# =====================================================================
def _costruisci_osservazioni_raw(dati_completi, rho, ewma_span, emivita):
    """Calcola una previsione raw 1X2 per ogni partita usando solo il passato.
    Restituisce un DataFrame ordinato temporalmente con stagione e probabilità.
    Nessun calibratore salvato viene letto.
    """
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True).copy()
    righe = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(
            prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
            emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed')
        )
        if m is None:
            continue
        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        righe.append({
            'indice': i,
            'data': partita.get('Date_parsed'),
            'stagione': partita.get('Stagione', ''),
            'p1': float(m['prob_1']) / 100.0,
            'px': float(m['prob_X']) / 100.0,
            'p2': float(m['prob_2']) / 100.0,
            'esito': esito,
        })
    return pd.DataFrame(righe)


def _fit_calibratore_1x2_temporale(df_train):
    """Calibrazione isotonic 1X2 su osservazioni storiche precedenti al test.
    Le tre probabilità sono trattate one-vs-rest, come nel progetto originale,
    e poi rinormalizzate quando applicate.
    """
    if df_train is None or len(df_train) < 100:
        return None
    oss = []
    for _, r in df_train.iterrows():
        oss.append((r['p1'], r['esito'] == '1'))
        oss.append((r['px'], r['esito'] == 'X'))
        oss.append((r['p2'], r['esito'] == '2'))
    if len(oss) < 100:
        return None
    return _allena_isotonic(oss)


def _applica_calibratore_row(r, calibratore):
    if calibratore is None:
        return {'1': r['p1'], 'X': r['px'], '2': r['p2']}
    vals = {
        '1': float(calibratore.predict([r['p1']])[0]),
        'X': float(calibratore.predict([r['px']])[0]),
        '2': float(calibratore.predict([r['p2']])[0]),
    }
    tot = sum(vals.values())
    if tot <= 0:
        return {'1': r['p1'], 'X': r['px'], '2': r['p2']}
    return {k: v / tot for k, v in vals.items()}


def _metriche_da_df(df_eval, calibratore=None):
    if df_eval is None or df_eval.empty:
        return None
    raw_obs = []
    cal_obs = []
    for _, r in df_eval.iterrows():
        raw_obs.append({
            'probs': {'1': r['p1'], 'X': r['px'], '2': r['p2']},
            'esito': r['esito']
        })
        cp = _applica_calibratore_row(r, calibratore)
        cal_obs.append({'probs': cp, 'esito': r['esito']})
    raw = calcola_metriche_1x2(raw_obs)
    cal = calcola_metriche_1x2(cal_obs)
    return raw, cal


def esegui_audit_calibrazione_oos(dati_completi, rho, ewma_span, emivita, modalita='storico'):
    """Confronto raw vs isotonic OOS.

    - corrente: train = stagione precedente, test = stagione corrente.
    - storico: train = primo 70% delle osservazioni temporali, test = ultimo 30%.
    In entrambi i casi il calibratore non vede gli esiti del test.
    """
    df = _costruisci_osservazioni_raw(dati_completi, rho, ewma_span, emivita)
    if df.empty:
        return None

    if modalita == 'corrente' and 'stagione' in df.columns:
        train = df[df['stagione'] == 'precedente'].copy()
        test = df[df['stagione'] == 'corrente'].copy()
        metodo = 'Train = stagione precedente; Test = stagione corrente'
    else:
        cut = max(1, int(len(df) * 0.70))
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()
        metodo = f'Train = primo 70% temporale ({len(train)}), Test = ultimo 30% ({len(test)})'

    calibratore = _fit_calibratore_1x2_temporale(train)
    if calibratore is None:
        return {
            'ok': False,
            'n_raw': len(df),
            'n_train': len(train),
            'n_test': len(test),
            'metodo': metodo,
            'motivo': 'Dati insufficienti per allenare la calibrazione isotonic (servono almeno 100 osservazioni 1X2 nel train).'
        }

    metriche = _metriche_da_df(test, calibratore)
    return {
        'ok': True,
        'n_raw': len(df),
        'n_train': len(train),
        'n_test': len(test),
        'metodo': metodo,
        'raw': metriche[0],
        'cal': metriche[1],
    }


def _tabella_calibrazione_metriche(met):
    righe = []
    for fascia, d in met['bins'].items():
        righe.append({
            'Fascia probabilità massima': fascia + '%',
            'Partite': d['n'],
            'Prob. media prevista': f"{(d['somma_p']/d['n']):.1f}%" if d['n'] else '—',
            'Accuracy reale': f"{(d['positivi']/d['n']*100):.1f}%" if d['n'] else '—',
        })
    return pd.DataFrame(righe)

# =====================================================================
# 🖥️ INTERFACCIA
# =====================================================================
st.title("🧪 COMBO — Audit Model V13.6 Operativa")


# =====================================================================
# 💰 V5 — CALIBRAZIONE OOS APPLICATA ALL'EV
# Questo modulo è separato e non modifica il motore originale.
# Per ogni partita storica:
#   1) calcola la probabilità raw usando SOLO il passato;
#   2) allena la calibrazione isotonic SOLO sulle osservazioni precedenti;
#   3) applica la calibrazione alla partita corrente;
#   4) seleziona la giocata sulla probabilità calibrata;
#   5) ricalcola EV = p_calibrata * quota_storica - 1.
# Il confronto Raw vs Calibrated è effettuato sul MEDESIMO campione
# warm-up OOS, così ROI e profitto sono confrontabili.
# =====================================================================
V5_CAL_MIN_TRAIN = 100


def _v5_fit_iso_binary(p_list, y_list):
    if len(p_list) < V5_CAL_MIN_TRAIN:
        return None
    if len(set(y_list)) < 2:
        return None
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    cal.fit(np.asarray(p_list, dtype=float), np.asarray(y_list, dtype=float))
    return cal


def _v5_raw_history(dati_completi, rho, ewma_span, emivita):
    """Previsioni raw walk-forward per tutto lo storico con quote disponibili."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    ch, cd, ca = classifica_colonne_quote(tutte.columns)
    co, cu = classifica_colonne_over_under(tutte.columns, '2.5')
    righe = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(
            prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
            emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed')
        )
        if m is None:
            continue
        q12 = quote_medie_grezze_1x2(partita, ch, cd, ca)
        qou = quote_medie_grezze_ou(partita, co, cu)
        if q12 is None or qou is None:
            continue
        esito12 = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        tot = partita['FTHG'] + partita['FTAG']
        esito_ou = 'Over' if tot > 2.5 else 'Under'
        p1 = float(m['prob_1']) / 100.0
        px = float(m['prob_X']) / 100.0
        p2 = float(m['prob_2']) / 100.0
        po = (100.0 - float(m['prob_under'][2.5])) / 100.0
        pu = 1.0 - po
        righe.append({
            'indice': i, 'data': partita.get('Date_parsed'),
            'casa': partita['HomeTeam'], 'trasferta': partita['AwayTeam'],
            'p1': p1, 'px': px, 'p2': p2, 'po': po, 'pu': pu,
            'esito12': esito12, 'esito_ou': esito_ou,
            'q1': q12['1'], 'qx': q12['X'], 'q2': q12['2'],
            'qo': qou['Over'], 'qu': qou['Under'],
        })
    return pd.DataFrame(righe)


def _v5_stats(bets):
    if not bets:
        return {'n': 0, 'strike': None, 'roi': None, 'profit': 0.0, 'ev': None,
                'avg_odds': None, 'max_dd': 0.0}
    equity = 0.0; peak = 0.0; max_dd = 0.0
    wins = 0; evs = []; odds = []
    for b in bets:
        if b['vinta']:
            wins += 1
            profit = b['quota'] - 1.0
        else:
            profit = -1.0
        equity += profit
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        evs.append(b['ev']); odds.append(b['quota'])
    n = len(bets)
    return {'n': n, 'strike': wins/n*100.0, 'roi': equity/n*100.0,
            'profit': equity, 'ev': float(np.mean(evs))*100.0,
            'avg_odds': float(np.mean(odds)), 'max_dd': max_dd}


def _v5_bucket_df(df, prob_col, ev_col='ev_cal'):
    if df.empty:
        return pd.DataFrame()
    bins = [0.0, .40, .50, .60, .70, .80, 1.000001]
    labels = ['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%']
    x = df.copy()
    x['fascia'] = pd.cut(x[prob_col], bins=bins, labels=labels, right=False, include_lowest=True)
    rows=[]
    for lab,g in x.groupby('fascia', observed=False):
        if g.empty: continue
        rows.append({
            'Fascia prob. modello': str(lab), 'Bet': len(g),
            'Prob. media': g[prob_col].mean()*100.0,
            'Prob. mercato fair media': g['p_mercato_fair'].mean()*100.0,
            'Frequenza reale': g['vinta'].mean()*100.0,
            'EV teorico medio': g[ev_col].mean()*100.0,
            'ROI': g['profitto'].sum()/len(g)*100.0,
        })
    return pd.DataFrame(rows)


def _v5_run_economic(dati_completi, rho, ewma_span, emivita, soglia_ev=0.10, warmup=100):
    """Raw vs calibrato OOS temporale, su 1X2 e O/U 2.5.
    La calibrazione di ogni riga usa esclusivamente righe precedenti.
    """
    hist = _v5_raw_history(dati_completi, rho, ewma_span, emivita)
    if hist.empty:
        return None
    raw12=[]; cal12=[]; rawou=[]; calou=[]
    prior12_p={'1':[], 'X':[], '2':[]}; prior12_y={'1':[], 'X':[], '2':[]}
    prior_ou_p=[]; prior_ou_y=[]
    for j, r in hist.iterrows():
        if j < warmup:
            # Accumula il train iniziale, ma non usa la riga corrente come propria calibrazione.
            prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
            prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
            prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
            prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)
            continue

        cal12_models = {s: _v5_fit_iso_binary(prior12_p[s], prior12_y[s]) for s in ['1','X','2']}
        cal_ou = _v5_fit_iso_binary(prior_ou_p, prior_ou_y)

        raw_probs={'1':r.p1,'X':r.px,'2':r.p2}
        raw_choice=max(raw_probs, key=raw_probs.get)
        raw_q={'1':r.q1,'X':r.qx,'2':r.q2}[raw_choice]
        raw_ev=raw_probs[raw_choice]*raw_q-1.0
        if raw_ev >= soglia_ev:
            fair={'1':1/r.q1,'X':1/r.qx,'2':1/r.q2}; sm=sum(fair.values())
            fair={k:v/sm for k,v in fair.items()}
            raw12.append({'data':r.data,'casa':r.casa,'trasferta':r.trasferta,'scelta':raw_choice,
                          'prob_modello':raw_probs[raw_choice],'p_mercato_fair':fair[raw_choice],
                          'quota':raw_q,'ev':raw_ev,'vinta':raw_choice==r.esito12})

        if all(cal12_models[s] is not None for s in ['1','X','2']):
            cv={s:float(cal12_models[s].predict([raw_probs[s]])[0]) for s in ['1','X','2']}
            tot=sum(cv.values())
            if tot>0: cv={s:v/tot for s,v in cv.items()}
        else:
            cv=raw_probs
        cal_choice=max(cv, key=cv.get)
        cal_q={'1':r.q1,'X':r.qx,'2':r.q2}[cal_choice]
        cal_ev=cv[cal_choice]*cal_q-1.0
        if cal_ev >= soglia_ev:
            fair={'1':1/r.q1,'X':1/r.qx,'2':1/r.q2}; sm=sum(fair.values())
            fair={k:v/sm for k,v in fair.items()}
            cal12.append({'data':r.data,'casa':r.casa,'trasferta':r.trasferta,'scelta':cal_choice,
                          'prob_modello':cv[cal_choice],'p_mercato_fair':fair[cal_choice],
                          'quota':cal_q,'ev_cal':cal_ev,'vinta':cal_choice==r.esito12})

        # O/U
        raw_ou={'Over':r.po,'Under':r.pu}
        raw_ou_choice=max(raw_ou,key=raw_ou.get)
        raw_ou_q={'Over':r.qo,'Under':r.qu}[raw_ou_choice]
        raw_ou_ev=raw_ou[raw_ou_choice]*raw_ou_q-1.0
        if raw_ou_ev >= soglia_ev:
            fair_o=1/r.qo; fair_u=1/r.qu; sm=fair_o+fair_u
            fair_ou={'Over':fair_o/sm,'Under':fair_u/sm}
            rawou.append({'data':r.data,'casa':r.casa,'trasferta':r.trasferta,'scelta':raw_ou_choice,
                          'prob_modello':raw_ou[raw_ou_choice],'p_mercato_fair':fair_ou[raw_ou_choice],
                          'quota':raw_ou_q,'ev':raw_ou_ev,'vinta':raw_ou_choice==r.esito_ou})

        if cal_ou is not None:
            pco=float(cal_ou.predict([r.po])[0])
            pco=min(max(pco,0.001),0.999)
            cal_ou_probs={'Over':pco,'Under':1-pco}
        else:
            cal_ou_probs=raw_ou
        cal_ou_choice=max(cal_ou_probs,key=cal_ou_probs.get)
        cal_ou_q={'Over':r.qo,'Under':r.qu}[cal_ou_choice]
        cal_ou_ev=cal_ou_probs[cal_ou_choice]*cal_ou_q-1.0
        if cal_ou_ev >= soglia_ev:
            fair_o=1/r.qo; fair_u=1/r.qu; sm=fair_o+fair_u
            fair_ou={'Over':fair_o/sm,'Under':fair_u/sm}
            calou.append({'data':r.data,'casa':r.casa,'trasferta':r.trasferta,'scelta':cal_ou_choice,
                          'prob_modello':cal_ou_probs[cal_ou_choice],'p_mercato_fair':fair_ou[cal_ou_choice],
                          'quota':cal_ou_q,'ev_cal':cal_ou_ev,'vinta':cal_ou_choice==r.esito_ou})

        # La riga corrente entra nel train SOLO dopo essere stata valutata.
        prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
        prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
        prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
        prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)

    def finish(rows, prob_key='prob_modello', ev_key='ev'):
        for b in rows:
            b['profitto'] = b['quota']-1.0 if b['vinta'] else -1.0
            if ev_key=='ev_cal':
                b['ev']=b['ev_cal']
        return rows
    raw12=finish(raw12); cal12=finish(cal12,'prob_modello','ev_cal')
    rawou=finish(rawou); calou=finish(calou,'prob_modello','ev_cal')
    return {'hist':hist,'raw12':pd.DataFrame(raw12),'cal12':pd.DataFrame(cal12),
            'rawou':pd.DataFrame(rawou),'calou':pd.DataFrame(calou)}


def _v5_show_pair(raw_df, cal_df, titolo):
    st.markdown(f"### {titolo}")
    for name,df,evcol in [('RAW',raw_df,'ev'),('CALIBRATO OOS',cal_df,'ev_cal')]:
        if df.empty:
            st.warning(f"{name}: nessuna scommessa qualificata.")
            continue
        # Normalizza colonna EV per la tabella bucket.
        d=df.copy()
        if evcol not in d.columns: d[evcol]=d['ev']
        s=_v5_stats([{'vinta':bool(x.vinta),'quota':float(x.quota),'ev':float(x.ev)} for _,x in d.iterrows()])
        st.write(f"**{name}**")
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric('Bet',s['n']); c2.metric('Strike',f"{s['strike']:.1f}%")
        c3.metric('ROI',f"{s['roi']:+.1f}%"); c4.metric('Profitto',f"{s['profit']:+.2f}u")
        c5.metric('EV medio',f"{s['ev']:+.1f}%")
        st.caption(f"Quota media {s['avg_odds']:.2f} · Max drawdown {s['max_dd']:.2f}u")
        st.dataframe(_v5_bucket_df(d,'prob_modello',evcol),use_container_width=True,hide_index=True)


# =====================================================================
# 🚀 V6 — BATCH TEST AUTOMATICO DEI 5 CAMPIONATI
# Esegue in una sola passata i test principali fatti manualmente sulla
# Serie A: audit grezzo, calibrazione OOS, audit quote/EV e V5 economico.
# V5 economico viene eseguito con EV minimo 10%, storico OOS, come nel test
# di riferimento usato per la Serie A. In più V3 viene riepilogato a 0/5/10/15/20%.
# =====================================================================
def _v6_stats_df(df):
    if df is None or df.empty:
        return {'bet':0,'strike':None,'roi':None,'profit':0.0,'avg_odds':None,'ev':None,'max_dd':None}
    wins = df['vinta'].astype(bool)
    profit = df['quota'].astype(float).where(wins, -1.0)
    cum = profit.cumsum()
    peak = cum.cummax()
    dd = peak-cum
    return {
        'bet': int(len(df)),
        'strike': float(wins.mean()*100),
        'roi': float(profit.mean()*100),
        'profit': float(profit.sum()),
        'avg_odds': float(df['quota'].mean()),
        'ev': float(df['ev'].mean()*100) if 'ev' in df.columns else (float(df['ev_raw'].mean()*100) if 'ev_raw' in df.columns else None),
        'max_dd': float(dd.max()) if len(dd) else 0.0,
    }


def _v6_run_batch(rho, ewma_span, emivita, soglia_v5=0.10, warmup=100):
    risultati=[]
    errori=[]
    campioni=list(CAMPIONATI_DOMESTICI.items())
    for nome, info in campioni:
        try:
            dati_b = carica_dati_campionato(info['id_fd'])
            if dati_b is None or dati_b.empty:
                errori.append((nome, 'dati non disponibili'))
                continue

            # V1/V2: modello grezzo storico OOS
            raw_obs = esegui_audit_1x2(dati_b, rho, ewma_span, emivita, stagione_corrente=False)
            raw_met = calcola_metriche_1x2(raw_obs) if raw_obs else None

            # V2: calibrazione OOS temporale
            cal_res = esegui_audit_calibrazione_oos(dati_b, rho, ewma_span, emivita, modalita='storico')

            # V4: quote/EV storico, 1X2 + O/U
            q12 = esegui_audit_quote_1x2(dati_b, rho, ewma_span, emivita)
            qou = esegui_audit_quote_ou(dati_b, rho, ewma_span, emivita)
            q12s = _v6_stats_df(q12)
            qous = _v6_stats_df(qou)

            # V5: RAW vs CALIBRATO OOS applicato realmente alla selezione EV
            v5 = _v5_run_economic(dati_b, rho, ewma_span, emivita, soglia_v5, warmup)
            v5raw12=_v6_stats_df(v5['raw12']) if v5 else _v6_stats_df(None)
            v5cal12=_v6_stats_df(v5['cal12']) if v5 else _v6_stats_df(None)
            v5rawou=_v6_stats_df(v5['rawou']) if v5 else _v6_stats_df(None)
            v5calou=_v6_stats_df(v5['calou']) if v5 else _v6_stats_df(None)

            # V3: economico RAW a tutte le soglie, storico OOS
            v3=[]
            for evp in [0,5,10,15,20]:
                r1,d1=esegui_backtest_roi_1x2(dati_b,rho,ewma_span,emivita,False,evp/100.0,False)
                r2,d2=esegui_backtest_roi_ou(dati_b,rho,ewma_span,emivita,False,evp/100.0,False)
                v3.append({
                    'campionato':nome,'EV minimo %':evp,
                    '1X2 bet':r1.get('scommesse',0) if r1 else 0,
                    '1X2 strike %':r1.get('strike_rate',None) if r1 else None,
                    '1X2 ROI %':r1.get('roi',None) if r1 else None,
                    '1X2 profit u':r1.get('profitto',None) if r1 else None,
                    'OU bet':r2.get('scommesse',0) if r2 else 0,
                    'OU strike %':r2.get('strike_rate',None) if r2 else None,
                    'OU ROI %':r2.get('roi',None) if r2 else None,
                    'OU profit u':r2.get('profitto',None) if r2 else None,
                })

            risultati.append({
                'campionato':nome,
                'partite_raw_audit': raw_met['n'] if raw_met else 0,
                'raw_accuracy_%': raw_met['accuracy'] if raw_met else None,
                'raw_brier': raw_met['brier'] if raw_met else None,
                'raw_logloss': raw_met['log_loss'] if raw_met else None,
                'cal_train': cal_res.get('n_train') if cal_res else None,
                'cal_test': cal_res.get('n_test') if cal_res else None,
                'cal_accuracy_%': cal_res.get('cal',{}).get('accuracy') if cal_res and cal_res.get('ok') else None,
                'cal_brier': cal_res.get('cal',{}).get('brier') if cal_res and cal_res.get('ok') else None,
                'cal_logloss': cal_res.get('cal',{}).get('log_loss') if cal_res and cal_res.get('ok') else None,
                'V4_1X2_bet':q12s['bet'],'V4_1X2_ROI_%':q12s['roi'],'V4_1X2_EV_%':q12s['ev'],
                'V4_OU_bet':qous['bet'],'V4_OU_ROI_%':qous['roi'],'V4_OU_EV_%':qous['ev'],
                'V5_1X2_RAW_bet':v5raw12['bet'],'V5_1X2_RAW_ROI_%':v5raw12['roi'],'V5_1X2_RAW_profit_u':v5raw12['profit'],
                'V5_1X2_CAL_bet':v5cal12['bet'],'V5_1X2_CAL_ROI_%':v5cal12['roi'],'V5_1X2_CAL_profit_u':v5cal12['profit'],
                'V5_OU_RAW_bet':v5rawou['bet'],'V5_OU_RAW_ROI_%':v5rawou['roi'],'V5_OU_RAW_profit_u':v5rawou['profit'],
                'V5_OU_CAL_bet':v5calou['bet'],'V5_OU_CAL_ROI_%':v5calou['roi'],'V5_OU_CAL_profit_u':v5calou['profit'],
            })
            risultati[-1]['_v3']=v3
        except Exception as e:
            errori.append((nome, f'{type(e).__name__}: {e}'))
    return risultati,errori


def mostra_batch_v6(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🚀 V6 — TEST COMPLETO AUTOMATICO: 5 CAMPIONATI')
    st.caption('Un solo pulsante esegue in sequenza i test principali fatti sulla Serie A: audit grezzo, calibrazione OOS, V4 quote/EV, V3 ROI alle soglie 0/5/10/15/20% e V5 RAW vs CALIBRATO OOS con EV 10%.')
    st.warning('⚠️ È un test massivo: può richiedere diversi minuti. I risultati sono storici OOS e le quote sono proxy storiche, non quote combo eseguibili.')
    if st.button('🚀 ESEGUI TUTTO SUI 5 CAMPIONATI', key='v6_batch_run'):
        prog=st.progress(0.0,text='Avvio test batch...')
        # La funzione stessa esegue i campionati in sequenza; aggiorniamo una barra
        # prima/dopo ogni campionato tramite una versione locale più semplice non è
        # possibile senza duplicare la logica. Manteniamo quindi un indicatore di attività.
        with st.spinner('Analisi completa dei 5 campionati in corso...'):
            res,err=_v6_run_batch(rho,ewma_span,emivita,0.10,100)
        prog.progress(1.0,text='Completato!')
        if res:
            df=pd.DataFrame([{k:v for k,v in x.items() if k!='_v3'} for x in res])
            st.success(f'Completati {len(res)} campionati su {len(CAMPIONATI_DOMESTICI)}.')
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.markdown('### 📊 V3 — ROI RAW per soglia EV')
            v3rows=[]
            for x in res: v3rows.extend(x.get('_v3',[]))
            if v3rows:
                st.dataframe(pd.DataFrame(v3rows),use_container_width=True,hide_index=True)
            st.markdown('### 🔬 V5 — confronto economico RAW vs CALIBRATO OOS (EV 10%)')
            v5cols=['campionato','V5_1X2_RAW_bet','V5_1X2_RAW_ROI_%','V5_1X2_CAL_bet','V5_1X2_CAL_ROI_%','V5_OU_RAW_bet','V5_OU_RAW_ROI_%','V5_OU_CAL_bet','V5_OU_CAL_ROI_%']
            st.dataframe(df[v5cols],use_container_width=True,hide_index=True)
            out=pd.DataFrame(v3rows)
            summary=df.to_csv(index=False).encode('utf-8')
            details=out.to_csv(index=False).encode('utf-8') if not out.empty else b''
            st.download_button('⬇️ Scarica riepilogo completo CSV',data=summary,file_name='V6_batch_5_campionati_summary.csv',mime='text/csv',key='v6_dl_summary')
            if details:
                st.download_button('⬇️ Scarica V3 tutte le soglie CSV',data=details,file_name='V6_batch_5_campionati_V3.csv',mime='text/csv',key='v6_dl_v3')
        if err:
            st.error('Alcuni campionati non sono stati completati:')
            for nome,msg in err: st.write(f'- **{nome}**: {msg}')
        st.info('Confronto da leggere senza classifiche: il batch serve a verificare se gli effetti osservati sulla Serie A si ripetono anche sugli altri campionati. V5 confronta selezioni diverse, quindi RAW e CAL non sono lo stesso identico insieme di scommesse.')


def mostra_sezione_v5_calibrazione_ev(dati, campionato, rho_val, ewma_span_val, emivita_val):
    st.divider()
    st.markdown('## 💰 V5 — Calibrazione OOS realmente applicata all\'EV')
    st.caption('Confronto RAW vs calibrato sullo stesso backtest storico. La calibrazione viene allenata solo sulle partite precedenti a ciascuna partita test; la probabilità calibrata determina selezione ed EV.')
    c1,c2=st.columns(2)
    with c1:
        periodo=st.selectbox('Periodo V5',['Storico OOS'],key='v5_periodo')
    with c2:
        soglia_pct=st.selectbox('EV minimo V5',[0,5,10,15,20],index=2,key='v5_ev')
    if st.button('🧪 Esegui V5: Raw vs Calibrato OOS',key='v5_run'):
        with st.spinner('Calcolo V5 walk-forward + calibrazione OOS...'):
            res=_v5_run_economic(dati,rho_val,ewma_span_val,emivita_val,soglia_pct/100.0,V5_CAL_MIN_TRAIN)
        if res is None:
            st.error('Nessun dato sufficiente.')
            return
        st.success(f"Campione storico raw: {len(res['hist'])} partite con quote disponibili. Warm-up calibrazione: {V5_CAL_MIN_TRAIN} partite.")
        _v5_show_pair(res['raw12'],res['cal12'],'1X2 — EV calcolato sulla probabilità usata per la selezione')
        _v5_show_pair(res['rawou'],res['calou'],'O/U 2.5 — EV calcolato sulla probabilità usata per la selezione')
        st.info('Lettura corretta: se il ROI cambia insieme all\'EV medio dopo la calibrazione, la calibrazione sta modificando davvero il filtro economico. Il confronto è OOS temporale e non usa i calibratori salvati.')
        allrows=[]
        for label,df in [('1X2 RAW',res['raw12']),('1X2 CAL',res['cal12']),('OU RAW',res['rawou']),('OU CAL',res['calou'])]:
            if not df.empty:
                z=df.copy(); z['versione']=label; allrows.append(z)
        if allrows:
            out=pd.concat(allrows,ignore_index=True)
            st.download_button('⬇️ Scarica dettaglio V5 CSV',data=out.to_csv(index=False).encode('utf-8'),file_name='v5_calibrazione_ev.csv',mime='text/csv',key='v5_dl')


st.caption("Versione operativa V13.6 — modello V11 completo + PLATT solo O/U 2.5 con salvaguardia monotona")

st.info("Uso operativo: il modello completo resta invariato; V13 applica PLATT esclusivamente a O/U 2.5 e solo con dati precedenti alla partita selezionata.")

# Parametri del modello fissati ai valori di default validati — non più
# esposti nell'interfaccia: erano controlli tecnici che richiedevano di
# sapere cosa fanno per essere usati bene, e nell'uso normale non si toccano.
rho_val = -0.10        # correzione Dixon-Coles (valore tipico da letteratura)
ewma_span_val = 6      # finestra della forma recente
emivita_val = 180      # decadimento temporale in giorni

with st.sidebar:
    st.header("⚙️ Configurazione & API")
    api_key_input = st.text_input("Chiave API football-data.org (per Coppe)", type="password",
                                   help="Gratuita: football-data.org/client/register")
    st.divider()
    usa_calibrazione = st.checkbox(
        "🎯 Applica calibrazione 1X2 (se allenata)",
        value=False,
        help="Calibrazione separata del mercato 1X2. Disattivata di default."
    )
    usa_platt_v13 = st.checkbox(
        "🎯 V13 — PLATT solo O/U 2.5",
        value=True,
        help="Corregge esclusivamente la probabilità Over/Under 2.5 usando un fit walk-forward sulle partite precedenti."
    )
    mostra_diagnostica_v11 = st.checkbox(
        "🔬 Mostra diagnostica V11",
        value=False,
        help="Mostra la diagnostica tecnica completa. Lascia disattivata per l'uso operativo."
    )

scelta_categoria = st.radio("Categoria Torneo", ["Campionati Nazionali (Gratuiti)", "Coppe Europee (Richiede API Key)"], horizontal=True)

if scelta_categoria == "Campionati Nazionali (Gratuiti)":
    campionato = st.selectbox("Seleziona Campionato", list(CAMPIONATI_DOMESTICI.keys()))
    info = CAMPIONATI_DOMESTICI[campionato]
    id_fd = info["id_fd"]

    with st.spinner("Caricamento dataset campionato e quote automatiche..."):
        dati = carica_dati_campionato(id_fd)
        fixture_future = carica_fixture_future(id_fd)

    if dati is None or len(dati) == 0:
        st.error("Impossibile scaricare i dati. Riprova tra poco.")
        st.stop()

    opzioni_partite, mappa_partite = [], []
    if not fixture_future.empty:
        for _, r in fixture_future.iterrows():
            opzioni_partite.append(f"FUTURA ({r.get('Date','?')}): {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
            mappa_partite.append(r.to_dict())

    storiche = dati[dati['FTHG'].notna()].tail(15)
    for _, r in storiche.iterrows():
        opzioni_partite.append(f"RECENTE ({r.get('Date','?')}): {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
        mappa_partite.append(r.to_dict())
    df_globale = pd.DataFrame()
    is_coppa = False

    with st.expander("🧪 AUDIT RIGOROSO — modello grezzo", expanded=False):
        st.write("**Obiettivo:** capire se le probabilità del modello sono affidabili prima di usare calibrazione e value bet.")
        col_a, col_b = st.columns(2)
        with col_a:
            run_curr = st.button("🧪 Audit stagione corrente", key="audit_current")
        with col_b:
            run_all = st.button("🧪 Audit tutto lo storico", key="audit_all")

        if run_curr or run_all:
            with st.spinner("Esecuzione audit partita per partita..."):
                obs = esegui_audit_1x2(dati, rho_val, ewma_span_val, emivita_val, stagione_corrente=run_curr)
            met = calcola_metriche_1x2(obs)
            if met is None:
                st.warning("Nessuna osservazione disponibile.")
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Partite", met["n"])
                c2.metric("Accuracy", f"{met['accuracy']:.1f}%")
                c3.metric("Brier", f"{met['brier']:.4f}")
                c4.metric("Log Loss", f"{met['log_loss']:.4f}")
                righe=[]
                for fascia,d in met["bins"].items():
                    righe.append({
                        "Fascia probabilità massima": fascia+"%",
                        "Partite": d["n"],
                        "Probabilità media prevista": f"{(d['somma_p']/d['n']):.1f}%" if d["n"] else "—",
                        "Accuracy reale": f"{(d['positivi']/d['n']*100):.1f}%" if d["n"] else "—",
                    })
                st.write("**Calibrazione delle previsioni principali**")
                st.table(pd.DataFrame(righe))
                st.caption("Una previsione al 70% dovrebbe, su un campione sufficientemente grande, verificarsi circa 70% delle volte. La tabella serve a controllare questa proprietà; non è una garanzia su piccoli campioni.")

    with st.expander("🧪 V2 — Confronto calibrazione fuori campione", expanded=False):
        st.write("**Obiettivo:** verificare se una calibrazione isotonic migliora le probabilità senza vedere gli esiti del periodo di test.")
        st.caption("La v2 non legge i calibratori salvati: li allena temporalmente e li applica solo al blocco successivo.")
        col_v2a, col_v2b = st.columns(2)
        with col_v2a:
            run_v2_curr = st.button("🧪 V2 corrente: calibrazione OOS", key="audit_v2_current")
        with col_v2b:
            run_v2_all = st.button("🧪 V2 storico: calibrazione OOS", key="audit_v2_all")

        if run_v2_curr or run_v2_all:
            modalita_v2 = 'corrente' if run_v2_curr else 'storico'
            with st.spinner("Calcolo previsioni walk-forward e calibrazione fuori campione..."):
                res_v2 = esegui_audit_calibrazione_oos(
                    dati, rho_val, ewma_span_val, emivita_val, modalita=modalita_v2
                )
            if res_v2 is None:
                st.warning("Nessuna osservazione disponibile.")
            elif not res_v2['ok']:
                st.warning(res_v2['motivo'])
                st.write(f"Train: **{res_v2['n_train']}** partite · Test: **{res_v2['n_test']}** partite")
                st.caption(res_v2['metodo'])
            else:
                st.success("✅ Test OOS valido: il calibratore non ha visto gli esiti del blocco di test.")
                st.caption(res_v2['metodo'])
                st.write(f"Dataset audit: **{res_v2['n_raw']}** partite · Train: **{res_v2['n_train']}** · Test: **{res_v2['n_test']}**")

                st.markdown("### Modello grezzo vs calibrato")
                tab_v2 = pd.DataFrame([
                    {
                        'Versione': 'Grezzo',
                        'Accuracy': f"{res_v2['raw']['accuracy']:.1f}%",
                        'Brier ↓': f"{res_v2['raw']['brier']:.4f}",
                        'Log Loss ↓': f"{res_v2['raw']['log_loss']:.4f}",
                    },
                    {
                        'Versione': 'Calibrato OOS',
                        'Accuracy': f"{res_v2['cal']['accuracy']:.1f}%",
                        'Brier ↓': f"{res_v2['cal']['brier']:.4f}",
                        'Log Loss ↓': f"{res_v2['cal']['log_loss']:.4f}",
                    },
                ])
                st.table(tab_v2)

                delta_brier = res_v2['cal']['brier'] - res_v2['raw']['brier']
                delta_ll = res_v2['cal']['log_loss'] - res_v2['raw']['log_loss']
                delta_acc = res_v2['cal']['accuracy'] - res_v2['raw']['accuracy']
                st.write(
                    f"**Delta calibrato − grezzo:** Accuracy {delta_acc:+.1f} punti · "
                    f"Brier {delta_brier:+.4f} · Log Loss {delta_ll:+.4f}"
                )

                st.markdown("### Calibrazione sul test — grezzo")
                st.table(_tabella_calibrazione_metriche(res_v2['raw']))
                st.markdown("### Calibrazione sul test — calibrato OOS")
                st.table(_tabella_calibrazione_metriche(res_v2['cal']))
                st.caption("Per Brier e Log Loss, più basso è migliore. La v2 non decide da sola se la calibrazione è utile: la confrontiamo numericamente sul test fuori campione.")

    with st.expander("📈 Verifica storica (backtest)"):

        st.caption("Quante volte la previsione principale del modello (il segno più probabile) "
                   "ha indovinato il risultato vero — su TUTTE le partite disponibili, senza "
                   "nessun filtro. Confronta stagione corrente e storico completo per vedere "
                   "se il modello regge o se il risultato dipende dal periodo.")

        richiedi_accordo = st.checkbox(
            "💡 Idea #1: valuta solo quando modello e mercato sono d'accordo sul favorito",
            value=False,
            help="Filtro di fiducia, non una correzione della stima: scarta le partite dove il "
                 "modello si discosta dal favorito del mercato. Riduce il campione."
        )

        col_bt_a, col_bt_b = st.columns(2)
        with col_bt_a:
            cliccato_corrente = st.button("🎯 Backtest — solo stagione corrente")
        with col_bt_b:
            cliccato_tutto = st.button("📚 Backtest — tutto lo storico")

        def mostra_risultato_backtest(risultato, etichetta):
            if risultato is None:
                st.warning(f"⚠️ {etichetta}: nessuna partita disponibile per questa modalità.")
                return
            if risultato["n_partite"] == 0:
                st.warning(f"⚠️ {etichetta}: nessuna partita valutabile (o tutte scartate dal filtro).")
                return
            win_rate_tot = risultato["n_corrette"] / risultato["n_partite"] * 100
            st.write(f"**{etichetta}**")
            c1, c2 = st.columns(2)
            c1.metric("Partite valutate", risultato["n_partite"])
            c2.metric("Accuratezza", f"{win_rate_tot:.1f}%")
            if risultato.get("n_scartate_disaccordo", 0) > 0 or risultato.get("n_scartate_no_quote", 0) > 0:
                st.caption(f"Scartate per disaccordo modello/mercato: {risultato['n_scartate_disaccordo']} — "
                          f"scartate per quote mancanti: {risultato['n_scartate_no_quote']}.")
            righe_segno = []
            for s, d in risultato["per_segno"].items():
                wr_s = (d["corrette"]/d["previste"]*100) if d["previste"] > 0 else None
                righe_segno.append({
                    "Segno previsto": s, "Volte previsto": d["previste"],
                    "Corrette": d["corrette"],
                    "Accuratezza": f"{wr_s:.1f}%" if wr_s is not None else "—",
                })
            st.table(pd.DataFrame(righe_segno))

        if cliccato_corrente:
            with st.spinner("Backtest su stagione corrente..."):
                ris = esegui_backtest_senza_filtro(dati, rho_val, ewma_span_val, emivita_val,
                                                    True, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                                    richiedi_accordo_mercato=richiedi_accordo)
            mostra_risultato_backtest(ris, "Solo stagione corrente (out-of-sample)")

        if cliccato_tutto:
            with st.spinner("Backtest su tutto lo storico (può richiedere più tempo)..."):
                ris = esegui_backtest_senza_filtro(dati, rho_val, ewma_span_val, emivita_val,
                                                    False, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                                    richiedi_accordo_mercato=richiedi_accordo)
            mostra_risultato_backtest(ris, "Tutto lo storico disponibile")

        if cliccato_corrente or cliccato_tutto:
            st.caption("Se il modello prevede quasi sempre '1' e quasi mai 'X', è normale: il pareggio "
                       "è statisticamente l'esito più difficile da prevedere, per qualunque modello. "
                       "Un'accuratezza intorno al 45-55% su 1X2 è nella norma per modelli di questo tipo — "
                       "il mercato stesso, con molte più informazioni, non fa enormemente meglio.")

        st.divider()
        st.write("**🔥 Backtest COMBO — accuratezza della combo più probabile**")
        st.caption("Stessa logica del backtest 1X2, applicata alla combo automatica con probabilità "
                   "più alta tra le 12 possibili (usa la calibrazione per-combo se allenata e attiva). "
                   "Più difficile del semplice 1X2 per costruzione: una combo richiede che TRE "
                   "condizioni si avverino insieme, non una sola.")
        st.caption("Il checkbox 'Idea #1' qui sopra filtra solo sul segno. Quello qui sotto (idea #1b, "
                   "da verificare) filtra ANCHE sull'Over/Under — copre 2 delle 3 dimensioni di una combo "
                   "invece di 1 sola. Prova le combinazioni: nessuno dei due, solo #1, solo #1b, entrambi.")
        richiedi_accordo_ou = st.checkbox(
            "💡 Idea #1b: richiedi accordo anche su Over/Under 2.5 col mercato",
            value=False,
        )

        col_cb_a, col_cb_b = st.columns(2)
        with col_cb_a:
            cliccato_combo_corrente = st.button("🎯 Backtest combo — solo stagione corrente")
        with col_cb_b:
            cliccato_combo_tutto = st.button("📚 Backtest combo — tutto lo storico")

        def mostra_risultato_backtest_combo(risultato, etichetta):
            if risultato is None:
                st.warning(f"⚠️ {etichetta}: nessuna partita disponibile per questa modalità.")
                return
            if risultato["n_partite"] == 0:
                st.warning(f"⚠️ {etichetta}: nessuna partita valutabile (o tutte scartate dai filtri).")
                return
            win_rate_combo = risultato["n_corrette"] / risultato["n_partite"] * 100
            st.write(f"**{etichetta}**")
            c1, c2 = st.columns(2)
            c1.metric("Partite valutate", risultato["n_partite"])
            c2.metric("Accuratezza combo", f"{win_rate_combo:.1f}%")
            if (risultato.get("n_scartate_disaccordo", 0) > 0 or risultato.get("n_scartate_disaccordo_ou", 0) > 0
                    or risultato.get("n_scartate_no_quote", 0) > 0):
                st.caption(f"Scartate per disaccordo sul segno: {risultato.get('n_scartate_disaccordo', 0)} — "
                          f"scartate per disaccordo su Over/Under: {risultato.get('n_scartate_disaccordo_ou', 0)} — "
                          f"scartate per quote mancanti: {risultato.get('n_scartate_no_quote', 0)}.")
            righe_combo_bt = []
            for chiave, d in sorted(risultato["per_combo"].items(), key=lambda x: x[1]["previste"], reverse=True):
                if d["previste"] == 0: continue
                wr_c = d["corrette"]/d["previste"]*100
                righe_combo_bt.append({
                    "Combo prevista": chiave.replace("_", " "), "Volte prevista": d["previste"],
                    "Corrette": d["corrette"], "Accuratezza": f"{wr_c:.1f}%",
                })
            if righe_combo_bt:
                st.table(pd.DataFrame(righe_combo_bt))

        if cliccato_combo_corrente:
            with st.spinner("Backtest combo su stagione corrente..."):
                ris_c = esegui_backtest_combo(dati, rho_val, ewma_span_val, emivita_val,
                                              True, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                              richiedi_accordo_mercato=richiedi_accordo,
                                              richiedi_accordo_ou=richiedi_accordo_ou)
            mostra_risultato_backtest_combo(ris_c, "Solo stagione corrente (out-of-sample)")

        if cliccato_combo_tutto:
            with st.spinner("Backtest combo su tutto lo storico (può richiedere più tempo)..."):
                ris_c = esegui_backtest_combo(dati, rho_val, ewma_span_val, emivita_val,
                                              False, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                              richiedi_accordo_mercato=richiedi_accordo,
                                              richiedi_accordo_ou=richiedi_accordo_ou)
            mostra_risultato_backtest_combo(ris_c, "Tutto lo storico disponibile")

        if cliccato_combo_corrente or cliccato_combo_tutto:
            st.caption("Confronta questa accuratezza con quella 1X2 qui sopra: se è molto più bassa, "
                       "è normale (tre condizioni insieme sono più difficili), non necessariamente "
                       "un problema — ma dà la misura reale di quanto ci si può fidare delle combo.")

        st.divider()
        st.write("**⚡ Test completo automatico (le 8 combinazioni insieme)**")
        st.caption("Invece di lanciare i due bottoni sopra 4 volte a mano (con/senza ciascun filtro) e "
                   "copiare ogni tabella, questo le fa tutte in un colpo solo e ti dà un file da "
                   "scaricare e mandarmi direttamente — niente più copia-incolla su Word.")

        if st.button("⚡ Esegui tutte le 8 combinazioni per questo campionato"):
            righe_test_completo = []
            combinazioni = [
                ("Solo stagione corrente", True, False, False),
                ("Solo stagione corrente", True, True, False),
                ("Solo stagione corrente", True, False, True),
                ("Solo stagione corrente", True, True, True),
                ("Tutto lo storico", False, False, False),
                ("Tutto lo storico", False, True, False),
                ("Tutto lo storico", False, False, True),
                ("Tutto lo storico", False, True, True),
            ]
            barra_avanzamento = st.progress(0.0, text="Avvio...")
            for idx, (etichetta_periodo, oos, filtro_segno, filtro_ou) in enumerate(combinazioni):
                barra_avanzamento.progress((idx)/8, text=f"{idx+1}/8 — {etichetta_periodo}, "
                                           f"segno={'sì' if filtro_segno else 'no'}, O/U={'sì' if filtro_ou else 'no'}...")
                ris = esegui_backtest_combo(dati, rho_val, ewma_span_val, emivita_val, oos, id_fd=id_fd,
                                            usa_calibrazione=usa_calibrazione,
                                            richiedi_accordo_mercato=filtro_segno, richiedi_accordo_ou=filtro_ou)
                if ris is None or ris["n_partite"] == 0:
                    righe_test_completo.append({
                        "Campionato": campionato, "Periodo": etichetta_periodo,
                        "Filtro segno": "sì" if filtro_segno else "no", "Filtro O/U": "sì" if filtro_ou else "no",
                        "Partite valutate": 0, "Corrette": 0, "Accuratezza %": None,
                    })
                else:
                    righe_test_completo.append({
                        "Campionato": campionato, "Periodo": etichetta_periodo,
                        "Filtro segno": "sì" if filtro_segno else "no", "Filtro O/U": "sì" if filtro_ou else "no",
                        "Partite valutate": ris["n_partite"], "Corrette": ris["n_corrette"],
                        "Accuratezza %": round(ris["n_corrette"]/ris["n_partite"]*100, 1),
                    })
            barra_avanzamento.progress(1.0, text="Completato!")

            df_test_completo = pd.DataFrame(righe_test_completo)
            st.dataframe(df_test_completo, use_container_width=True, hide_index=True)

            csv_bytes = df_test_completo.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Scarica questi risultati come CSV",
                data=csv_bytes,
                file_name=f"backtest_combo_{id_fd}.csv",
                mime="text/csv",
            )
            st.caption("Scarica il CSV per ogni campionato che testi, poi mandameli tutti insieme — "
                       "posso leggerli direttamente, non serve più copiarli su Word.")


        mostra_sezione_roi_v3(dati, campionato, id_fd, rho_val, ewma_span_val, emivita_val)
        mostra_audit_quote_v4(dati, campionato, id_fd, rho_val, ewma_span_val, emivita_val)
        mostra_batch_v6(rho_val, ewma_span_val, emivita_val)

        mostra_sezione_v5_calibrazione_ev(dati, campionato, rho_val, ewma_span_val, emivita_val)

        st.divider()
        st.write("**🎯 Calibrazione (1X2 + le 12 combo automatiche)**")
        st.caption("Allena i correttori su tutto lo storico disponibile. Una sola passata sui dati "
                   "calcola il modello una volta per partita e allena tutti i 13 correttori insieme "
                   "(1X2 + le 12 combinazioni segno × Gol/No Gol × Over/Under 2.5).")
        if st.button("🎯 Allena tutte le calibrazioni per questo campionato"):
            with st.spinner("Allenamento in corso (può richiedere qualche secondo)..."):
                risultati_calib = allena_tutte_le_calibrazioni(dati, rho_val, ewma_span_val, emivita_val, id_fd)
            ok_1x2 = risultati_calib["1x2"]["ok"]
            n_combo_ok = sum(1 for k, v in risultati_calib.items() if k != "1x2" and v["ok"])
            if ok_1x2:
                st.success(f"✅ 1X2: calibrato su {risultati_calib['1x2']['n']} osservazioni.")
            else:
                st.warning(f"⚠️ 1X2: campione insufficiente ({risultati_calib['1x2']['n']} osservazioni, "
                          f"ne servono almeno 100).")
            st.write(f"**Combo calibrate con successo: {n_combo_ok} su 12.**")
            righe_combo_calib = []
            for chiave, v in risultati_calib.items():
                if chiave == "1x2": continue
                righe_combo_calib.append({"Combo": chiave.replace("_", " "), "Osservazioni": v["n"],
                                          "Calibrata": "✅" if v["ok"] else "⚠️ campione scarso"})
            st.dataframe(pd.DataFrame(righe_combo_calib), use_container_width=True, hide_index=True)
            st.caption("Le combo più rare (es. 'X + NoGoal + Over') hanno naturalmente meno osservazioni "
                       "delle più comuni — è normale, non un errore.")

else:
    if not api_key_input:
        st.warning("⚠️ Inserisci la tua chiave API di football-data.org nella barra laterale per sbloccare le coppe europee.")
        st.stop()
    campionato = st.selectbox("Seleziona Coppe", list(CAMPIONATI_COPPE.keys()))
    info = CAMPIONATI_COPPE[campionato]
    code_api = info["code"]
    if not info.get("gratis_confermato", False):
        st.info("ℹ️ Questa competizione non risultava nell'elenco confermato del piano gratuito "
                "di football-data.org — se ricevi 'Accesso Negato' è un limite del piano, non un bug.")

    with st.spinner("Connessione alle API delle Coppe e caricamento dati di supporto..."):
        risultato_api = carica_dati_api_europee(code_api, api_key_input)
        df_globale = carica_tutti_i_campionati()

    if isinstance(risultato_api, str) and risultato_api == "ERRORE_403":
        st.error("❌ **Accesso Negato (Errore 403)**: la tua chiave API gratuita non ha accesso a questa competizione.")
        st.stop()
    elif isinstance(risultato_api, str):
        st.error(f"Errore di connessione API: {risultato_api}")
        st.stop()

    dati = risultato_api
    if dati is None or len(dati) == 0:
        st.error("Nessun dato trovato per questa competizione.")
        st.stop()

    dati_storico = dati[dati['Status'] == 'FINISHED'].copy()
    dati_future = dati[dati['Status'] != 'FINISHED'].copy()

    opzioni_partite, mappa_partite = [], []
    for _, r in dati_future.iterrows():
        opzioni_partite.append(f"FUTURA ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
        mappa_partite.append(r.to_dict())
    for _, r in dati_storico.tail(10).iterrows():
        opzioni_partite.append(f"GIOCATA ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
        mappa_partite.append(r.to_dict())
    is_coppa = True

if not opzioni_partite:
    st.warning("Nessuna partita disponibile al momento.")
else:
    scelta = st.selectbox("Seleziona Partita", opzioni_partite)
    idx_sel = opzioni_partite.index(scelta)
    partita_sel = mappa_partite[idx_sel]

    # =====================================================================
    # 🔧 FIX #2 — FILTRO NO-LOOK-AHEAD
    # Prima: il modello riceveva TUTTO il dataset, comprese le giornate
    # successive alla partita selezionata (per le partite "GIOCATA"/
    # "RECENTE" già nel passato) — calcolava quindi le statistiche anche
    # con risultati che, a quella data, non erano ancora accaduti.
    # Ora: filtriamo sempre a "solo partite precedenti alla data selezionata".
    # =====================================================================
    data_rif_sel = partita_sel.get('Date_parsed')
    if pd.notna(data_rif_sel):
        dati_filtrati = dati[(dati['FTHG'].notna()) & (dati['Date_parsed'] < data_rif_sel)].copy() if 'Date_parsed' in dati.columns else dati
        df_globale_filtrato = df_globale[(df_globale['FTHG'].notna()) & (df_globale['Date_parsed'] < data_rif_sel)].copy() \
            if (df_globale is not None and not df_globale.empty and 'Date_parsed' in df_globale.columns) else df_globale
    else:
        dati_filtrati = dati[dati['FTHG'].notna()].copy() if 'FTHG' in dati.columns else dati
        df_globale_filtrato = df_globale

    modello = calcola_modello_completo(dati_filtrati, partita_sel['HomeTeam'], partita_sel['AwayTeam'],
                                        rho_val, ewma_span_val, emivita_val, df_globale_filtrato,
                                        data_riferimento=data_rif_sel if pd.notna(data_rif_sel) else None)

    calib_1x2_info = None
    if modello is not None and not is_coppa and usa_calibrazione:
        calib_1x2_info = carica_calibratore(id_fd, "1x2")
        if calib_1x2_info:
            modello = applica_calibrazione_1x2(modello, calib_1x2_info["calibratore"])

    platt_v13_info = None
    if modello is not None and not is_coppa and usa_platt_v13:
        try:
            train_platt = _v13_training_ou_platt(
                dati, rho_val, ewma_span_val, emivita_val,
                data_rif_sel if pd.notna(data_rif_sel) else None,
                n_train=100
            )
            platt_v13, platt_coef, platt_status = _v13_fit_platt_ou(train_platt)
            p_raw = 1.0 - float(modello['prob_under'][2.5]) / 100.0
            if platt_v13 is not None:
                p_new = _v13_predict_platt_ou(p_raw, platt_v13)
                modello = _v13_apply_platt_ou(modello, platt_v13)
                platt_v13_info = {
                    'stato': 'applicato',
                    'n_train': len(train_platt),
                    'coef': platt_coef,
                    'raw_over': p_raw * 100.0,
                    'platt_over': p_new * 100.0,
                    'raw_under': (1.0 - p_raw) * 100.0,
                    'platt_under': (1.0 - p_new) * 100.0,
                }
            else:
                platt_v13_info = {
                    'stato': 'escluso',
                    'n_train': len(train_platt),
                    'coef': platt_coef,
                    'raw_over': p_raw * 100.0,
                    'raw_under': (1.0 - p_raw) * 100.0,
                    'motivo': platt_status,
                }
        except Exception as _platt_err:
            platt_v13_info = {'errore': f'{type(_platt_err).__name__}: {_platt_err}'}

    if modello is None:
        st.warning(f"⚠️ Impossibile elaborare il match per **{partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}**.")
    else:
        st.subheader(f"📊 Analisi Match: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        if calib_1x2_info:
            st.caption(f"🎯 Probabilità 1X2 corrette con calibrazione (allenata su "
                       f"{calib_1x2_info['n_osservazioni']} osservazioni, {calib_1x2_info['timestamp']}).")
        if platt_v13_info and platt_v13_info.get('stato') == 'applicato':
            st.info(
                f"🎯 **V13 PLATT O/U 2.5 applicato** — training walk-forward: {platt_v13_info['n_train']} partite · "
                f"pendenza: {platt_v13_info['coef']:.4f}. "
                f"Over 2.5: {platt_v13_info['raw_over']:.1f}% → {platt_v13_info['platt_over']:.1f}% · "
                f"Under 2.5: {platt_v13_info['raw_under']:.1f}% → {platt_v13_info['platt_under']:.1f}%."
            )
        elif platt_v13_info and platt_v13_info.get('stato') == 'escluso':
            st.info(
                f"ℹ️ **V13 PLATT O/U 2.5 escluso automaticamente** — training: {platt_v13_info['n_train']} partite · "
                f"pendenza: {platt_v13_info['coef']:.4f}. "
                f"Il fit non e' monotono; resta la probabilita RAW: "
                f"Over 2.5 {platt_v13_info['raw_over']:.1f}% · Under 2.5 {platt_v13_info['raw_under']:.1f}%."
            )
        elif platt_v13_info and 'errore' in platt_v13_info:
            st.warning(f"⚠️ PLATT O/U 2.5 non applicato per errore tecnico: {platt_v13_info['errore']}")

        # Avviso trasparenza dati (sostituisce il vecchio generatore silenzioso di dati finti)
        SOGLIA_AVVISO = 5
        avvisi = []
        if modello["n_casa"] < SOGLIA_AVVISO:
            avvisi.append(f"{partita_sel['HomeTeam']} (solo {modello['n_casa']} partite trovate)")
        if modello["n_trasf"] < SOGLIA_AVVISO:
            avvisi.append(f"{partita_sel['AwayTeam']} (solo {modello['n_trasf']} partite trovate)")
        if avvisi:
            st.warning("⚠️ Stima poco affidabile per: " + " e ".join(avvisi) +
                       " — il modello converge verso la media di lega/competizione per compensare "
                       "la scarsità di dati specifici, ma la previsione resta meno precisa del solito.")

        def crea_tabella(dati_dict, col_nome="Mercato"):
            df = pd.DataFrame(list(dati_dict.items()), columns=[col_nome, "Probabilità (%)"])
            df["Probabilità (%)"] = df["Probabilità (%)"].round(1)
            df = df.sort_values(by="Probabilità (%)", ascending=False).reset_index(drop=True)
            df["Probabilità (%)"] = df["Probabilità (%)"].astype(str) + "%"
            return df

        st.markdown("### 🏆 Esito Finale (1X2)")
        df_1x2 = crea_tabella({"1 (Casa)": modello['prob_1'], "X (Pareggio)": modello['prob_X'], "2 (Trasferta)": modello['prob_2']}, "Segno")
        st.dataframe(df_1x2, use_container_width=True, hide_index=True)

        quote, quote_ou_25 = None, None  # usate più sotto per la stima combo approssimata (Punto C)

        if not is_coppa:
            # =====================================================================
            # ✅ INDICATORE DI CONCORDANZA MODELLO/MERCATO — come scegliere le partite
            # Il backtest ha confermato: quando la previsione principale del modello
            # coincide col favorito del mercato, l'accuratezza sale (idea #1: segno,
            # +5/+10 punti; idea #1b: anche Over/Under 2.5, ulteriore miglioramento
            # in La Liga, Bundesliga, Ligue 1 — più debole ma comunque presente su
            # Serie A e Premier League). Due badge separati, stesso criterio già
            # validato nel backtest, applicato partita per partita.
            # =====================================================================
            colonne_h_pre, colonne_d_pre, colonne_a_pre = classifica_colonne_quote(dati.columns)
            quote_pre = quote_mercato_normalizzate(partita_sel, colonne_h_pre, colonne_d_pre, colonne_a_pre)
            if quote_pre:
                probabilita_1x2 = {"1": modello['prob_1'], "X": modello['prob_X'], "2": modello['prob_2']}
                previsione_modello = max(probabilita_1x2, key=probabilita_1x2.get)
                quote_per_segno_pre = {"1": quote_pre["q_casa_equa"], "X": quote_pre["q_x_equa"], "2": quote_pre["q_trasf_equa"]}
                favorito_mercato_pre = min(quote_per_segno_pre, key=quote_per_segno_pre.get)
                if previsione_modello == favorito_mercato_pre:
                    st.success(f"✅ **Segno — modello e mercato d'accordo** (entrambi favoriscono '{previsione_modello}') — "
                              f"nel backtest, su questo tipo di partite l'accuratezza è stata sensibilmente più alta.")
                else:
                    st.warning(f"⚠️ **Segno — disaccordo**: il modello preferisce '{previsione_modello}', il mercato "
                              f"favorisce '{favorito_mercato_pre}' — nel backtest, questo tipo di partite ha "
                              f"un'accuratezza più bassa. Trattala con più cautela.")

            colonne_over_pre, colonne_under_pre = classifica_colonne_over_under(dati.columns, "2.5")
            quote_ou_pre = quote_over_under_normalizzate(partita_sel, colonne_over_pre, colonne_under_pre)
            if quote_ou_pre:
                p_under_modello = modello['prob_under'][2.5]
                previsione_ou_modello = "Under" if p_under_modello >= 50 else "Over"
                favorito_ou_mercato_pre = "Over" if quote_ou_pre["q_over_equa"] < quote_ou_pre["q_under_equa"] else "Under"
                if previsione_ou_modello == favorito_ou_mercato_pre:
                    st.success(f"✅ **Over/Under 2.5 — modello e mercato d'accordo** (entrambi favoriscono "
                              f"'{previsione_ou_modello}') — segnale aggiuntivo utile soprattutto per le combo.")
                else:
                    st.warning(f"⚠️ **Over/Under 2.5 — disaccordo**: il modello preferisce '{previsione_ou_modello}', "
                              f"il mercato favorisce '{favorito_ou_mercato_pre}'.")

        if not is_coppa:
            st.markdown("### 💰 Controllo Value Bet (media multi-bookmaker, quote depurate dal margine)")
            colonne_h, colonne_d, colonne_a = classifica_colonne_quote(dati.columns)
            quote = quote_mercato_normalizzate(partita_sel, colonne_h, colonne_d, colonne_a)
            colonne_over, colonne_under = classifica_colonne_over_under(dati.columns, "2.5")
            quote_ou_25 = quote_over_under_normalizzate(partita_sel, colonne_over, colonne_under)
            if quote:
                ev_1 = (modello['prob_1'] / 100.0) * quote['q_casa_equa']
                ev_x = (modello['prob_X'] / 100.0) * quote['q_x_equa']
                ev_2 = (modello['prob_2'] / 100.0) * quote['q_trasf_equa']
                dati_ev = [
                    {"Segno": "1 (Casa)", "Quota Equa": f"{quote['q_casa_equa']:.2f}", "Valutazione": "🔥 ALTO VALORE" if ev_1 > 1.05 else ("📈 Leggero Valore" if ev_1 > 1.0 else "Nessun Valore")},
                    {"Segno": "X (Pareggio)", "Quota Equa": f"{quote['q_x_equa']:.2f}", "Valutazione": "🔥 ALTO VALORE" if ev_x > 1.05 else ("📈 Leggero Valore" if ev_x > 1.0 else "Nessun Valore")},
                    {"Segno": "2 (Trasferta)", "Quota Equa": f"{quote['q_trasf_equa']:.2f}", "Valutazione": "🔥 ALTO VALORE" if ev_2 > 1.05 else ("📈 Leggero Valore" if ev_2 > 1.0 else "Nessun Valore")},
                ]
                st.caption(f"Media su {quote['n_bookmakers']} bookmaker, margine rimosso: {(quote['overround']-1)*100:.1f}%")
                st.dataframe(pd.DataFrame(dati_ev), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Quote dei bookmaker non disponibili per questa specifica partita.")
        else:
            st.markdown("### 🎯 Quota Equa Statistica (Coppe Europee)")
            st.caption("Nessuna quota di mercato disponibile per le coppe: il modello mostra la quota equa "
                       "basata sulla probabilità pura. Cerca sui bookmaker quote superiori a questi valori.")
            q_fair_1 = 100.0 / modello['prob_1'] if modello['prob_1'] > 0 else 0.0
            q_fair_x = 100.0 / modello['prob_X'] if modello['prob_X'] > 0 else 0.0
            q_fair_2 = 100.0 / modello['prob_2'] if modello['prob_2'] > 0 else 0.0
            dati_fair = [
                {"Segno": "1 (Casa)", "Probabilità": f"{modello['prob_1']:.1f}%", "Quota Equa Minima": f"{q_fair_1:.2f}"},
                {"Segno": "X (Pareggio)", "Probabilità": f"{modello['prob_X']:.1f}%", "Quota Equa Minima": f"{q_fair_x:.2f}"},
                {"Segno": "2 (Trasferta)", "Probabilità": f"{modello['prob_2']:.1f}%", "Quota Equa Minima": f"{q_fair_2:.2f}"},
            ]
            st.dataframe(pd.DataFrame(dati_fair), use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ⚽ Goal / No Goal")
            st.dataframe(crea_tabella({"Goal": modello['prob_goal'], "No Goal": modello['prob_nogoal']}, "Opzione"),
                        use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("### 📉 Under / Over")
            oo_dict = {}
            for soglia, prob_u in modello['prob_under'].items():
                oo_dict[f"Under {soglia}"] = prob_u
                oo_dict[f"Over {soglia}"] = 100 - prob_u
            st.dataframe(crea_tabella(oo_dict, "Linea"), use_container_width=True, hide_index=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("### 🏠 Multigol Casa")
            st.dataframe(crea_tabella(modello['multigol_casa'], "Intervallo"), use_container_width=True, hide_index=True)
        with col_d:
            st.markdown("### ✈️ Multigol Ospite")
            st.dataframe(crea_tabella(modello['multigol_trasf'], "Intervallo"), use_container_width=True, hide_index=True)

        st.markdown("### 🔥 Combo Rapide")
        st.dataframe(crea_tabella(modello['combo'], "Combinazione"), use_container_width=True, hide_index=True)

        # =====================================================================
        # 🏆 TOP COMBO AUTOMATICHE
        # Scopre da sola le combinazioni più probabili tra tutte le 12
        # possibili (segno × Gol/No Gol × Over/Under 2.5), riusando la stessa
        # funzione calcola_combo_libera del motore libero qui sotto — nessuna
        # logica duplicata, un solo posto dove le combo vengono calcolate.
        # =====================================================================
        st.markdown("### 🏆 Top 4 Combo Automatiche (su Over/Under 2.5)")
        st.caption("Tutte le 12 combinazioni possibili (1/X/2 × Gol/No Gol × Over/Under 2.5), calibrate "
                   "se disponibile, filtrate per affidabilità (esclude combo troppo rare o su dati scarsi) "
                   "e ordinate per probabilità.")

        # Punto A — filtro di affidabilità: escludiamo le combo troppo rare
        # (rumore, non segnale) o calcolate su squadre con pochi dati specifici.
        SOGLIA_MIN_PROB_COMBO = 8.0  # sotto questa probabilità, troppo raro per essere utile
        dati_scarsi = bool(avvisi)

        tutte_le_combo = {}
        for segno_auto in ["1", "X", "2"]:
            for gg_auto in ["Goal", "NoGoal"]:
                for tipo_auto in ["Over", "Under"]:
                    prob_grezza = calcola_combo_libera(modello['griglia'], segno=segno_auto, soglia_gol=2.5,
                                                        tipo_soglia=tipo_auto, gol_nogol=gg_auto)
                    nome_chiave = f"{segno_auto}_{gg_auto}_{tipo_auto}"
                    prob_finale = prob_grezza
                    calib_combo_info = None
                    if not is_coppa and usa_calibrazione:
                        calib_combo_info = carica_calibratore(id_fd, nome_chiave)
                        if calib_combo_info:
                            prob_finale = float(calib_combo_info["calibratore"].predict([prob_grezza/100])[0]) * 100
                    chiave_vis = f"{segno_auto} + {'Gol' if gg_auto=='Goal' else 'No Gol'} + {tipo_auto} 2.5"
                    tutte_le_combo[chiave_vis] = {
                        "prob": prob_finale, "segno": segno_auto, "tipo": tipo_auto,
                        "calibrata": calib_combo_info is not None,
                    }

        combo_affidabili = {k: v for k, v in tutte_le_combo.items() if v["prob"] >= SOGLIA_MIN_PROB_COMBO}
        top4_combo = dict(sorted(combo_affidabili.items(), key=lambda x: x[1]["prob"], reverse=True)[:4])

        if dati_scarsi:
            st.warning("⚠️ Combo calcolate su dati limitati per una delle due squadre (vedi avviso sopra) "
                      "— trattale con più cautela del solito.")
        if not top4_combo:
            st.info(f"Nessuna combo sopra la soglia minima di affidabilità ({SOGLIA_MIN_PROB_COMBO}%) "
                   "per questa partita.")
        else:
            righe_top4 = []
            for chiave, info_c in top4_combo.items():
                q_stimata, non_prezzate = stima_quota_combo_approssimata(info_c["segno"], 2.5, info_c["tipo"], quote, quote_ou_25)
                righe_top4.append({
                    "Combo": chiave + (" 🎯" if info_c["calibrata"] else ""),
                    "Probabilità (%)": f"{info_c['prob']:.1f}%",
                    "Quota stimata*": f"~{q_stimata:.2f}" if q_stimata else "n/d",
                })
            st.dataframe(pd.DataFrame(righe_top4), use_container_width=True, hide_index=True)
            st.caption("🎯 = probabilità corretta con calibrazione specifica per questa combo. "
                       "*Quota STIMATA per approssimazione (1X2 × Over/Under come se fossero indipendenti — "
                       "non lo sono del tutto). Il componente Gol/No Gol non ha una quota nel file, quindi "
                       "non entra nella stima: il numero reale del bookmaker sarà diverso.")

        st.divider()
        st.markdown("### ⚔️ Ultimi Scontri Diretti (H2H)")
        df_h2h = estrai_scontri_diretti(partita_sel['HomeTeam'], partita_sel['AwayTeam'], dati_filtrati, df_globale_filtrato)
        if not df_h2h.empty:
            cols_mostra = [c for c in ['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam'] if c in df_h2h.columns]
            st.dataframe(df_h2h[cols_mostra], use_container_width=True, hide_index=True)
        else:
            st.info("Nessun precedente recente trovato negli archivi disponibili tra queste due squadre.")

        st.divider()
        st.markdown("### 🎯 Statistiche Match (Stimate)")
        nota_stima = ""
        if avvisi:
            nota_stima = " ⚠️ (basate parzialmente sulla media di lega per scarsità di dati specifici)"
        st.info(f"🚩 Angoli Totali Stimati: **{modello['angoli_stimati']}** | "
                f"🎯 Tiri in Porta Totali Stimati: **{modello['tiri_stimati']}**{nota_stima}")
# =====================================================================
# V8 — VALIDAZIONE STATISTICA AGGREGATA / OOS
# Non modifica il modello originale. Parte dai quattro DataFrame V5 OOS per
# ciascun campionato e aggiunge: bootstrap ROI, IC95%, confronto CAL-RAW,
# bootstrap della differenza di ROI e aggregazione sui 5 campionati.
#
# Nota metodologica: RAW e CAL non sono necessariamente gli stessi bet.
# La differenza CAL-RAW è quindi una differenza tra strategie, non un test
# appaiato sugli stessi eventi.
# =====================================================================

def _v7_bootstrap_roi(df, n_boot=2000, seed=42):
    """Bootstrap a livello di scommessa per ROI/profitto."""
    if df is None or df.empty or 'vinta' not in df.columns or 'quota' not in df.columns:
        return {'n': 0, 'roi': None, 'roi_lo': None, 'roi_hi': None,
                'profit': None, 'profit_lo': None, 'profit_hi': None}
    import numpy as _np
    wins = df['vinta'].astype(bool).to_numpy()
    odds = df['quota'].astype(float).to_numpy()
    profit = _np.where(wins, odds - 1.0, -1.0)
    roi = float(profit.mean() * 100.0)
    total = float(profit.sum())
    rng = _np.random.default_rng(seed)
    idx = rng.integers(0, len(profit), size=(n_boot, len(profit)))
    samples = profit[idx].mean(axis=1) * 100.0
    lo, hi = _np.percentile(samples, [2.5, 97.5])
    totals = profit[idx].sum(axis=1)
    plo, phi = _np.percentile(totals, [2.5, 97.5])
    return {'n': int(len(profit)), 'roi': roi, 'roi_lo': float(lo), 'roi_hi': float(hi),
            'profit': total, 'profit_lo': float(plo), 'profit_hi': float(phi)}


def _v7_make_row(campionato, mercato, versione, df, ev_col='ev', seed=42):
    if df is None or df.empty:
        return {'campionato': campionato, 'mercato': mercato, 'versione': versione,
                'bet': 0, 'strike_%': None, 'brier': None, 'logloss': None,
                'EV_%': None, 'ROI_%': None, 'ROI_IC95_low_%': None,
                'ROI_IC95_high_%': None, 'profit_u': None,
                'profit_IC95_low_u': None, 'profit_IC95_high_u': None,
                'quota_media': None, 'max_dd_u': None}
    d = df.copy()
    if 'vinta' in d.columns and 'quota' in d.columns:
        s = _v6_stats_df(d)
        bs = _v7_bootstrap_roi(d, seed=seed)
    else:
        s = {'bet': len(d), 'strike': None, 'roi': None, 'profit': None,
             'avg_odds': None, 'ev': None, 'max_dd': None}
        bs = {'roi_lo': None, 'roi_hi': None, 'profit_lo': None, 'profit_hi': None}
    ev = None
    if ev_col in d.columns and len(d):
        ev = float(d[ev_col].astype(float).mean() * 100.0)
    elif 'ev' in d.columns and len(d):
        ev = float(d['ev'].astype(float).mean() * 100.0)
    return {'campionato': campionato, 'mercato': mercato, 'versione': versione,
            'bet': int(s.get('bet', len(d))), 'strike_%': s.get('strike'),
            'brier': None, 'logloss': None, 'EV_%': ev,
            'ROI_%': s.get('roi'), 'ROI_IC95_low_%': bs.get('roi_lo'),
            'ROI_IC95_high_%': bs.get('roi_hi'), 'profit_u': s.get('profit'),
            'profit_IC95_low_u': bs.get('profit_lo'),
            'profit_IC95_high_u': bs.get('profit_hi'),
            'quota_media': s.get('avg_odds'), 'max_dd_u': s.get('max_dd')}


def _v8_bootstrap_compare(df_raw, df_cal, n_boot=3000, seed=42):
    import numpy as _np
    def prof(df):
        if df is None or df.empty or 'vinta' not in df.columns or 'quota' not in df.columns:
            return _np.array([], dtype=float)
        return _np.where(df['vinta'].astype(bool).to_numpy(),
                         df['quota'].astype(float).to_numpy()-1.0, -1.0)
    a, b = prof(df_raw), prof(df_cal)
    if len(a)==0 or len(b)==0:
        return {'delta':None,'lo':None,'hi':None,'p_two_sided':None}
    rng=_np.random.default_rng(seed)
    ia=rng.integers(0,len(a),size=(n_boot,len(a)))
    ib=rng.integers(0,len(b),size=(n_boot,len(b)))
    deltas=(b[ib].mean(axis=1)-a[ia].mean(axis=1))*100.0
    delta=(b.mean()-a.mean())*100.0
    lo,hi=_np.percentile(deltas,[2.5,97.5])
    # Two-sided bootstrap sign probability; descriptive, not a formal paired test.
    p=2.0*min(float((deltas<=0).mean()), float((deltas>=0).mean()))
    return {'delta':float(delta),'lo':float(lo),'hi':float(hi),'p_two_sided':float(min(1.0,p))}


def _v8_aggregate(rows, camp_weights=False):
    if not rows:
        return pd.DataFrame()
    d=pd.DataFrame(rows)
    # Aggregazione economica corretta: somma profitti / numero di bet.
    out=[]
    for (merc,ver), g in d.groupby(['mercato','versione'], sort=False):
        bets=pd.to_numeric(g['bet'],errors='coerce').fillna(0).sum()
        profit=pd.to_numeric(g['profit_u'],errors='coerce').fillna(0).sum()
        roi=100.0*profit/bets if bets else None
        out.append({'mercato':merc,'versione':ver,'bet_totali':int(bets),
                    'profit_totale_u':float(profit),'ROI_aggregato_%':roi,
                    'quota_media_pesata':None})
    return pd.DataFrame(out)


def _v8_build_all(n_boot=3000, warmup=100):
    rows=[]; compares=[]; errors=[]
    for camp, info in CAMPIONATI_DOMESTICI.items():
        try:
            dati_b=carica_dati_campionato(info['id_fd'])
            v5=_v5_run_economic(dati_b, rho_val, ewma_span_val, emivita_val, 0.10, warmup)
            pairs=[('1X2','RAW',v5.get('raw12')),('1X2','CALIBRATO OOS',v5.get('cal12')),
                   ('O/U 2.5','RAW',v5.get('rawou')),('O/U 2.5','CALIBRATO OOS',v5.get('calou'))]
            for merc,ver,d in pairs:
                r=_v7_make_row(camp,merc,ver,d,'ev',seed=42)
                rows.append(r)
            compares.append((camp,'1X2',_v8_bootstrap_compare(v5.get('raw12'),v5.get('cal12'),n_boot,101)))
            compares.append((camp,'O/U 2.5',_v8_bootstrap_compare(v5.get('rawou'),v5.get('calou'),n_boot,202)))
        except Exception as e:
            errors.append((camp,f'{type(e).__name__}: {e}'))
    return rows,compares,errors



# =====================================================================
# V9 — CONFRONTO APPAIATO CAL vs RAW SULLA STESSA OPPORTUNITÀ
# Per ogni riga OOS, RAW e CAL vedono la stessa partita e le stesse quote.
# Si calcolano due viste: ALL OOS (0 profit se la strategia non entra) e
# COMMON BETS (solo righe dove entrambe entrano). Il bootstrap è appaiato
# sulla differenza di profitto per riga, non su due campioni indipendenti.
# =====================================================================
def _v9_build_paired(dati_completi, rho, ewma_span, emivita, soglia_ev=0.10, warmup=100):
    hist = _v5_raw_history(dati_completi, rho, ewma_span, emivita)
    if hist.empty:
        return None
    rows12=[]; rowsou=[]
    prior12_p={'1':[], 'X':[], '2':[]}; prior12_y={'1':[], 'X':[], '2':[]}
    prior_ou_p=[]; prior_ou_y=[]
    for j,r in hist.iterrows():
        if j < warmup:
            prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
            prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
            prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
            prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)
            continue

        cal12_models={s:_v5_fit_iso_binary(prior12_p[s],prior12_y[s]) for s in ['1','X','2']}
        rawp={'1':float(r.p1),'X':float(r.px),'2':float(r.p2)}
        if all(cal12_models[s] is not None for s in ['1','X','2']):
            cp={s:float(cal12_models[s].predict([rawp[s]])[0]) for s in ['1','X','2']}
            tot=sum(cp.values()); cp={s:v/tot for s,v in cp.items()} if tot>0 else rawp.copy()
        else: cp=rawp.copy()
        rc=max(rawp,key=rawp.get); cc=max(cp,key=cp.get)
        rq={'1':float(r.q1),'X':float(r.qx),'2':float(r.q2)}
        raw_ev=rawp[rc]*rq[rc]-1.0; cal_ev=cp[cc]*rq[cc]-1.0
        raw_in=raw_ev>=soglia_ev; cal_in=cal_ev>=soglia_ev
        raw_prof=(rq[rc]-1.0 if rc==r.esito12 else -1.0) if raw_in else 0.0
        cal_prof=(rq[cc]-1.0 if cc==r.esito12 else -1.0) if cal_in else 0.0
        rows12.append({'data':r.data,'raw_in':raw_in,'cal_in':cal_in,'raw_profit':raw_prof,'cal_profit':cal_prof,
                       'raw_ev':raw_ev,'cal_ev':cal_ev,'raw_choice':rc,'cal_choice':cc,'quota_raw':rq[rc],'quota_cal':rq[cc],
                       'raw_hit':bool(rc==r.esito12) if raw_in else None,'cal_hit':bool(cc==r.esito12) if cal_in else None})

        cal_ou_model=_v5_fit_iso_binary(prior_ou_p,prior_ou_y)
        rawou={'Over':float(r.po),'Under':float(r.pu)}
        if cal_ou_model is not None:
            pco=min(max(float(cal_ou_model.predict([r.po])[0]),0.001),0.999)
            cou={'Over':pco,'Under':1-pco}
        else: cou=rawou.copy()
        ro=max(rawou,key=rawou.get); co=max(cou,key=cou.get)
        oq={'Over':float(r.qo),'Under':float(r.qu)}
        rev=rawou[ro]*oq[ro]-1.0; cev=cou[co]*oq[co]-1.0
        rin=rev>=soglia_ev; cin=cev>=soglia_ev
        rp=(oq[ro]-1.0 if ro==r.esito_ou else -1.0) if rin else 0.0
        cpv=(oq[co]-1.0 if co==r.esito_ou else -1.0) if cin else 0.0
        rowsou.append({'data':r.data,'raw_in':rin,'cal_in':cin,'raw_profit':rp,'cal_profit':cpv,
                       'raw_ev':rev,'cal_ev':cev,'raw_choice':ro,'cal_choice':co,'quota_raw':oq[ro],'quota_cal':oq[co],
                       'raw_hit':bool(ro==r.esito_ou) if rin else None,'cal_hit':bool(co==r.esito_ou) if cin else None})

        prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
        prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
        prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
        prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)
    return {'1X2':pd.DataFrame(rows12),'O/U 2.5':pd.DataFrame(rowsou)}


def _v9_paired_stats(df, n_boot=3000, seed=42, common_only=False):
    import numpy as _np
    if df is None or df.empty: return {'n':0,'raw_bets':0,'cal_bets':0,'delta_profit':None,'delta_roi_pp':None,'lo':None,'hi':None,'p_two_sided':None}
    d=df[df.raw_in & df.cal_in].copy() if common_only else df.copy()
    if d.empty: return {'n':0,'raw_bets':0,'cal_bets':0,'delta_profit':None,'delta_roi_pp':None,'lo':None,'hi':None,'p_two_sided':None}
    raw=d.raw_profit.to_numpy(float); cal=d.cal_profit.to_numpy(float); diff=cal-raw
    rng=_np.random.default_rng(seed); idx=rng.integers(0,len(diff),size=(n_boot,len(diff)))
    boot=diff[idx].mean(axis=1)*100.0
    lo,hi=_np.percentile(boot,[2.5,97.5]); delta=diff.mean()*100.0
    p=2*min(float((boot<=0).mean()),float((boot>=0).mean()))
    raw_b=int(d.raw_in.sum()); cal_b=int(d.cal_in.sum())
    raw_profit=float(raw.sum()); cal_profit=float(cal.sum())
    raw_roi=100*raw_profit/raw_b if raw_b else None; cal_roi=100*cal_profit/cal_b if cal_b else None
    return {'n':len(d),'raw_bets':raw_b,'cal_bets':cal_b,'raw_profit':raw_profit,'cal_profit':cal_profit,
            'raw_roi':raw_roi,'cal_roi':cal_roi,'delta_profit':delta,'delta_roi_pp':(cal_roi-raw_roi) if raw_roi is not None and cal_roi is not None else None,
            'lo':float(lo),'hi':float(hi),'p_two_sided':float(min(1,p))}


def _v9_build_all(n_boot=3000,warmup=100):
    rows=[]; errors=[]
    for camp,info in CAMPIONATI_DOMESTICI.items():
        try:
            dati=carica_dati_campionato(info['id_fd'])
            paired=_v9_build_paired(dati,rho_val,ewma_span_val,emivita_val,0.10,warmup)
            for merc,df in paired.items():
                for mode,common in [('ALL OOS',False),('COMMON BETS',True)]:
                    st=_v9_paired_stats(df,n_boot,42 if merc=='1X2' else 43,common)
                    rows.append({'campionato':camp,'mercato':merc,'universo':mode,**st})
        except Exception as e: errors.append((camp,f'{type(e).__name__}: {e}'))
    return pd.DataFrame(rows),errors


def mostra_v9_validation():
    st.divider(); st.markdown('## 🧪 V9 — PAIRED TEST RAW vs CAL OOS')
    st.caption('RAW e CAL vengono confrontati sulla stessa partita e sulle stesse quote. Il bootstrap ricampiona la differenza CAL−RAW per partita; non sono due bootstrap indipendenti.')
    c1,c2=st.columns(2)
    with c1: nb=st.number_input('Bootstrap V9',500,10000,3000,500,key='v9_boot')
    with c2: wu=st.number_input('Warm-up V9',50,300,100,10,key='v9_warm')
    st.info('ALL OOS assegna profitto 0 quando una strategia non entra: misura l’effetto complessivo della strategia. COMMON BETS usa solo le partite in cui entrambe entrano: misura il confronto diretto sulle opportunità condivise.')
    if st.button('🧪 ESEGUI V9 — PAIRED TEST',key='v9_run'):
        try:
            with st.spinner('V9 in esecuzione sui 5 campionati...'):
                out,errors=_v9_build_all(int(nb),int(wu))
            if not out.empty:
                cols=['campionato','mercato','universo','n','raw_bets','cal_bets','raw_profit','cal_profit','raw_roi','cal_roi','delta_profit','delta_roi_pp','lo','hi','p_two_sided']
                st.dataframe(out[cols],use_container_width=True,hide_index=True)
                st.download_button('⬇️ Scarica V9 paired test',data=out.to_csv(index=False).encode('utf-8'),file_name='V9_paired_test.csv',mime='text/csv',key='v9_dl')
                st.markdown('### Come leggere il test')
                st.write('`delta_profit` è il vantaggio medio per partita di CAL rispetto a RAW, in unità. `lo/hi` è il suo IC95% bootstrap. Se l’intervallo attraversa 0, il campione non separa nettamente le due strategie.')
            if errors:
                st.error('Campionati non completati:'); [st.write(f'- **{n}**: {m}') for n,m in errors]
        except Exception as e: st.error(f'Errore V9: {type(e).__name__}: {e}')

def mostra_v8_validation():
    st.divider()
    st.markdown('## 🧪 V8 — VALIDAZIONE STATISTICA AGGREGATA / OOS')
    st.caption('Confronto RAW vs CALIBRATO OOS sui 5 campionati, con bootstrap del ROI, IC95% e intervallo della differenza CAL−RAW. I risultati economici restano backtest storici con quote proxy.')
    c1,c2=st.columns(2)
    with c1:
        n_boot=st.number_input('Bootstrap V8',min_value=500,max_value=10000,value=3000,step=500,key='v8_boot')
    with c2:
        warm=st.number_input('Warm-up OOS',min_value=50,max_value=300,value=100,step=10,key='v8_warm')
    st.warning('⚠️ V8 può richiedere alcuni minuti: esegue una sola volta per campionato e poi calcola il confronto statistico. RAW e CAL non sono necessariamente gli stessi bet.')
    if st.button('🧪 ESEGUI V8 — VALIDAZIONE COMPLETA',key='v8_run'):
        try:
            with st.spinner('V8 in esecuzione sui 5 campionati...'):
                rows,compares,errors=_v8_build_all(int(n_boot),int(warm))
            if rows:
                out=pd.DataFrame(rows)
                st.markdown('### 1. Risultati per campionato')
                st.dataframe(out,use_container_width=True,hide_index=True)
                st.download_button('⬇️ Scarica V8 risultati per campionato',data=out.to_csv(index=False).encode('utf-8'),file_name='V8_per_campionato.csv',mime='text/csv',key='v8_dl1')
                agg=_v8_aggregate(rows)
                st.markdown('### 2. Aggregazione dei 5 campionati')
                st.dataframe(agg,use_container_width=True,hide_index=True)
                st.download_button('⬇️ Scarica V8 aggregato',data=agg.to_csv(index=False).encode('utf-8'),file_name='V8_aggregato.csv',mime='text/csv',key='v8_dl2')
            if compares:
                comp_rows=[]
                for camp,merc,x in compares:
                    comp_rows.append({'campionato':camp,'mercato':merc,
                                      'delta_ROI_CAL_minus_RAW_pp':x['delta'],
                                      'IC95_low_pp':x['lo'],'IC95_high_pp':x['hi'],
                                      'p_bootstrap_due_code':x['p_two_sided']})
                comp=pd.DataFrame(comp_rows)
                st.markdown('### 3. Differenza CAL − RAW')
                st.dataframe(comp,use_container_width=True,hide_index=True)
                st.caption('Interpretazione: se l’IC95% della differenza attraversa 0, il campione non mostra una separazione netta tra le due strategie. Il p-value bootstrap è descrittivo e non sostituisce un test appaiato.')
                st.download_button('⬇️ Scarica V8 confronto CAL-RAW',data=comp.to_csv(index=False).encode('utf-8'),file_name='V8_confronto_CAL_RAW.csv',mime='text/csv',key='v8_dl3')
            if errors:
                st.error('Campionati non completati:')
                for nome,msg in errors: st.write(f'- **{nome}**: {msg}')
        except Exception as e:
            st.error(f'Errore V8: {type(e).__name__}: {e}')

try:
    mostra_v8_validation()
except Exception as _v8_err:
    st.error(f'V8 non disponibile: {type(_v8_err).__name__}: {_v8_err}')

# V9 — render della sezione nell'app
try:
    mostra_v9_validation()
except Exception as _v9_err:
    st.error(f"V9 non disponibile: {type(_v9_err).__name__}: {_v9_err}")


# =====================================================================
# V10 — AGGREGATO PAIRED + BOOTSTRAP STRATIFICATO + QUALITÀ PROBABILISTICA
# =====================================================================
def _v10_build_paired(dati_completi, rho, ewma_span, emivita, soglia_ev=0.10, warmup=100):
    """Come V9, ma conserva anche le probabilità RAW/CAL per Brier e Log Loss OOS."""
    hist = _v5_raw_history(dati_completi, rho, ewma_span, emivita)
    if hist.empty:
        return None
    rows12=[]; rowsou=[]
    prior12_p={'1':[], 'X':[], '2':[]}; prior12_y={'1':[], 'X':[], '2':[]}
    prior_ou_p=[]; prior_ou_y=[]
    for j,r in hist.iterrows():
        if j < warmup:
            prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
            prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
            prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
            prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)
            continue

        # 1X2: calibratori isotonic separati, fit solo sul passato.
        cal12_models={s:_v5_fit_iso_binary(prior12_p[s],prior12_y[s]) for s in ['1','X','2']}
        rawp={'1':float(r.p1),'X':float(r.px),'2':float(r.p2)}
        if all(cal12_models[s] is not None for s in ['1','X','2']):
            cp={s:float(cal12_models[s].predict([rawp[s]])[0]) for s in ['1','X','2']}
            tot=sum(cp.values()); cp={s:v/tot for s,v in cp.items()} if tot>0 else rawp.copy()
        else:
            cp=rawp.copy()
        rc=max(rawp,key=rawp.get); cc=max(cp,key=cp.get)
        rq={'1':float(r.q1),'X':float(r.qx),'2':float(r.q2)}
        raw_ev=rawp[rc]*rq[rc]-1.0; cal_ev=cp[cc]*rq[cc]-1.0
        raw_in=raw_ev>=soglia_ev; cal_in=cal_ev>=soglia_ev
        raw_prof=(rq[rc]-1.0 if rc==r.esito12 else -1.0) if raw_in else 0.0
        cal_prof=(rq[cc]-1.0 if cc==r.esito12 else -1.0) if cal_in else 0.0
        rows12.append({
            'data':r.data,'raw_in':raw_in,'cal_in':cal_in,'raw_profit':raw_prof,'cal_profit':cal_prof,
            'raw_ev':raw_ev,'cal_ev':cal_ev,'raw_choice':rc,'cal_choice':cc,
            'quota_raw':rq[rc],'quota_cal':rq[cc],
            'raw_hit':bool(rc==r.esito12) if raw_in else None,'cal_hit':bool(cc==r.esito12) if cal_in else None,
            'p1_raw':rawp['1'],'px_raw':rawp['X'],'p2_raw':rawp['2'],
            'p1_cal':cp['1'],'px_cal':cp['X'],'p2_cal':cp['2'],'esito':r.esito12
        })

        # O/U 2.5: isotonic sull'Over, Under = 1-Over.
        cal_ou_model=_v5_fit_iso_binary(prior_ou_p,prior_ou_y)
        rawou={'Over':float(r.po),'Under':float(r.pu)}
        if cal_ou_model is not None:
            pco=min(max(float(cal_ou_model.predict([r.po])[0]),0.001),0.999)
            cou={'Over':pco,'Under':1-pco}
        else:
            cou=rawou.copy()
        ro=max(rawou,key=rawou.get); co=max(cou,key=cou.get)
        oq={'Over':float(r.qo),'Under':float(r.qu)}
        rev=rawou[ro]*oq[ro]-1.0; cev=cou[co]*oq[co]-1.0
        rin=rev>=soglia_ev; cin=cev>=soglia_ev
        rp=(oq[ro]-1.0 if ro==r.esito_ou else -1.0) if rin else 0.0
        cpv=(oq[co]-1.0 if co==r.esito_ou else -1.0) if cin else 0.0
        rowsou.append({
            'data':r.data,'raw_in':rin,'cal_in':cin,'raw_profit':rp,'cal_profit':cpv,
            'raw_ev':rev,'cal_ev':cev,'raw_choice':ro,'cal_choice':co,
            'quota_raw':oq[ro],'quota_cal':oq[co],
            'raw_hit':bool(ro==r.esito_ou) if rin else None,'cal_hit':bool(co==r.esito_ou) if cin else None,
            'po_raw':rawou['Over'],'pu_raw':rawou['Under'],
            'po_cal':cou['Over'],'pu_cal':cou['Under'],'esito':r.esito_ou
        })

        prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
        prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
        prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
        prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)
    return {'1X2':pd.DataFrame(rows12),'O/U 2.5':pd.DataFrame(rowsou)}


def _v10_metrics(df, merc, common_only=False):
    """Metriche OOS descrittive RAW/CAL, inclusi Brier e Log Loss."""
    if df is None or df.empty:
        return None
    d=df[df.raw_in & df.cal_in].copy() if common_only else df.copy()
    if d.empty:
        return None
    if merc=='1X2':
        ymap={'1':0,'X':1,'2':2}
        y=np.array([ymap[x] for x in d.esito],dtype=int)
        raw=np.column_stack([d.p1_raw,d.px_raw,d.p2_raw]).astype(float)
        cal=np.column_stack([d.p1_cal,d.px_cal,d.p2_cal]).astype(float)
        raw_brier=np.mean(np.sum((raw-np.eye(3)[y])**2,axis=1))
        cal_brier=np.mean(np.sum((cal-np.eye(3)[y])**2,axis=1))
        raw_ll=float(-np.mean(np.log(np.clip(raw[np.arange(len(y)),y],EPS_PROB,1-EPS_PROB))))
        cal_ll=float(-np.mean(np.log(np.clip(cal[np.arange(len(y)),y],EPS_PROB,1-EPS_PROB))))
        raw_acc=float(np.mean(np.argmax(raw,axis=1)==y)*100)
        cal_acc=float(np.mean(np.argmax(cal,axis=1)==y)*100)
    else:
        y=np.array([1 if x=='Over' else 0 for x in d.esito],dtype=int)
        raw=np.asarray(d.po_raw,dtype=float); cal=np.asarray(d.po_cal,dtype=float)
        raw_brier=float(np.mean((raw-y)**2)); cal_brier=float(np.mean((cal-y)**2))
        raw_ll=float(-np.mean(y*np.log(np.clip(raw,EPS_PROB,1-EPS_PROB))+(1-y)*np.log(np.clip(1-raw,EPS_PROB,1-EPS_PROB))))
        cal_ll=float(-np.mean(y*np.log(np.clip(cal,EPS_PROB,1-EPS_PROB))+(1-y)*np.log(np.clip(1-cal,EPS_PROB,1-EPS_PROB))))
        raw_acc=float(np.mean((raw>=0.5)==(y==1))*100); cal_acc=float(np.mean((cal>=0.5)==(y==1))*100)
    return {'n':len(d),'raw_acc':raw_acc,'cal_acc':cal_acc,'delta_acc_pp':cal_acc-raw_acc,
            'raw_brier':raw_brier,'cal_brier':cal_brier,'delta_brier':cal_brier-raw_brier,
            'raw_logloss':raw_ll,'cal_logloss':cal_ll,'delta_logloss':cal_ll-raw_ll}


def _v10_bootstrap_stratified(dfs, merc, n_boot=3000, common_only=False, seed=42):
    """Bootstrap stratificato per campionato. Ogni replica ricampiona dentro ogni lega e poi aggrega."""
    valid=[]
    for camp,df in dfs.items():
        if df is None or df.empty: continue
        d=df[df.raw_in & df.cal_in].copy() if common_only else df.copy()
        if not d.empty: valid.append((camp,d.reset_index(drop=True)))
    if not valid: return None
    rng=np.random.default_rng(seed)
    deltas=[]; raw_rois=[]; cal_rois=[]; db=[]; dll=[]
    # Ogni lega mantiene il proprio peso in numero di opportunità osservate.
    for _ in range(int(n_boot)):
        total_raw_profit=total_cal_profit=0.0; total_raw_bets=total_cal_bets=0
        braw=bcal=llraw=llcal=0.0; nprob=0
        for camp,d in valid:
            idx=rng.integers(0,len(d),size=len(d)); x=d.iloc[idx]
            total_raw_profit += float(x.raw_profit.sum()); total_cal_profit += float(x.cal_profit.sum())
            total_raw_bets += int(x.raw_in.sum()); total_cal_bets += int(x.cal_in.sum())
            m=_v10_metrics(x,merc,False)
            if m:
                braw += m['raw_brier']*m['n']; bcal += m['cal_brier']*m['n']
                llraw += m['raw_logloss']*m['n']; llcal += m['cal_logloss']*m['n']; nprob += m['n']
        raw_roi=100*total_raw_profit/total_raw_bets if total_raw_bets else np.nan
        cal_roi=100*total_cal_profit/total_cal_bets if total_cal_bets else np.nan
        if np.isfinite(raw_roi) and np.isfinite(cal_roi):
            raw_rois.append(raw_roi); cal_rois.append(cal_roi); deltas.append(cal_roi-raw_roi)
        if nprob:
            db.append(bcal/nprob-braw/nprob); dll.append(llcal/nprob-llraw/nprob)
    def ci(a):
        return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5))) if a else (None,None)
    lo,hi=ci(deltas); blo,bhi=ci(db); llo,lhi=ci(dll)
    obs_raw_profit=sum(float(d.raw_profit.sum()) for _,d in valid); obs_cal_profit=sum(float(d.cal_profit.sum()) for _,d in valid)
    obs_raw_bets=sum(int(d.raw_in.sum()) for _,d in valid); obs_cal_bets=sum(int(d.cal_in.sum()) for _,d in valid)
    obs_raw_roi=100*obs_raw_profit/obs_raw_bets if obs_raw_bets else None
    obs_cal_roi=100*obs_cal_profit/obs_cal_bets if obs_cal_bets else None
    def p2(a):
        if not a: return None
        return float(min(1,2*min(np.mean(np.asarray(a)<=0),np.mean(np.asarray(a)>=0))))
    return {'leagues':len(valid),'n':sum(len(d) for _,d in valid),'raw_bets':obs_raw_bets,'cal_bets':obs_cal_bets,
            'raw_profit':obs_raw_profit,'cal_profit':obs_cal_profit,'raw_roi':obs_raw_roi,'cal_roi':obs_cal_roi,
            'delta_roi_pp':obs_cal_roi-obs_raw_roi if obs_raw_roi is not None and obs_cal_roi is not None else None,
            'roi_lo':lo,'roi_hi':hi,'roi_p':p2(deltas),
            'brier_delta':float(np.mean([_v10_metrics(d,merc,False)['delta_brier']*len(d) for _,d in valid])/sum(len(d) for _,d in valid)),
            'brier_lo':blo,'brier_hi':bhi,'brier_p':p2(db),
            'logloss_delta':float(np.mean([_v10_metrics(d,merc,False)['delta_logloss']*len(d) for _,d in valid])/sum(len(d) for _,d in valid)),
            'logloss_lo':llo,'logloss_hi':lhi,'logloss_p':p2(dll)}


def _v10_build_all(n_boot=3000,warmup=100):
    per=[]; errors=[]; dfs12={}; dfsou={}
    for camp,info in CAMPIONATI_DOMESTICI.items():
        try:
            dati=carica_dati_campionato(info['id_fd'])
            paired=_v10_build_paired(dati,rho_val,ewma_span_val,emivita_val,0.10,warmup)
            if paired is None: raise ValueError('dati OOS vuoti')
            dfs12[camp]=paired['1X2']; dfsou[camp]=paired['O/U 2.5']
            for merc,df in paired.items():
                for common in [False,True]:
                    m=_v10_metrics(df,merc,common)
                    if m:
                        per.append({'campionato':camp,'mercato':merc,'universo':'COMMON BETS' if common else 'ALL OOS',**m})
        except Exception as e:
            errors.append((camp,f'{type(e).__name__}: {e}'))
    agg=[]
    for merc,dfs in [('1X2',dfs12),('O/U 2.5',dfsou)]:
        for common in [False,True]:
            b=_v10_bootstrap_stratified(dfs,merc,n_boot,common,42 if merc=='1X2' else 43)
            if b:
                agg.append({'mercato':merc,'universo':'COMMON BETS' if common else 'ALL OOS',**b})
    return pd.DataFrame(per),pd.DataFrame(agg),errors


def mostra_v10_validation():
    st.divider(); st.markdown('## 🧪 V10 — AGGREGATO 5 LEGHE: PAIRED + QUALITÀ PROBABILISTICA')
    st.caption('V10 consolida le 5 leghe con bootstrap stratificato per campionato. Separa 1X2 e O/U 2.5, ALL OOS e COMMON BETS, e confronta anche Accuracy, Brier e Log Loss OOS.')
    c1,c2=st.columns(2)
    with c1: nb=st.number_input('Bootstrap V10',500,10000,3000,500,key='v10_boot')
    with c2: wu=st.number_input('Warm-up V10',50,300,100,10,key='v10_warm')
    st.info('Interpretazione: ROI/Profit misurano l’effetto economico storico; Brier e Log Loss misurano la qualità delle probabilità. Per Brier e Log Loss più basso è migliore. Gli IC bootstrap sono stratificati per lega e il p-value è descrittivo.')
    if st.button('🧪 ESEGUI V10 — AGGREGATO 5 LEGHE',key='v10_run'):
        try:
            with st.spinner('V10 in esecuzione: una passata per ciascuna delle 5 leghe...'):
                per,agg,errors=_v10_build_all(int(nb),int(wu))
            if not per.empty:
                st.markdown('### 1. Metriche OOS per lega')
                cols=['campionato','mercato','universo','n','raw_acc','cal_acc','delta_acc_pp','raw_brier','cal_brier','delta_brier','raw_logloss','cal_logloss','delta_logloss']
                st.dataframe(per[cols],use_container_width=True,hide_index=True)
                st.download_button('⬇️ Scarica V10 metriche per lega',data=per.to_csv(index=False).encode('utf-8'),file_name='V10_metriche_per_lega.csv',mime='text/csv',key='v10_dl1')
            if not agg.empty:
                st.markdown('### 2. Aggregato 5 leghe — bootstrap stratificato')
                cols=['mercato','universo','leagues','n','raw_bets','cal_bets','raw_profit','cal_profit','raw_roi','cal_roi','delta_roi_pp','roi_lo','roi_hi','roi_p','brier_delta','brier_lo','brier_hi','brier_p','logloss_delta','logloss_lo','logloss_hi','logloss_p']
                st.dataframe(agg[cols],use_container_width=True,hide_index=True)
                st.download_button('⬇️ Scarica V10 aggregato',data=agg.to_csv(index=False).encode('utf-8'),file_name='V10_aggregato_stratificato.csv',mime='text/csv',key='v10_dl2')
                st.markdown('### 3. Regola di lettura')
                st.write('Per ROI, Brier e Log Loss guarda il delta **CAL − RAW** insieme all’IC95%. Per ROI un delta positivo favorisce CAL; per Brier/Log Loss un delta negativo indica probabilità migliori. Un IC che attraversa 0 non separa nettamente le versioni nel campione.')
            if errors:
                st.error('Campionati non completati:')
                for nome,msg in errors: st.write(f'- **{nome}**: {msg}')
        except Exception as e:
            st.error(f'Errore V10: {type(e).__name__}: {e}')

try:
    mostra_v10_validation()
except Exception as _v10_err:
    st.error(f"V10 non disponibile: {type(_v10_err).__name__}: {_v10_err}")

# =====================================================================
# V11 — DIAGNOSTICA DI CALIBRAZIONE OOS PER FASCE DI PROBABILITÀ
# =====================================================================
def _v11_build_paired(dati_completi, rho, ewma_span, emivita, soglia_ev=0.10, warmup=100):
    """V11: come V10, ma mantiene anche le probabilità della scelta RAW/CAL
    per costruire reliability tables OOS per fasce di confidenza."""
    hist = _v5_raw_history(dati_completi, rho, ewma_span, emivita)
    if hist.empty:
        return None
    rows12=[]; rowsou=[]
    prior12_p={'1':[], 'X':[], '2':[]}; prior12_y={'1':[], 'X':[], '2':[]}
    prior_ou_p=[]; prior_ou_y=[]
    for j,r in hist.iterrows():
        if j < warmup:
            prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
            prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
            prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
            prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)
            continue

        cal12_models={s:_v5_fit_iso_binary(prior12_p[s],prior12_y[s]) for s in ['1','X','2']}
        rawp={'1':float(r.p1),'X':float(r.px),'2':float(r.p2)}
        if all(cal12_models[s] is not None for s in ['1','X','2']):
            cp={s:float(cal12_models[s].predict([rawp[s]])[0]) for s in ['1','X','2']}
            tot=sum(cp.values()); cp={s:v/tot for s,v in cp.items()} if tot>0 else rawp.copy()
        else:
            cp=rawp.copy()
        rc=max(rawp,key=rawp.get); cc=max(cp,key=cp.get)
        rq={'1':float(r.q1),'X':float(r.qx),'2':float(r.q2)}
        raw_ev=rawp[rc]*rq[rc]-1.0; cal_ev=cp[cc]*rq[cc]-1.0
        raw_in=raw_ev>=soglia_ev; cal_in=cal_ev>=soglia_ev
        raw_prof=(rq[rc]-1.0 if rc==r.esito12 else -1.0) if raw_in else 0.0
        cal_prof=(rq[cc]-1.0 if cc==r.esito12 else -1.0) if cal_in else 0.0
        rows12.append({
            'data':r.data,'raw_in':raw_in,'cal_in':cal_in,'raw_profit':raw_prof,'cal_profit':cal_prof,
            'raw_ev':raw_ev,'cal_ev':cal_ev,'raw_choice':rc,'cal_choice':cc,
            'quota_raw':rq[rc],'quota_cal':rq[cc],
            'raw_hit':bool(rc==r.esito12),'cal_hit':bool(cc==r.esito12),
            'raw_conf':rawp[rc],'cal_conf':cp[cc],
            'p1_raw':rawp['1'],'px_raw':rawp['X'],'p2_raw':rawp['2'],
            'p1_cal':cp['1'],'px_cal':cp['X'],'p2_cal':cp['2'],'esito':r.esito12
        })

        cal_ou_model=_v5_fit_iso_binary(prior_ou_p,prior_ou_y)
        rawou={'Over':float(r.po),'Under':float(r.pu)}
        if cal_ou_model is not None:
            pco=min(max(float(cal_ou_model.predict([r.po])[0]),0.001),0.999)
            cou={'Over':pco,'Under':1-pco}
        else:
            cou=rawou.copy()
        ro=max(rawou,key=rawou.get); co=max(cou,key=cou.get)
        oq={'Over':float(r.qo),'Under':float(r.qu)}
        rev=rawou[ro]*oq[ro]-1.0; cev=cou[co]*oq[co]-1.0
        rin=rev>=soglia_ev; cin=cev>=soglia_ev
        rp=(oq[ro]-1.0 if ro==r.esito_ou else -1.0) if rin else 0.0
        cpv=(oq[co]-1.0 if co==r.esito_ou else -1.0) if cin else 0.0
        rowsou.append({
            'data':r.data,'raw_in':rin,'cal_in':cin,'raw_profit':rp,'cal_profit':cpv,
            'raw_ev':rev,'cal_ev':cev,'raw_choice':ro,'cal_choice':co,
            'quota_raw':oq[ro],'quota_cal':oq[co],
            'raw_hit':bool(ro==r.esito_ou),'cal_hit':bool(co==r.esito_ou),
            'raw_conf':rawou[ro],'cal_conf':cou[co],
            'po_raw':rawou['Over'],'pu_raw':rawou['Under'],
            'po_cal':cou['Over'],'pu_cal':cou['Under'],'esito':r.esito_ou
        })

        prior12_p['1'].append(r.p1); prior12_y['1'].append(1 if r.esito12=='1' else 0)
        prior12_p['X'].append(r.px); prior12_y['X'].append(1 if r.esito12=='X' else 0)
        prior12_p['2'].append(r.p2); prior12_y['2'].append(1 if r.esito12=='2' else 0)
        prior_ou_p.append(r.po); prior_ou_y.append(1 if r.esito_ou=='Over' else 0)
    return {'1X2':pd.DataFrame(rows12),'O/U 2.5':pd.DataFrame(rowsou)}


def _v11_band_table(df, merc, common_only=False):
    if df is None or df.empty:
        return pd.DataFrame()
    d=df[df.raw_in & df.cal_in].copy() if common_only else df.copy()
    if d.empty:
        return pd.DataFrame()
    bins=[0.0,0.40,0.50,0.60,0.70,0.80,1.0000001]
    labels=['0-40%','40-50%','50-60%','60-70%','70-80%','80-100%']
    out=[]
    for ver in ['raw','cal']:
        conf=np.asarray(d[f'{ver}_conf'],dtype=float)
        hit=np.asarray(d[f'{ver}_hit'],dtype=bool)
        idx=np.digitize(conf,bins[1:-1],right=False)
        for k,label in enumerate(labels):
            mask=idx==k
            n=int(mask.sum())
            if not n: continue
            p=conf[mask]; h=hit[mask]
            profit=np.asarray(d.loc[mask,'{}_profit'.format(ver)],dtype=float)
            bets=np.asarray(d.loc[mask,'{}_in'.format(ver)],dtype=bool)
            nb=int(bets.sum())
            out.append({
                'version':ver.upper(),'fascia':label,'n_opportunita':n,
                'prob_media':float(p.mean()),'frequenza_reale':float(h.mean()*100),
                'errore_cal_pp':float(h.mean()*100-p.mean()*100),
                'accuracy_pct':float(h.mean()*100),
                'bets_ev10':nb,
                'roi_ev10_pct':float(100*profit.sum()/nb) if nb else np.nan,
                'profit_ev10_u':float(profit.sum())
            })
    return pd.DataFrame(out)


def _v11_aggregate_band_tables(dfs, merc, common_only=False):
    frames=[]
    for camp,df in dfs.items():
        t=_v11_band_table(df,merc,common_only)
        if not t.empty:
            t.insert(0,'campionato',camp); frames.append(t)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def _v11_build_all(warmup=100):
    per=[]; dfs12={}; dfsou={}; errors=[]
    for camp,info in CAMPIONATI_DOMESTICI.items():
        try:
            dati=carica_dati_campionato(info['id_fd'])
            paired=_v11_build_paired(dati,rho_val,ewma_span_val,emivita_val,0.10,warmup)
            if paired is None: raise ValueError('dati OOS vuoti')
            dfs12[camp]=paired['1X2']; dfsou[camp]=paired['O/U 2.5']
            for merc,df in paired.items():
                for common in [False,True]:
                    t=_v11_band_table(df,merc,common)
                    if not t.empty:
                        t.insert(0,'campionato',camp); t.insert(1,'mercato',merc); t.insert(2,'universo','COMMON BETS' if common else 'ALL OOS'); per.append(t)
        except Exception as e:
            errors.append((camp,f'{type(e).__name__}: {e}'))
    return pd.concat(per,ignore_index=True) if per else pd.DataFrame(), dfs12, dfsou, errors


def mostra_v11_validation():
    st.divider(); st.markdown('## 🔬 V11 — DIAGNOSTICA CALIBRAZIONE OOS PER FASCE')
    st.caption('V11 verifica se RAW e CAL sono calibrati per fasce di confidenza OOS. Per 1X2 usa la probabilità della scelta più probabile; per O/U usa la probabilità della scelta Over/Under più probabile. Include hit-rate reale, errore di calibrazione e ROI sulle sole selezioni EV≥10%.')
    c1=st.columns(1)[0]
    wu=st.number_input('Warm-up V11',50,300,100,10,key='v11_warm')
    st.info('Interpretazione: errore_cal_pp = frequenza reale − probabilità media. Valore vicino a 0 indica migliore calibrazione. Valore negativo indica overconfidence; positivo indica underconfidence. ROI è separato dalla qualità probabilistica.')
    if st.button('🔬 ESEGUI V11 — DIAGNOSTICA 5 LEGHE',key='v11_run'):
        try:
            with st.spinner('V11 in esecuzione: una passata per ciascuna delle 5 leghe...'):
                per,dfs12,dfsou,errors=_v11_build_all(int(wu))
            for merc,dfs in [('1X2',dfs12),('O/U 2.5',dfsou)]:
                for common in [False,True]:
                    titolo=f'{merc} — {"COMMON BETS" if common else "ALL OOS"}'
                    st.markdown(f'### {titolo}')
                    agg=_v11_aggregate_band_tables(dfs,merc,common)
                    if not agg.empty:
                        st.dataframe(agg,use_container_width=True,hide_index=True)
                        st.download_button(f'⬇️ Scarica V11 {merc} {"COMMON" if common else "ALL"}',data=agg.to_csv(index=False).encode('utf-8'),file_name=f'V11_{merc.replace("/","_")}_{"COMMON" if common else "ALL"}.csv',mime='text/csv',key=f'v11_dl_{merc}_{common}')
            if not per.empty:
                st.markdown('### Tabella completa per lega')
                st.dataframe(per,use_container_width=True,hide_index=True)
                st.download_button('⬇️ Scarica V11 completo',data=per.to_csv(index=False).encode('utf-8'),file_name='V11_diagnostica_calibrazione.csv',mime='text/csv',key='v11_dl_all')
            if errors:
                st.error('Campionati non completati:')
                for nome,msg in errors: st.write(f'- **{nome}**: {msg}')
        except Exception as e:
            st.error(f'Errore V11: {type(e).__name__}: {e}')

if mostra_diagnostica_v11:
    try:
        mostra_v11_validation()
    except Exception as _v11_err:
        st.error(f"V11 non disponibile: {type(_v11_err).__name__}: {_v11_err}")


# =====================================================================
# 🧪 V14 — AUDIT VALUE + COMBO
# Diagnostica separata: NON modifica il motore operativo V11/V13.
# 1) Value O/U 2.5: confronta RAW vs PLATT+guardia usando prezzo medio
#    grezzo dei bookmaker e probabilita mercato normalizzata bookmaker-per-bookmaker.
# 2) Combo: valuta le 12 probabilita congiunte direttamente dalla griglia
#    di punteggio, senza moltiplicare probabilita come se fossero indipendenti.
#    Nessuna calibrazione per-combo salvata viene usata in questo audit.
# =====================================================================

def _v14_build_ou_hist(dati, rho, ewma_span, emivita):
    if dati is None or dati.empty or 'FTHG' not in dati.columns or 'FTAG' not in dati.columns:
        return []
    tutte = dati[dati['FTHG'].notna() & dati['FTAG'].notna()].reset_index(drop=True)
    hist = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        try:
            m = calcola_modello_completo(
                tutte.iloc[:i], partita['HomeTeam'], partita['AwayTeam'],
                rho, ewma_span, emivita, pd.DataFrame(),
                data_riferimento=partita.get('Date_parsed')
            )
            if m is None or 'prob_under' not in m or 2.5 not in m['prob_under']:
                continue
            p_over = float(np.clip(1.0 - float(m['prob_under'][2.5]) / 100.0, 0.001, 0.999))
            hist.append({
                'i': i,
                'data': partita.get('Date_parsed'),
                'home': partita['HomeTeam'],
                'away': partita['AwayTeam'],
                'p_raw': p_over,
                'y_over': int(float(partita['FTHG']) + float(partita['FTAG']) > 2.5),
                'row': partita,
            })
        except Exception:
            continue
    return hist


def _v14_stats(bets):
    if not bets:
        return {'bets': 0, 'wins': 0, 'strike': np.nan, 'profit': 0.0, 'roi': np.nan,
                'avg_ev': np.nan, 'avg_edge_pp': np.nan, 'avg_odds': np.nan}
    n = len(bets)
    wins = sum(int(b['win']) for b in bets)
    profit = sum(float(b['profit']) for b in bets)
    return {
        'bets': n,
        'wins': wins,
        'strike': 100.0 * wins / n,
        'profit': profit,
        'roi': 100.0 * profit / n,
        'avg_ev': 100.0 * float(np.mean([b['ev'] for b in bets])),
        'avg_edge_pp': 100.0 * float(np.mean([b['edge'] for b in bets])),
        'avg_odds': float(np.mean([b['quota_media_book'] for b in bets])),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def _v14_value_leg(id_fd, rho, ewma_span, emivita, n_train=100, solo_corrente=False):
    dati = carica_dati_campionato(id_fd)
    hist = _v14_build_ou_hist(dati, rho, ewma_span, emivita)
    if len(hist) <= n_train:
        return None, pd.DataFrame()
    tutte = dati[dati['FTHG'].notna() & dati['FTAG'].notna()].reset_index(drop=True)
    cols_over, cols_under = classifica_colonne_over_under(tutte.columns, '2.5')
    if not cols_over or not cols_under:
        return None, pd.DataFrame()

    test = [(j, hist[j]) for j in range(n_train, len(hist))]
    if solo_corrente:
        test = [(j, h) for j, h in test if str(h['row'].get('Stagione','')) == 'corrente']

    details = []
    applied = excluded = 0
    for j, h in test:
        if j < n_train:
            continue
        train_df = pd.DataFrame([{'p_raw': x['p_raw'], 'y_over': x['y_over']}
                                 for x in hist[j-n_train:j]])
        cal, coef, reason = _v13_fit_platt_ou(train_df)
        p_raw = h['p_raw']
        if cal is not None:
            p_platt = _v13_predict_platt_ou(p_raw, cal)
            applied += 1
        else:
            p_platt = p_raw
            excluded += 1

        qfair = _audit_quote_ou(h['row'], cols_over, cols_under)
        qraw = quote_medie_grezze_ou(h['row'], cols_over, cols_under)
        if qfair is None or qraw is None:
            continue

        p_market = qfair['p_market_norm_media_bookmaker']
        for strategy, p_over in [('RAW', p_raw), ('PLATT', p_platt)]:
            probs = {'Over': p_over, 'Under': 1.0 - p_over}
            scelta = max(probs, key=probs.get)
            p_sel = float(probs[scelta])
            odds = float(qraw[scelta])
            ev = p_sel * odds - 1.0
            edge = p_sel - float(p_market[scelta])
            esito = 'Over' if h['y_over'] else 'Under'
            details.append({
                'Campionato': next((k for k,v in CAMPIONATI_DOMESTICI.items() if v['id_fd']==id_fd), id_fd),
                'data': h['data'], 'casa': h['home'], 'trasferta': h['away'],
                'strategy': strategy, 'scelta': scelta, 'prob': p_sel,
                'prob_over_raw': p_raw, 'prob_over_platt': p_platt,
                'quota_media_book': odds, 'p_market_fair': float(p_market[scelta]),
                'edge': edge, 'ev': ev, 'esito': esito,
                'win': scelta == esito, 'profit': odds - 1.0 if scelta == esito else -1.0,
                'platt_applied': bool(cal is not None), 'coef': coef if cal is not None else np.nan,
                'platt_reason': reason,
            })

    detail = pd.DataFrame(details)
    if detail.empty:
        return None, detail
    rows = []
    for soglia_pct in [0, 5, 10]:
        thr = soglia_pct / 100.0
        for strategy in ['RAW', 'PLATT']:
            d = detail[(detail['strategy'] == strategy) & (detail['ev'] >= thr)].copy()
            s = _v14_stats(d.to_dict('records'))
            rows.append({
                'Campionato': next((k for k,v in CAMPIONATI_DOMESTICI.items() if v['id_fd']==id_fd), id_fd),
                'EV minimo %': soglia_pct, 'Strategia': strategy,
                'Bet': s['bets'], 'Strike %': s['strike'], 'Profit': s['profit'], 'ROI %': s['roi'],
                'EV medio %': s['avg_ev'], 'Edge medio pp': s['avg_edge_pp'], 'Quota media': s['avg_odds'],
            })
    summary = pd.DataFrame(rows)
    meta = {
        'test_opportunities': len(test),
        'quoted_opportunities': int(detail[['data','casa','trasferta']].drop_duplicates().shape[0] / 2),
        'platt_applied_n': applied,
        'platt_excluded_n': excluded,
        'platt_applied_pct': 100.0 * applied / max(1, len(test)),
    }
    return {'summary': summary, 'meta': meta}, detail


@st.cache_data(show_spinner=False, ttl=3600)
def _v14_combo_leg(id_fd, rho, ewma_span, emivita, solo_corrente=False):
    dati = carica_dati_campionato(id_fd)
    if dati is None or dati.empty:
        return None
    tutte = dati[dati['FTHG'].notna() & dati['FTAG'].notna()].reset_index(drop=True)
    rows = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        if solo_corrente and str(partita.get('Stagione','')) != 'corrente':
            continue
        try:
            m = calcola_modello_completo(
                tutte.iloc[:i], partita['HomeTeam'], partita['AwayTeam'],
                rho, ewma_span, emivita, pd.DataFrame(),
                data_riferimento=partita.get('Date_parsed')
            )
            if m is None or 'griglia' not in m:
                continue
            probs = {}
            for s in ['1','X','2']:
                for g in ['Goal','NoGoal']:
                    for t in ['Over','Under']:
                        key = f'{s}_{g}_{t}'
                        probs[key] = float(calcola_combo_libera(
                            m['griglia'], segno=s, soglia_gol=2.5, tipo_soglia=t, gol_nogol=g
                        ) / 100.0)
            esito = '1' if float(partita['FTHG']) > float(partita['FTAG']) else ('2' if float(partita['FTHG']) < float(partita['FTAG']) else 'X')
            gg = 'Goal' if float(partita['FTHG']) > 0 and float(partita['FTAG']) > 0 else 'NoGoal'
            tt = 'Over' if float(partita['FTHG']) + float(partita['FTAG']) > 2.5 else 'Under'
            true_key = f'{esito}_{gg}_{tt}'
            if true_key not in probs:
                continue
            order = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            top1_key, top1_p = order[0]
            top2_p = float(order[1][1]) if len(order) > 1 else np.nan
            top4_keys = {k for k,_ in order[:4]}
            # Le 12 classi sono mutuamente esclusive ed esaustive.
            brier = sum((p - (1.0 if k == true_key else 0.0))**2 for k,p in probs.items())
            logloss = -np.log(np.clip(probs[true_key], 1e-9, 1.0))
            rows.append({
                'data': partita.get('Date_parsed'),
                'casa': partita['HomeTeam'], 'trasferta': partita['AwayTeam'],
                'true_combo': true_key, 'top1_combo': top1_key,
                'top1_prob': top1_p, 'top2_prob': top2_p,
                'top1_hit': top1_key == true_key,
                'top4_hit': true_key in top4_keys,
                'brier': brier, 'logloss': logloss,
                'true_combo_prob': float(probs[true_key]),
                **{f'prob_{k}': float(v) for k, v in probs.items()},
            })
        except Exception:
            continue
    if not rows:
        return None, pd.DataFrame()
    d = pd.DataFrame(rows)
    summary = {
        'test_n': len(d),
        'top1_accuracy_pct': 100.0 * d['top1_hit'].mean(),
        'top4_hit_rate_pct': 100.0 * d['top4_hit'].mean(),
        'mean_top1_prob_pct': 100.0 * d['top1_prob'].mean(),
        'brier_12class': float(d['brier'].mean()),
        'logloss_12class': float(d['logloss'].mean()),
    }
    return summary, d


def mostra_v14_value_combo_audit():
    st.divider()
    st.markdown('## 🧪 V14 — AUDIT VALUE + COMBO')
    st.caption('Diagnostica separata dall’operativa. Il Value usa probabilità RAW/PLATT e prezzi medi bookmaker; le Combo usano direttamente la griglia congiunta delle 12 classi, senza calibrazione per-combo salvata.')
    with st.expander('Apri V14 — Value + Combo Audit', expanded=False):
        c1,c2,c3 = st.columns(3)
        with c1:
            periodo = st.selectbox('Periodo V14', ['Tutto lo storico','Stagione corrente OOS'], key='v14_periodo')
        with c2:
            ntrain = st.number_input('Training PLATT V14', 50, 200, 100, 10, key='v14_train')
        with c3:
            solo_corrente = periodo == 'Stagione corrente OOS'
            st.write('')
            st.caption('Soglie Value: 0%, 5%, 10%')

        if st.button('🔬 ESEGUI V14 — VALUE + COMBO', key='v14_run'):
            value_tables=[]; combo_tables=[]; value_details=[]; combo_details=[]; errors=[]
            with st.spinner('V14 in esecuzione sulle 5 leghe...'):
                for camp,info in CAMPIONATI_DOMESTICI.items():
                    try:
                        v,dv = _v14_value_leg(info['id_fd'],rho_val,ewma_span_val,emivita_val,int(ntrain),solo_corrente)
                        if v is not None:
                            value_tables.append(v['summary'])
                            if dv is not None and not dv.empty:
                                value_details.append(dv)
                        c,dc = _v14_combo_leg(info['id_fd'],rho_val,ewma_span_val,emivita_val,solo_corrente)
                        if c is not None:
                            combo_tables.append({'Campionato':camp,**c})
                            if dc is not None and not dc.empty:
                                combo_details.append(dc.assign(Campionato=camp))
                    except Exception as e:
                        errors.append((camp,f'{type(e).__name__}: {e}'))

            if value_tables:
                st.markdown('### 1. VALUE O/U 2.5 — risultati per lega')
                vdf = pd.concat(value_tables, ignore_index=True)
                st.dataframe(vdf.round(4), use_container_width=True, hide_index=True)
                st.caption('EV usa la quota media grezza dei bookmaker presenti nel CSV. Edge medio = probabilità modello − probabilità di mercato fair normalizzata bookmaker-per-bookmaker.')
                if value_details:
                    vd = pd.concat(value_details, ignore_index=True)
                    st.download_button('⬇️ Scarica V14 Value dettaglio CSV',data=vd.to_csv(index=False).encode('utf-8'),file_name='V14_value_ou25_dettaglio.csv',mime='text/csv',key='v14_value_dl')
                    agg=[]
                    for thr in [0,5,10]:
                        d=vd[vd['ev'] >= thr/100.0].copy()
                        # stesso universo comune per confronto RAW vs PLATT
                        keys=['Campionato','data','casa','trasferta']
                        cnt=d.groupby(keys)['strategy'].nunique()
                        common_keys=cnt[cnt==2].index
                        common=d.set_index(keys).loc[common_keys].reset_index() if len(common_keys) else pd.DataFrame()
                        if not common.empty:
                            w=common.pivot_table(index=keys,columns='strategy',values=['profit','win','ev','prob','scelta'],aggfunc='first')
                            w.columns=[f'{a}_{b}' for a,b in w.columns]
                            w=w.reset_index()
                            delta=float((w['profit_PLATT']-w['profit_RAW']).mean())
                            agg.append({'EV minimo %':thr,'Common bets':len(w),'RAW ROI %':100*w['profit_RAW'].mean(),'PLATT ROI %':100*w['profit_PLATT'].mean(),'Δ ROI pp':100*delta,'RAW strike %':100*w['win_RAW'].mean(),'PLATT strike %':100*w['win_PLATT'].mean()})
                    if agg:
                        st.markdown('### 2. VALUE — confronto paired sulle stesse occasioni')
                        st.dataframe(pd.DataFrame(agg).round(4),use_container_width=True,hide_index=True)

            if combo_tables:
                st.markdown('### 3. COMBO — qualità OOS delle 12 probabilità congiunte')
                cdf=pd.DataFrame(combo_tables)
                st.dataframe(cdf.round(4),use_container_width=True,hide_index=True)
                st.caption('Top 1 = combo con probabilità più alta. Top 4 = vera combo presente nelle prime quattro. Brier e Log Loss sono calcolati sulle 12 classi congiunte, che sono mutuamente esclusive.')
                if combo_details:
                    cd=pd.concat(combo_details,ignore_index=True)
                    st.download_button('⬇️ Scarica V14 Combo dettaglio CSV',data=cd.to_csv(index=False).encode('utf-8'),file_name='V14_combo_12classi_dettaglio.csv',mime='text/csv',key='v14_combo_dl')

            if errors:
                st.warning('Campionati non completati:')
                for nome,msg in errors:
                    st.write(f'- **{nome}**: {msg}')


# =====================================================================
# 🧪 V14.1 — AUDIT CALIBRAZIONE COMBO (SOLO DIAGNOSTICO)
# Misura la calibrazione della probabilità assegnata alla Top 1 combo:
# probabilità media prevista vs frequenza reale di Top 1 corretta.
# NON modifica il motore operativo e NON ricalibra le combo.
# =====================================================================

def _v14_1_calibration_top1(df):
    if df is None or df.empty or 'top1_prob' not in df.columns or 'top1_hit' not in df.columns:
        return pd.DataFrame(), {'ece_pp': np.nan, 'n': 0}

    x = df.copy()
    bins = [0.0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.000001]
    labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    x['fascia_top1'] = pd.cut(
        x['top1_prob'].astype(float), bins=bins, labels=labels,
        right=False, include_lowest=True
    )
    rows = []
    for lab, g in x.groupby('fascia_top1', observed=False):
        if g.empty:
            continue
        prob = float(g['top1_prob'].mean())
        freq = float(g['top1_hit'].mean())
        rows.append({
            'Fascia Top 1': str(lab),
            'N': int(len(g)),
            'Prob. media prevista %': 100.0 * prob,
            'Frequenza reale %': 100.0 * freq,
            'Errore calibrazione pp': 100.0 * (freq - prob),
        })

    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab, {'ece_pp': np.nan, 'n': len(x)}

    # ECE: errore assoluto medio pesato per numerosità.
    n = len(x)
    tab['peso'] = tab['N'] / n
    ece = float((tab['peso'] * tab['Errore calibrazione pp'].abs()).sum())
    tab = tab.drop(columns=['peso'])
    return tab, {'ece_pp': ece, 'n': n}


def mostra_v14_1_combo_calibrazione(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.1 — CALIBRAZIONE COMBO')
    st.caption('Audit della Top 1: confronta la probabilità media prevista con la frequenza reale della Top 1. Solo diagnostico: non modifica le probabilità operative.')
    with st.expander('Apri V14.1 — Audit calibrazione Combo', expanded=False):
        periodo = st.selectbox(
            'Periodo V14.1',
            ['Tutto lo storico', 'Stagione corrente OOS'],
            key='v14_1_periodo'
        )
        solo_corrente = periodo == 'Stagione corrente OOS'

        if st.button('🔬 ESEGUI V14.1 — CALIBRAZIONE COMBO', key='v14_1_run'):
            tutti = []
            errori = []
            with st.spinner('Calcolo calibrazione Top 1 sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        res, det = _v14_combo_leg(
                            info['id_fd'], rho, ewma_span, emivita, solo_corrente
                        )
                        if det is not None and not det.empty:
                            tutti.append(det.assign(Campionato=camp))
                    except Exception as e:
                        errori.append((camp, f'{type(e).__name__}: {e}'))

            if tutti:
                all_df = pd.concat(tutti, ignore_index=True)
                tab, met = _v14_1_calibration_top1(all_df)

                st.markdown('### 1. Calibrazione aggregata — Top 1')
                st.write(
                    f"**Partite:** {met['n']} · **ECE:** {met['ece_pp']:.2f} pp "
                    f"(errore assoluto medio pesato sulle fasce)"
                )
                st.dataframe(tab.round(4), use_container_width=True, hide_index=True)
                st.caption('Errore calibrazione pp = frequenza reale − probabilità media prevista. Valori negativi indicano sovrastima della probabilità.')

                st.markdown('### 2. Calibrazione per campionato')
                rows = []
                for camp, g in all_df.groupby('Campionato'):
                    t, m = _v14_1_calibration_top1(g)
                    for _, r in t.iterrows():
                        rows.append({
                            'Campionato': camp,
                            'Fascia Top 1': r['Fascia Top 1'],
                            'N': r['N'],
                            'Prob. media prevista %': r['Prob. media prevista %'],
                            'Frequenza reale %': r['Frequenza reale %'],
                            'Errore calibrazione pp': r['Errore calibrazione pp'],
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)

                st.markdown('### 3. Dettaglio OOS')
                st.dataframe(
                    all_df[['Campionato','data','casa','trasferta','true_combo','top1_combo','top1_prob','top1_hit','top4_hit']]
                    .round(6),
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    '⬇️ Scarica V14.1 calibrazione Combo CSV',
                    data=all_df.to_csv(index=False).encode('utf-8'),
                    file_name='V14_1_combo_calibrazione_top1.csv',
                    mime='text/csv',
                    key='v14_1_combo_dl'
                )

            if errori:
                st.warning('Campionati non completati:')
                for nome, msg in errori:
                    st.write(f'- **{nome}**: {msg}')

mostra_v14_value_combo_audit()

mostra_v14_1_combo_calibrazione(rho_val, ewma_span_val, emivita_val)

# =====================================================================
# 🧪 V14.2 — AUDIT DELLE 12 COMBO PER CLASSE (SOLO DIAGNOSTICO)
# Per ogni classe misura la calibrazione one-vs-rest:
#   - frequenza reale della classe;
#   - probabilità media prevista dal modello;
#   - errore = reale - previsto.
# Inoltre mostra supporto reale e quante volte la classe è stata scelta come Top 1.
# NON modifica il motore operativo e NON ricalibra le Combo.
# =====================================================================

def _v14_2_combo_per_class(df):
    if df is None or df.empty:
        return pd.DataFrame(), {'n': 0, 'mae_pp': np.nan, 'max_abs_error_pp': np.nan}

    combo_keys = [f'{s}_{g}_{t}'
                  for s in ['1','X','2']
                  for g in ['Goal','NoGoal']
                  for t in ['Over','Under']]
    rows = []
    n = len(df)
    for key in combo_keys:
        pcol = f'prob_{key}'
        if pcol not in df.columns:
            continue
        y = (df['true_combo'].astype(str) == key).astype(float)
        p = pd.to_numeric(df[pcol], errors='coerce')
        ok = p.notna()
        if not ok.any():
            continue
        p = p[ok]
        y = y[ok]
        pred_rate = float(p.mean())
        real_rate = float(y.mean())
        err_pp = 100.0 * (real_rate - pred_rate)
        top1_sel = int((df.loc[ok, 'top1_combo'].astype(str) == key).sum())
        top1_hit = int(((df.loc[ok, 'top1_combo'].astype(str) == key) & (df.loc[ok, 'true_combo'].astype(str) == key)).sum())
        true_n = int(y.sum())
        rows.append({
            'Combo': key,
            'N totale': int(len(p)),
            'N esiti reali': true_n,
            'Frequenza reale %': 100.0 * real_rate,
            'Prob. media prevista %': 100.0 * pred_rate,
            'Errore calibrazione pp': err_pp,
            'Prob. media quando reale %': 100.0 * float(p[y > 0.5].mean()) if true_n else np.nan,
            'Top 1 selezionata N': top1_sel,
            'Top 1 corretta %': 100.0 * top1_hit / true_n if true_n else np.nan,
        })

    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab, {'n': n, 'mae_pp': np.nan, 'max_abs_error_pp': np.nan}
    mae = float(tab['Errore calibrazione pp'].abs().mean())
    max_abs = float(tab['Errore calibrazione pp'].abs().max())
    return tab, {'n': n, 'mae_pp': mae, 'max_abs_error_pp': max_abs}


def mostra_v14_2_combo_per_class(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.2 — AUDIT DELLE 12 COMBO')
    st.caption('Calibrazione one-vs-rest per ciascuna delle 12 classi. Solo diagnostico: non modifica le probabilità operative.')
    with st.expander('Apri V14.2 — Audit 12 Combo per classe', expanded=False):
        periodo = st.selectbox(
            'Periodo V14.2',
            ['Tutto lo storico', 'Stagione corrente OOS'],
            key='v14_2_periodo'
        )
        solo_corrente = periodo == 'Stagione corrente OOS'

        if st.button('🔬 ESEGUI V14.2 — AUDIT 12 COMBO', key='v14_2_run'):
            tutti = []
            errori = []
            with st.spinner('Calcolo calibrazione delle 12 Combo sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        res, det = _v14_combo_leg(
                            info['id_fd'], rho, ewma_span, emivita, solo_corrente
                        )
                        if det is not None and not det.empty:
                            tutti.append(det.assign(Campionato=camp))
                    except Exception as e:
                        errori.append((camp, f'{type(e).__name__}: {e}'))

            if tutti:
                all_df = pd.concat(tutti, ignore_index=True)
                tabs = []
                for camp, g in all_df.groupby('Campionato'):
                    t, m = _v14_2_combo_per_class(g)
                    if not t.empty:
                        t.insert(0, 'Campionato', camp)
                        tabs.append(t)

                if tabs:
                    out = pd.concat(tabs, ignore_index=True)
                    agg, met = _v14_2_combo_per_class(all_df)

                    st.markdown('### 1. Aggregato — tutte le 12 Combo')
                    st.write(
                        f"**Partite:** {met['n']} · **MAE calibrazione:** {met['mae_pp']:.2f} pp · "
                        f"**Errore massimo assoluto:** {met['max_abs_error_pp']:.2f} pp"
                    )
                    st.dataframe(agg.round(4), use_container_width=True, hide_index=True)
                    st.caption('Errore calibrazione pp = frequenza reale − probabilità media prevista. Negativo = sovrastima; positivo = sottostima.')

                    st.markdown('### 2. Per campionato')
                    st.dataframe(out.round(4), use_container_width=True, hide_index=True)

                    st.markdown('### 3. Dettaglio OOS')
                    cols = ['Campionato','data','casa','trasferta','true_combo','top1_combo','top1_prob','true_combo_prob','top1_hit','top4_hit']
                    extra = [c for c in all_df.columns if c.startswith('prob_')]
                    st.dataframe(
                        all_df[cols + extra].round(6),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.download_button(
                        '⬇️ Scarica V14.2 audit 12 Combo CSV',
                        data=out.to_csv(index=False).encode('utf-8'),
                        file_name='V14_2_combo_per_class.csv',
                        mime='text/csv',
                        key='v14_2_combo_dl'
                    )

            if errori:
                st.warning('Campionati non completati:')
                for nome, msg in errori:
                    st.write(f'- **{nome}**: {msg}')


mostra_v14_2_combo_per_class(rho_val, ewma_span_val, emivita_val)



# =====================================================================
# 🧪 V14.3 — AUDIT DEL RANKING DELLE 12 COMBO (SOLO DIAGNOSTICO)
# Verifica se il ranking delle 12 combo contiene informazione oltre la Top 1.
# Per ogni partita ordina le 12 probabilita' dalla piu' alta alla piu' bassa
# e misura: hit@1..hit@5, probabilita' media assegnata alla combo nella
# posizione, rank medio della combo realmente verificata e margine Top1-Top2.
# NON modifica il motore operativo e NON calibra le combo.
# =====================================================================

def _v14_3_ranking_leg(id_fd, rho, ewma_span, emivita, solo_corrente=False):
    dati = carica_dati_campionato(id_fd)
    if dati is None or dati.empty:
        return pd.DataFrame()
    tutte = dati[dati['FTHG'].notna() & dati['FTAG'].notna()].reset_index(drop=True)
    rows = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        if solo_corrente and str(partita.get('Stagione','')) != 'corrente':
            continue
        try:
            m = calcola_modello_completo(
                tutte.iloc[:i], partita['HomeTeam'], partita['AwayTeam'],
                rho, ewma_span, emivita, pd.DataFrame(),
                data_riferimento=partita.get('Date_parsed')
            )
            if m is None or 'griglia' not in m:
                continue

            probs = {}
            for s in ['1','X','2']:
                for g in ['Goal','NoGoal']:
                    for t in ['Over','Under']:
                        key = f'{s}_{g}_{t}'
                        probs[key] = float(np.clip(
                            calcola_combo_libera(
                                m['griglia'], segno=s, soglia_gol=2.5,
                                tipo_soglia=t, gol_nogol=g
                            ) / 100.0,
                            0.0, 1.0
                        ))

            esito = '1' if float(partita['FTHG']) > float(partita['FTAG']) else (
                '2' if float(partita['FTHG']) < float(partita['FTAG']) else 'X'
            )
            gg = 'Goal' if float(partita['FTHG']) > 0 and float(partita['FTAG']) > 0 else 'NoGoal'
            tt = 'Over' if float(partita['FTHG']) + float(partita['FTAG']) > 2.5 else 'Under'
            true_key = f'{esito}_{gg}_{tt}'
            if true_key not in probs:
                continue

            order = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            rank_map = {k: r for r, (k, _) in enumerate(order, start=1)}
            prob_map = dict(order)
            true_rank = int(rank_map[true_key])
            top1_p = float(order[0][1])
            top2_p = float(order[1][1])

            row = {
                'data': partita.get('Date_parsed'),
                'casa': partita['HomeTeam'],
                'trasferta': partita['AwayTeam'],
                'true_combo': true_key,
                'true_rank': true_rank,
                'true_combo_prob': float(prob_map[true_key]),
                'top1_prob': top1_p,
                'top2_prob': top2_p,
                'margin_top1_top2': top1_p - top2_p,
                'hit_at_1': true_rank <= 1,
                'hit_at_2': true_rank <= 2,
                'hit_at_3': true_rank <= 3,
                'hit_at_4': true_rank <= 4,
                'hit_at_5': true_rank <= 5,
                'ranked_keys': '|'.join(k for k, _ in order),
            }
            for rr, (_, pp) in enumerate(order[:5], start=1):
                row[f'rank{rr}_prob'] = float(pp)
                row[f'rank{rr}_combo'] = order[rr-1][0]
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def _v14_3_ranking_summary(df):
    if df is None or df.empty:
        return pd.DataFrame(), {}
    rows = []
    n = len(df)
    for k in range(1, 6):
        pcol = f'rank{k}_prob'
        rows.append({
            'Posizione': f'Top {k}',
            'N': int(n),
            'Hit rate cumulativo %': 100.0 * float(df[f'hit_at_{k}'].mean()),
            'Prob. media posizione %': 100.0 * float(df[pcol].mean()) if pcol in df.columns else np.nan,
        })
    mean_rank=float(df['true_rank'].mean())
    mrr=float(np.mean(1.0/df['true_rank'].astype(float)))
    return pd.DataFrame(rows), {
        'n': n,
        'mean_true_rank': mean_rank,
        'mrr': mrr,
        'top1_top2_margin_pp': 100.0 * float(df['margin_top1_top2'].mean()),
        'top1_prob_pp': 100.0 * float(df['top1_prob'].mean()),
    }


def mostra_v14_3_combo_ranking(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.3 — AUDIT RANKING DELLE 12 COMBO')
    st.caption('Misura quanto il ranking delle 12 combo informa oltre la sola Top 1. Solo diagnostico: nessuna modifica alle probabilita operative.')
    with st.expander('Apri V14.3 — Ranking Combo', expanded=False):
        periodo = st.selectbox(
            'Periodo V14.3',
            ['Tutto lo storico', 'Stagione corrente OOS'],
            key='v14_3_periodo'
        )
        solo_corrente = periodo == 'Stagione corrente OOS'

        if st.button('🔬 ESEGUI V14.3 — RANKING COMBO', key='v14_3_run'):
            ranking_all=[]
            errors=[]
            full_all=[]
            with st.spinner('Calcolo ranking Top 1 → Top 5 sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        d = _v14_3_ranking_leg(
                            info['id_fd'], rho, ewma_span, emivita, solo_corrente
                        )
                        if d is not None and not d.empty:
                            d=d.copy(); d.insert(0,'Campionato',camp)
                            ranking_all.append(d)
                    except Exception as e:
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            if ranking_all:
                all_rank=pd.concat(ranking_all, ignore_index=True)
                summary_rows=[]
                for camp,g in all_rank.groupby('Campionato'):
                    s,met=_v14_3_ranking_summary(g)
                    for _,r in s.iterrows():
                        summary_rows.append({
                            'Campionato':camp,
                            'Posizione':r['Posizione'],
                            'N':int(r['N']),
                            'Hit rate cumulativo %':r['Hit rate cumulativo %'],
                            'Prob. media posizione %':r['Prob. media posizione %'],
                        })
                agg_s,agg_m=_v14_3_ranking_summary(all_rank)

                st.markdown('### 1. Ranking aggregato')
                st.write(
                    f"**Partite:** {agg_m['n']} · **Rank medio combo reale:** {agg_m['mean_true_rank']:.2f} · "
                    f"**MRR:** {agg_m['mrr']:.4f} · **Margine medio Top1−Top2:** {agg_m['top1_top2_margin_pp']:.2f} pp"
                )
                st.dataframe(agg_s.round(4), use_container_width=True, hide_index=True)

                st.markdown('### 2. Per campionato')
                st.dataframe(pd.DataFrame(summary_rows).round(4), use_container_width=True, hide_index=True)

                st.markdown('### 3. Distribuzione del rank reale')
                dist=[]
                for camp,g in all_rank.groupby('Campionato'):
                    vc=g['true_rank'].value_counts().reindex(range(1,13),fill_value=0)
                    for rank,count in vc.items():
                        dist.append({'Campionato':camp,'Rank combo reale':int(rank),'N':int(count),'%':100.0*count/len(g)})
                dist_df=pd.DataFrame(dist)
                st.dataframe(dist_df.round(4),use_container_width=True,hide_index=True)

                st.markdown('### 4. Margine Top 1 − Top 2')
                margin=[]
                for camp,g in all_rank.groupby('Campionato'):
                    margin.append({
                        'Campionato':camp,
                        'N':len(g),
                        'Margine medio pp':100.0*float(g['margin_top1_top2'].mean()),
                        'Mediana pp':100.0*float(g['margin_top1_top2'].median()),
                    })
                st.dataframe(pd.DataFrame(margin).round(4),use_container_width=True,hide_index=True)

                st.markdown('### 5. Dettaglio OOS')
                st.dataframe(
                    all_rank[['Campionato','data','casa','trasferta','true_combo','true_rank','true_combo_prob','top1_prob','top2_prob','rank3_prob','rank4_prob','rank5_prob','margin_top1_top2','hit_at_1','hit_at_2','hit_at_3','hit_at_4','hit_at_5']].round(6),
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    '⬇️ Scarica V14.3 audit ranking CSV',
                    data=all_rank.to_csv(index=False).encode('utf-8'),
                    file_name='V14_3_combo_ranking.csv',
                    mime='text/csv',
                    key='v14_3_dl'
                )

            if errors:
                st.warning('Campionati non completati:')
                for nome,msg in errors:
                    st.write(f'- **{nome}**: {msg}')


mostra_v14_3_combo_ranking(rho_val, ewma_span_val, emivita_val)


# =====================================================================
# 🧪 V14.4 — AUDIT CONFIDENCE / MARGINE TOP 1 (SOLO DIAGNOSTICO)
# Divide le partite per il margine Top1−Top2 e misura:
#   - frequenza di Top1 corretta;
#   - frequenza della combo reale dentro Top4;
#   - probabilità media Top1;
#   - numero di osservazioni.
# NON modifica il motore operativo e NON cambia le probabilità.
# =====================================================================

def _v14_4_margin_audit(df):
    if df is None or df.empty or 'margin_top1_top2' not in df.columns:
        return pd.DataFrame()

    x = df.copy()
    x['margin_pp'] = 100.0 * pd.to_numeric(x['margin_top1_top2'], errors='coerce')
    x = x.dropna(subset=['margin_pp'])

    bins = [-np.inf, 2.0, 5.0, 10.0, np.inf]
    labels = ['< 2 pp', '2–5 pp', '5–10 pp', '> 10 pp']
    x['fascia_margine'] = pd.cut(
        x['margin_pp'], bins=bins, labels=labels,
        right=False, include_lowest=True
    )

    rows = []
    for lab, g in x.groupby('fascia_margine', observed=False):
        if g.empty:
            continue
        rows.append({
            'Margine Top1−Top2': str(lab),
            'N': int(len(g)),
            'Top 1 hit rate %': 100.0 * float(g['hit_at_1'].mean()),
            'Top 4 hit rate %': 100.0 * float(g['hit_at_4'].mean()),
            'Prob. media Top 1 %': 100.0 * float(g['top1_prob'].mean()),
            'Rank medio combo reale': float(g['true_rank'].mean()),
            'Margine medio pp': float(g['margin_pp'].mean()),
        })
    return pd.DataFrame(rows)


def mostra_v14_4_confidence_margin(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.4 — AUDIT CONFIDENCE / MARGINE COMBO')
    st.caption('Verifica quanto la qualità della Top 1 cambia quando il margine tra prima e seconda combo è piccolo o grande. Solo diagnostico: nessuna modifica alle probabilità operative.')
    with st.expander('Apri V14.4 — Confidence / Margin Audit', expanded=False):
        periodo = st.selectbox(
            'Periodo V14.4',
            ['Tutto lo storico', 'Stagione corrente OOS'],
            key='v14_4_periodo'
        )
        solo_corrente = periodo == 'Stagione corrente OOS'

        if st.button('🔬 ESEGUI V14.4 — CONFIDENCE / MARGINE', key='v14_4_run'):
            tutti = []
            errori = []
            with st.spinner('Calcolo confidence/margine sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        d = _v14_3_ranking_leg(
                            info['id_fd'], rho, ewma_span, emivita, solo_corrente
                        )
                        if d is not None and not d.empty:
                            tutti.append(d.assign(Campionato=camp))
                    except Exception as e:
                        errori.append((camp, f'{type(e).__name__}: {e}'))

            if tutti:
                all_df = pd.concat(tutti, ignore_index=True)

                st.markdown('### 1. Aggregato — qualità per margine')
                agg = _v14_4_margin_audit(all_df)
                st.dataframe(agg.round(4), use_container_width=True, hide_index=True)

                st.markdown('### 2. Per campionato')
                rows = []
                for camp, g in all_df.groupby('Campionato'):
                    t = _v14_4_margin_audit(g)
                    if not t.empty:
                        t.insert(0, 'Campionato', camp)
                        rows.append(t)
                if rows:
                    st.dataframe(pd.concat(rows, ignore_index=True).round(4), use_container_width=True, hide_index=True)

                st.markdown('### 3. Densità delle fasce')
                dens = all_df.copy()
                dens['Margine pp'] = 100.0 * dens['margin_top1_top2']
                dens['Fascia'] = pd.cut(
                    dens['Margine pp'], bins=[-np.inf, 2, 5, 10, np.inf],
                    labels=['< 2 pp', '2–5 pp', '5–10 pp', '> 10 pp'],
                    right=False, include_lowest=True
                )
                dens_tab = (
                    dens.groupby('Fascia', observed=False)
                    .size().rename('N').reset_index()
                )
                dens_tab['% partite'] = 100.0 * dens_tab['N'] / max(1, len(dens))
                st.dataframe(dens_tab.round(4), use_container_width=True, hide_index=True)

                st.markdown('### 4. Dettaglio OOS')
                cols = [
                    'Campionato','data','casa','trasferta','true_combo','true_rank',
                    'top1_prob','top2_prob','margin_top1_top2',
                    'hit_at_1','hit_at_4'
                ]
                st.dataframe(all_df[cols].round(6), use_container_width=True, hide_index=True)

                st.download_button(
                    '⬇️ Scarica V14.4 confidence / margin CSV',
                    data=agg.to_csv(index=False).encode('utf-8'),
                    file_name='V14_4_combo_confidence_margin.csv',
                    mime='text/csv',
                    key='v14_4_dl'
                )

            if errori:
                st.warning('Campionati non completati:')
                for nome, msg in errori:
                    st.write(f'- **{nome}**: {msg}')


mostra_v14_4_confidence_margin(rho_val, ewma_span_val, emivita_val)


# =====================================================================
# 🧪 V14.5 — AUDIT SELEZIONE TOP K COMBO (NO QUOTE INVENTATE)
# Misura l'efficienza della selezione Top 1/2/3/4:
#   - copertura reale cumulativa;
#   - massa di probabilità cumulativa delle prime K;
#   - rapporto copertura reale / probabilità cumulativa;
#   - rank medio della combo reale;
#   - frequenza della combo reale esattamente al rank K.
# Non calcola ROI combo perché il dataset storico non contiene quote eseguibili
# per le combo libere 1X2 + Goal/NoGoal + O/U 2.5. Nessuna quota sintetica viene
# usata per evitare di confondere un proxy con un prezzo bookmaker reale.
# =====================================================================

def _v14_5_selection_audit(df):
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    req = ['true_rank','top1_prob','top2_prob','rank3_prob','rank4_prob']
    if not all(c in x.columns for c in req):
        return pd.DataFrame()
    prob_cols = {1:'top1_prob',2:'top2_prob',3:'rank3_prob',4:'rank4_prob'}
    rows=[]
    for k in range(1,5):
        mass = sum(pd.to_numeric(x[prob_cols[j]], errors='coerce').fillna(0.0) for j in range(1,k+1))
        coverage = (pd.to_numeric(x['true_rank'], errors='coerce') <= k)
        n = int(coverage.notna().sum())
        cov_rate = float(coverage.mean())
        mass_mean = float(mass.mean())
        rows.append({
            'Selezione': f'Top {k}',
            'N': len(x),
            'Copertura reale %': 100.0 * cov_rate,
            'Prob. cumulativa media %': 100.0 * mass_mean,
            'Rapporto copertura/probabilità %': 100.0 * cov_rate / mass_mean if mass_mean > 0 else np.nan,
            'Rank medio combo reale': float(pd.to_numeric(x['true_rank'], errors='coerce').mean()),
        })
    return pd.DataFrame(rows)


def _v14_5_exact_rank(df):
    if df is None or df.empty or 'true_rank' not in df.columns:
        return pd.DataFrame()
    ranks=[]
    for k in range(1,10):
        n=int((df['true_rank']==k).sum())
        ranks.append({'Rank reale':k,'N':n,'% partite':100.0*n/len(df)})
    return pd.DataFrame(ranks)


def mostra_v14_5_combo_selection(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.5 — AUDIT SELEZIONE COMBO')
    st.caption('Confronta Top 1/2/3/4 senza inventare quote combo: misura quanta copertura reale otteniamo per quanta probabilità il modello concentra nelle prime K combinazioni.')
    with st.expander('Apri V14.5 — Selection Audit', expanded=False):
        periodo = st.selectbox(
            'Periodo V14.5',
            ['Tutto lo storico', 'Stagione corrente OOS'],
            key='v14_5_periodo'
        )
        solo_corrente = periodo == 'Stagione corrente OOS'

        if st.button('🔬 ESEGUI V14.5 — SELECTION AUDIT', key='v14_5_run'):
            tutti=[]; errori=[]
            with st.spinner('Calcolo efficienza Top K sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        d = _v14_3_ranking_leg(info['id_fd'], rho, ewma_span, emivita, solo_corrente)
                        if d is not None and not d.empty:
                            tutti.append(d.assign(Campionato=camp))
                    except Exception as e:
                        errori.append((camp, f'{type(e).__name__}: {e}'))

            if tutti:
                all_df=pd.concat(tutti,ignore_index=True)
                st.markdown('### 1. Aggregato — Top K')
                sel=_v14_5_selection_audit(all_df)
                st.dataframe(sel.round(4),use_container_width=True,hide_index=True)
                st.caption('Il rapporto copertura/probabilità è un indicatore descrittivo: >100% significa che la frequenza osservata supera la massa probabilistica media indicata per quella selezione.')

                st.markdown('### 2. Per campionato')
                rows=[]
                for camp,g in all_df.groupby('Campionato'):
                    t=_v14_5_selection_audit(g)
                    if not t.empty:
                        t.insert(0,'Campionato',camp)
                        rows.append(t)
                if rows:
                    st.dataframe(pd.concat(rows,ignore_index=True).round(4),use_container_width=True,hide_index=True)

                st.markdown('### 3. Rank esatto della combo reale')
                rank_rows=[]
                for camp,g in all_df.groupby('Campionato'):
                    t=_v14_5_exact_rank(g)
                    if not t.empty:
                        t.insert(0,'Campionato',camp)
                        rank_rows.append(t)
                if rank_rows:
                    st.dataframe(pd.concat(rank_rows,ignore_index=True).round(4),use_container_width=True,hide_index=True)

                st.markdown('### 4. Nota economica')
                st.info('Il dataset storico contiene quote pre-partita per 1X2 e O/U 2.5, ma non una quota eseguibile per ciascuna combo libera 1X2 + Goal/NoGoal + O/U. Per questo V14.5 non pubblica ROI combo sintetico: farlo richiederebbe introdurre un prezzo ipotetico.')

                st.download_button(
                    '⬇️ Scarica V14.5 selection audit CSV',
                    data=all_df.to_csv(index=False).encode('utf-8'),
                    file_name='V14_5_combo_selection_audit_dettaglio.csv',
                    mime='text/csv', key='v14_5_dl'
                )
            if errori:
                st.warning('Campionati non completati:')
                for nome,msg in errori:
                    st.write(f'- **{nome}**: {msg}')

mostra_v14_5_combo_selection(rho_val, ewma_span_val, emivita_val)


# =====================================================================
# 🧪 V14.6 — AUDIT SOGLIE CONFIDENZA COMBO (SOLO DIAGNOSTICO)
# Verifica descrittiva della qualità Top 1 quando imponiamo soglie
# predefinite su probabilità Top 1 e margine Top1-Top2.
# NON modifica le probabilità operative e NON propone automaticamente
# una soglia da usare in produzione.
# =====================================================================

def _v14_6_threshold_audit(df, prob_thresholds=(0.20,0.25,0.30,0.35), margin_thresholds=(0.0,0.02,0.05,0.10)):
    if df is None or df.empty:
        return pd.DataFrame()
    req = ['top1_prob','margin_top1_top2','hit_at_1','hit_at_4']
    if any(c not in df.columns for c in req):
        return pd.DataFrame()
    rows=[]
    for pt in prob_thresholds:
        for mt in margin_thresholds:
            d=df[(pd.to_numeric(df['top1_prob'],errors='coerce') >= pt) &
                 (pd.to_numeric(df['margin_top1_top2'],errors='coerce') >= mt)].copy()
            if d.empty:
                rows.append({
                    'Soglia prob. Top1 %':100*pt,
                    'Soglia margine pp':100*mt,
                    'N':0,
                    'Top1 hit %':np.nan,
                    'Top4 hit %':np.nan,
                    'Prob. media Top1 %':np.nan,
                    'Margine medio pp':np.nan,
                    'Copertura/Prob. cumulativa %':np.nan,
                })
                continue
            rows.append({
                'Soglia prob. Top1 %':100*pt,
                'Soglia margine pp':100*mt,
                'N':int(len(d)),
                'Top1 hit %':100*pd.to_numeric(d['hit_at_1'],errors='coerce').mean(),
                'Top4 hit %':100*pd.to_numeric(d['hit_at_4'],errors='coerce').mean(),
                'Prob. media Top1 %':100*pd.to_numeric(d['top1_prob'],errors='coerce').mean(),
                'Margine medio pp':100*pd.to_numeric(d['margin_top1_top2'],errors='coerce').mean(),
                'Copertura/Prob. cumulativa %':100*pd.to_numeric(d['hit_at_1'],errors='coerce').mean()/max(1e-9,pd.to_numeric(d['top1_prob'],errors='coerce').mean()),
            })
    return pd.DataFrame(rows)


def mostra_v14_6_confidence_threshold_audit(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.6 — AUDIT SOGLIE CONFIDENZA COMBO')
    st.caption('Test descrittivo di soglie predefinite su probabilità Top 1 e margine Top1−Top2. Solo diagnostico: nessuna soglia viene trasferita all’operativa.')
    with st.expander('Apri V14.6 — Confidence Threshold Audit', expanded=False):
        periodo = st.selectbox('Periodo V14.6',['Tutto lo storico','Stagione corrente OOS'],key='v14_6_periodo')
        solo_corrente = periodo == 'Stagione corrente OOS'
        if st.button('🔬 ESEGUI V14.6 — SOGLIE CONFIDENZA',key='v14_6_run'):
            tutti=[]; errori=[]
            with st.spinner('Calcolo soglie di confidenza sulle 5 leghe...'):
                for camp,info in CAMPIONATI_DOMESTICI.items():
                    try:
                        _,det=_v14_combo_leg(info['id_fd'],rho,ewma_span,emivita,solo_corrente)
                        if det is not None and not det.empty:
                            # V14.3 usa già top1_prob/top2_prob e il margine.
                            if 'margin_top1_top2' in det.columns and 'hit_at_1' in det.columns:
                                tutti.append(det.assign(Campionato=camp))
                            else:
                                x=det.copy()
                                if 'top2_prob' not in x.columns:
                                    # Compatibilità con dettagli legacy: ricava Top 2
                                    # dalle colonne prob_* escludendo la Top 1.
                                    prob_cols=[c for c in x.columns if c.startswith('prob_')]
                                    def _legacy_top2(row):
                                        vals=[]
                                        for c in prob_cols:
                                            try: vals.append(float(row[c]))
                                            except Exception: pass
                                        vals=sorted(vals, reverse=True)
                                        return vals[1] if len(vals)>1 else np.nan
                                    x['top2_prob']=x.apply(_legacy_top2,axis=1)
                                x['margin_top1_top2']=pd.to_numeric(x['top1_prob'],errors='coerce')-pd.to_numeric(x['top2_prob'],errors='coerce')
                                x['hit_at_1']=x['top1_hit']
                                x['hit_at_4']=x['top4_hit']
                                tutti.append(x.assign(Campionato=camp))
                    except Exception as e:
                        errori.append((camp,f'{type(e).__name__}: {e}'))
            if tutti:
                all_df=pd.concat(tutti,ignore_index=True)
                st.markdown('### 1. Aggregato — matrice probabilità × margine')
                tab=_v14_6_threshold_audit(all_df)
                st.dataframe(tab.round(4),use_container_width=True,hide_index=True)
                st.caption('Le soglie sono fissate ex ante nel codice e servono a descrivere la stabilità della selezione; non sono ottimizzate sui risultati.')
                st.markdown('### 2. Per campionato')
                rows=[]
                for camp,g in all_df.groupby('Campionato'):
                    t=_v14_6_threshold_audit(g)
                    t.insert(0,'Campionato',camp)
                    rows.append(t)
                if rows:
                    st.dataframe(pd.concat(rows,ignore_index=True).round(4),use_container_width=True,hide_index=True)
                st.markdown('### 3. Dettaglio OOS')
                st.dataframe(all_df[['Campionato','data','casa','trasferta','true_combo','top1_combo','top1_prob','top2_prob','margin_top1_top2','hit_at_1','hit_at_4']].round(6),use_container_width=True,hide_index=True)
                st.download_button('⬇️ Scarica V14.6 soglie confidenza CSV',data=tab.to_csv(index=False).encode('utf-8'),file_name='V14_6_confidence_threshold_audit.csv',mime='text/csv',key='v14_6_dl')
            if errori:
                st.warning('Campionati non completati:')
                for nome,msg in errori:
                    st.write(f'- **{nome}**: {msg}')

mostra_v14_6_confidence_threshold_audit(rho_val, ewma_span_val, emivita_val)


# =====================================================================
# 🧪 V14.7 — STABILITÀ SOGLIA TOP 1 >= 35% (SOLO DIAGNOSTICO)
# Verifica se una soglia fissata ex ante resta coerente tra:
#   1) tutto lo storico OOS;
#   2) stagione corrente OOS.
# Nessuna soglia viene trasferita all'operativa.
# =====================================================================

def _v14_7_wilson_interval(wins, n, z=1.959963984540054):
    if n <= 0:
        return (np.nan, np.nan)
    p = wins / n
    den = 1.0 + (z*z)/n
    center = (p + (z*z)/(2.0*n)) / den
    half = z * np.sqrt((p*(1.0-p)/n) + (z*z)/(4.0*n*n)) / den
    return center-half, center+half


def _v14_7_stability_summary(df, prob_threshold=0.35):
    if df is None or df.empty:
        return {
            'N': 0, 'Top1 hit %': np.nan, 'IC95% Top1 low %': np.nan,
            'IC95% Top1 high %': np.nan, 'Top4 hit %': np.nan,
            'Prob. media Top1 %': np.nan, 'Margine medio pp': np.nan,
            'Copertura/Prob. %': np.nan,
        }
    x = df.copy()
    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    # Compatibilità con _v14_combo_leg(), che restituisce top1_hit/top4_hit.
    if 'hit_at_1' not in x.columns and 'top1_hit' in x.columns:
        x['hit_at_1'] = x['top1_hit']
    if 'hit_at_4' not in x.columns and 'top4_hit' in x.columns:
        x['hit_at_4'] = x['top4_hit']
    if 'margin_top1_top2' not in x.columns and {'top1_prob','top2_prob'}.issubset(x.columns):
        x['margin_top1_top2'] = (
            pd.to_numeric(x['top1_prob'], errors='coerce')
            - pd.to_numeric(x['top2_prob'], errors='coerce')
        )
    x['hit_at_1'] = pd.to_numeric(x['hit_at_1'], errors='coerce')
    x['hit_at_4'] = pd.to_numeric(x['hit_at_4'], errors='coerce')
    x['margin_top1_top2'] = pd.to_numeric(x.get('margin_top1_top2'), errors='coerce')
    x = x[x['top1_prob'] >= prob_threshold].copy()
    x = x.dropna(subset=['hit_at_1','hit_at_4','top1_prob'])
    if x.empty:
        return {
            'N': 0, 'Top1 hit %': np.nan, 'IC95% Top1 low %': np.nan,
            'IC95% Top1 high %': np.nan, 'Top4 hit %': np.nan,
            'Prob. media Top1 %': np.nan, 'Margine medio pp': np.nan,
            'Copertura/Prob. %': np.nan,
        }
    n = len(x)
    wins = int(x['hit_at_1'].sum())
    lo, hi = _v14_7_wilson_interval(wins, n)
    mean_p = float(x['top1_prob'].mean())
    return {
        'N': int(n),
        'Top1 hit %': 100.0 * wins / n,
        'IC95% Top1 low %': 100.0 * lo,
        'IC95% Top1 high %': 100.0 * hi,
        'Top4 hit %': 100.0 * float(x['hit_at_4'].mean()),
        'Prob. media Top1 %': 100.0 * mean_p,
        'Margine medio pp': 100.0 * float(x['margin_top1_top2'].mean()) if 'margin_top1_top2' in x.columns else np.nan,
        'Copertura/Prob. %': 100.0 * float(x['hit_at_1'].mean()) / max(1e-9, mean_p),
    }


def mostra_v14_7_threshold_stability(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.7 — STABILITÀ SOGLIA TOP 1 ≥ 35%')
    st.caption('La soglia del 35% è fissata ex ante. Il test confronta tutto lo storico OOS con la sola stagione corrente OOS, senza modificare l’operativa.')
    with st.expander('Apri V14.7 — Threshold Stability', expanded=False):
        if st.button('🔬 ESEGUI V14.7 — STABILITÀ SOGLIA 35%', key='v14_7_run'):
            all_hist=[]; all_current=[]; errors=[]
            with st.spinner('Verifica stabilità della soglia ≥35% sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        # Tutto lo storico OOS
                        _, d_all = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                        if d_all is not None and not d_all.empty:
                            all_hist.append(d_all.assign(Campionato=camp, Periodo='Tutto lo storico OOS'))
                        # Stagione corrente OOS
                        _, d_cur = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, True)
                        if d_cur is not None and not d_cur.empty:
                            all_current.append(d_cur.assign(Campionato=camp, Periodo='Stagione corrente OOS'))
                    except Exception as e:
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            frames = all_hist + all_current
            if frames:
                all_df = pd.concat(frames, ignore_index=True)

                st.markdown('### 1. Confronto aggregato')
                rows=[]
                for periodo, g in all_df.groupby('Periodo'):
                    s = _v14_7_stability_summary(g, 0.35)
                    rows.append({'Periodo':periodo, **s})
                agg=pd.DataFrame(rows)
                st.dataframe(agg.round(4), use_container_width=True, hide_index=True)
                st.caption('IC95% = intervallo di confidenza binomiale di Wilson per il Top 1 hit rate. Non è una garanzia predittiva futura.')

                st.markdown('### 2. Per campionato e periodo')
                rows=[]
                for (camp, periodo), g in all_df.groupby(['Campionato','Periodo']):
                    s=_v14_7_stability_summary(g,0.35)
                    rows.append({'Campionato':camp, 'Periodo':periodo, **s})
                per=pd.DataFrame(rows)
                st.dataframe(per.round(4), use_container_width=True, hide_index=True)

                st.markdown('### 3. Stabilità del campione')
                hist_n = int((all_df[all_df['Periodo']=='Tutto lo storico OOS']['top1_prob'] >= 0.35).sum()) if 'Tutto lo storico OOS' in set(all_df['Periodo']) else 0
                cur_n = int((all_df[all_df['Periodo']=='Stagione corrente OOS']['top1_prob'] >= 0.35).sum()) if 'Stagione corrente OOS' in set(all_df['Periodo']) else 0
                st.write(f'**Selezioni ≥35%:** storico {hist_n} · stagione corrente {cur_n}.')
                if hist_n > 0 and cur_n > 0:
                    h = _v14_7_stability_summary(all_df[all_df['Periodo']=='Tutto lo storico OOS'],0.35)
                    c = _v14_7_stability_summary(all_df[all_df['Periodo']=='Stagione corrente OOS'],0.35)
                    delta = c['Top1 hit %'] - h['Top1 hit %']
                    st.info(f"Differenza hit rate stagione corrente − storico: **{delta:+.2f} pp**. Il confronto serve a verificare stabilità, non a scegliere automaticamente una nuova soglia.")

                # Normalizza le colonne richieste dall'export. Alcune versioni V14
                # producono top1_hit/top4_hit, mentre l'audit usa hit_at_1/hit_at_4.
                export_df = all_df.copy()
                if 'hit_at_1' not in export_df.columns and 'top1_hit' in export_df.columns:
                    export_df['hit_at_1'] = export_df['top1_hit']
                if 'hit_at_4' not in export_df.columns and 'top4_hit' in export_df.columns:
                    export_df['hit_at_4'] = export_df['top4_hit']
                if 'margin_top1_top2' not in export_df.columns and {'top1_prob','top2_prob'}.issubset(export_df.columns):
                    export_df['margin_top1_top2'] = (
                        pd.to_numeric(export_df['top1_prob'], errors='coerce')
                        - pd.to_numeric(export_df['top2_prob'], errors='coerce')
                    )
                export_cols = ['Campionato','Periodo','data','casa','trasferta','true_combo',
                               'top1_combo','top1_prob','top2_prob','hit_at_1','hit_at_4',
                               'margin_top1_top2']
                missing_export = [c for c in export_cols if c not in export_df.columns]
                if missing_export:
                    st.warning(f'Export V14.7 non disponibile: colonne mancanti {missing_export}')
                else:
                    out = export_df[export_cols].copy()
                    st.download_button(
                        '⬇️ Scarica V14.7 threshold stability CSV',
                        data=out.to_csv(index=False).encode('utf-8'),
                        file_name='V14_7_threshold_stability_35pct.csv',
                        mime='text/csv', key='v14_7_dl'
                    )

            if errors:
                st.warning('Campionati non completati:')
                for nome,msg in errors:
                    st.write(f'- **{nome}**: {msg}')

mostra_v14_7_threshold_stability(rho_val, ewma_span_val, emivita_val)


# =====================================================================
# 🧪 V14.8 — SENSIBILITÀ DELLA SOGLIA TOP 1 (SOLO DIAGNOSTICO)
# Confronta più soglie intorno al 35% su:
#   1) tutto lo storico OOS;
#   2) stagione corrente OOS.
# Nessuna soglia viene trasferita all'operativa.
# =====================================================================

def _v14_8_threshold_sensitivity_summary(df, prob_threshold):
    if df is None or df.empty:
        return {
            'Soglia %': 100.0 * prob_threshold, 'N': 0,
            'Top1 hit %': np.nan, 'IC95% Top1 low %': np.nan,
            'IC95% Top1 high %': np.nan, 'Top4 hit %': np.nan,
            'Prob. media Top1 %': np.nan, 'Margine medio pp': np.nan,
            'Copertura/Prob. %': np.nan,
        }
    x = df.copy()
    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    if 'hit_at_1' not in x.columns and 'top1_hit' in x.columns:
        x['hit_at_1'] = x['top1_hit']
    if 'hit_at_4' not in x.columns and 'top4_hit' in x.columns:
        x['hit_at_4'] = x['top4_hit']
    if 'margin_top1_top2' not in x.columns and {'top1_prob','top2_prob'}.issubset(x.columns):
        x['margin_top1_top2'] = (
            pd.to_numeric(x['top1_prob'], errors='coerce')
            - pd.to_numeric(x['top2_prob'], errors='coerce')
        )
    x['hit_at_1'] = pd.to_numeric(x.get('hit_at_1'), errors='coerce')
    x['hit_at_4'] = pd.to_numeric(x.get('hit_at_4'), errors='coerce')
    x['margin_top1_top2'] = pd.to_numeric(x.get('margin_top1_top2'), errors='coerce')
    x = x[x['top1_prob'] >= prob_threshold].dropna(subset=['hit_at_1','hit_at_4','top1_prob'])
    if x.empty:
        return {
            'Soglia %': 100.0 * prob_threshold, 'N': 0,
            'Top1 hit %': np.nan, 'IC95% Top1 low %': np.nan,
            'IC95% Top1 high %': np.nan, 'Top4 hit %': np.nan,
            'Prob. media Top1 %': np.nan, 'Margine medio pp': np.nan,
            'Copertura/Prob. %': np.nan,
        }
    n = len(x)
    wins = int(x['hit_at_1'].sum())
    lo, hi = _v14_7_wilson_interval(wins, n)
    mean_p = float(x['top1_prob'].mean())
    return {
        'Soglia %': 100.0 * prob_threshold,
        'N': int(n),
        'Top1 hit %': 100.0 * wins / n,
        'IC95% Top1 low %': 100.0 * lo,
        'IC95% Top1 high %': 100.0 * hi,
        'Top4 hit %': 100.0 * float(x['hit_at_4'].mean()),
        'Prob. media Top1 %': 100.0 * mean_p,
        'Margine medio pp': 100.0 * float(x['margin_top1_top2'].mean()),
        'Copertura/Prob. %': 100.0 * float(x['hit_at_1'].mean()) / max(1e-9, mean_p),
    }


def mostra_v14_8_threshold_sensitivity(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.8 — SENSIBILITÀ DELLA SOGLIA TOP 1')
    st.caption('Test diagnostico ex post su soglie 30%, 32.5%, 35%, 37.5% e 40%. Confronta storico OOS e stagione corrente OOS senza modificare l’operativa.')
    with st.expander('Apri V14.8 — Threshold Sensitivity', expanded=False):
        if st.button('🔬 ESEGUI V14.8 — SENSIBILITÀ SOGLIA', key='v14_8_run'):
            all_hist=[]; all_current=[]; errors=[]
            with st.spinner('Calcolo sensibilità soglia sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        _, d_all = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                        if d_all is not None and not d_all.empty:
                            all_hist.append(d_all.assign(Campionato=camp, Periodo='Tutto lo storico OOS'))
                        _, d_cur = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, True)
                        if d_cur is not None and not d_cur.empty:
                            all_current.append(d_cur.assign(Campionato=camp, Periodo='Stagione corrente OOS'))
                    except Exception as e:
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            frames = all_hist + all_current
            if frames:
                all_df = pd.concat(frames, ignore_index=True)
                thresholds = (0.30, 0.325, 0.35, 0.375, 0.40)

                st.markdown('### 1. Sensibilità aggregata')
                rows=[]
                for periodo, g in all_df.groupby('Periodo'):
                    for thr in thresholds:
                        rows.append({'Periodo': periodo, **_v14_8_threshold_sensitivity_summary(g, thr)})
                sens=pd.DataFrame(rows)
                st.dataframe(sens.round(4), use_container_width=True, hide_index=True)

                st.markdown('### 2. Differenza stagione corrente − storico')
                hist = {float(r['Soglia %']): r for _, r in sens[sens['Periodo']=='Tutto lo storico OOS'].iterrows()}
                cur = {float(r['Soglia %']): r for _, r in sens[sens['Periodo']=='Stagione corrente OOS'].iterrows()}
                delta_rows=[]
                for thr in sorted(set(hist) & set(cur)):
                    delta_rows.append({
                        'Soglia %': thr,
                        'N storico': hist[thr]['N'],
                        'N corrente': cur[thr]['N'],
                        'Δ Top1 hit pp': cur[thr]['Top1 hit %'] - hist[thr]['Top1 hit %'],
                        'Δ Top4 pp': cur[thr]['Top4 hit %'] - hist[thr]['Top4 hit %'],
                        'Δ Prob. media Top1 pp': cur[thr]['Prob. media Top1 %'] - hist[thr]['Prob. media Top1 %'],
                        'Δ Margine medio pp': cur[thr]['Margine medio pp'] - hist[thr]['Margine medio pp'],
                        'Δ Copertura/Prob. pp': cur[thr]['Copertura/Prob. %'] - hist[thr]['Copertura/Prob. %'],
                    })
                delta_df=pd.DataFrame(delta_rows)
                st.dataframe(delta_df.round(4), use_container_width=True, hide_index=True)
                st.caption('Le differenze descrivono la stabilità della metrica al variare della soglia; non costituiscono una selezione automatica della soglia.')

                st.markdown('### 3. Focus intorno al 35%')
                focus = sens[sens['Soglia %'].isin([32.5,35.0,37.5])].copy()
                st.dataframe(focus.round(4), use_container_width=True, hide_index=True)

                export_cols=['Periodo','Soglia %','N','Top1 hit %','IC95% Top1 low %','IC95% Top1 high %','Top4 hit %','Prob. media Top1 %','Margine medio pp','Copertura/Prob. %']
                st.download_button(
                    '⬇️ Scarica V14.8 sensitivity CSV',
                    data=sens[export_cols].to_csv(index=False).encode('utf-8'),
                    file_name='V14_8_threshold_sensitivity.csv',
                    mime='text/csv', key='v14_8_dl'
                )

            if errors:
                st.warning('Campionati non completati:')
                for nome,msg in errors:
                    st.write(f'- **{nome}**: {msg}')

mostra_v14_8_threshold_sensitivity(rho_val, ewma_span_val, emivita_val)

# =====================================================================
# 🧪 V14.9 — STABILITÀ TEMPORALE DELLA SOGLIA 35% (SOLO DIAGNOSTICO)
# Divide lo storico OOS in 4 finestre cronologiche di numerosità simile e
# verifica se la selezione Top1 >=35% mantiene comportamento coerente nel tempo.
# Nessuna modifica all'operativa e nessuna ottimizzazione della soglia.
# =====================================================================

def _v14_9_window_summary(df, prob_threshold=0.35):
    if df is None or df.empty:
        return {
            'N OOS': 0, 'N selezioni >= soglia': 0,
            'Quota selezionata %': np.nan, 'Top1 hit %': np.nan,
            'IC95% Top1 low %': np.nan, 'IC95% Top1 high %': np.nan,
            'Top4 hit %': np.nan, 'Prob. media Top1 %': np.nan,
            'Margine medio pp': np.nan, 'Copertura/Prob. %': np.nan,
        }
    x = df.copy()
    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    if 'hit_at_1' not in x.columns and 'top1_hit' in x.columns:
        x['hit_at_1'] = x['top1_hit']
    if 'hit_at_4' not in x.columns and 'top4_hit' in x.columns:
        x['hit_at_4'] = x['top4_hit']
    if 'margin_top1_top2' not in x.columns and {'top1_prob','top2_prob'}.issubset(x.columns):
        x['margin_top1_top2'] = (
            pd.to_numeric(x['top1_prob'], errors='coerce')
            - pd.to_numeric(x['top2_prob'], errors='coerce')
        )
    x['hit_at_1'] = pd.to_numeric(x.get('hit_at_1'), errors='coerce')
    x['hit_at_4'] = pd.to_numeric(x.get('hit_at_4'), errors='coerce')
    x['margin_top1_top2'] = pd.to_numeric(x.get('margin_top1_top2'), errors='coerce')
    x = x.dropna(subset=['top1_prob','hit_at_1','hit_at_4'])
    n_oos = len(x)
    s = x[x['top1_prob'] >= prob_threshold].copy()
    n = len(s)
    if n == 0:
        return {
            'N OOS': int(n_oos), 'N selezioni >= soglia': 0,
            'Quota selezionata %': 0.0, 'Top1 hit %': np.nan,
            'IC95% Top1 low %': np.nan, 'IC95% Top1 high %': np.nan,
            'Top4 hit %': np.nan, 'Prob. media Top1 %': np.nan,
            'Margine medio pp': np.nan, 'Copertura/Prob. %': np.nan,
        }
    wins = int(s['hit_at_1'].sum())
    lo, hi = _v14_7_wilson_interval(wins, n)
    mean_p = float(s['top1_prob'].mean())
    return {
        'N OOS': int(n_oos),
        'N selezioni >= soglia': int(n),
        'Quota selezionata %': 100.0 * n / max(1, n_oos),
        'Top1 hit %': 100.0 * wins / n,
        'IC95% Top1 low %': 100.0 * lo,
        'IC95% Top1 high %': 100.0 * hi,
        'Top4 hit %': 100.0 * float(s['hit_at_4'].mean()),
        'Prob. media Top1 %': 100.0 * mean_p,
        'Margine medio pp': 100.0 * float(s['margin_top1_top2'].mean()) if 'margin_top1_top2' in s.columns else np.nan,
        'Copertura/Prob. %': 100.0 * float(s['hit_at_1'].mean()) / max(1e-9, mean_p),
    }


def mostra_v14_9_temporal_stability(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.9 — STABILITÀ TEMPORALE SOGLIA 35%')
    st.caption('Test diagnostico: tutto lo storico OOS viene ordinato per data e diviso in 4 finestre cronologiche. La soglia 35% resta fissa ex ante e non viene ottimizzata.')
    with st.expander('Apri V14.9 — Temporal Stability', expanded=False):
        if st.button('🔬 ESEGUI V14.9 — STABILITÀ TEMPORALE 35%', key='v14_9_run'):
            frames=[]; errors=[]
            with st.spinner('Calcolo stabilità temporale sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        _, d = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                        if d is not None and not d.empty:
                            frames.append(d.assign(Campionato=camp))
                    except Exception as e:
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            if frames:
                all_df = pd.concat(frames, ignore_index=True)
                all_df['data'] = pd.to_datetime(all_df['data'], errors='coerce')
                all_df = all_df.dropna(subset=['data']).sort_values('data').reset_index(drop=True)

                st.markdown('### 1. Stabilità aggregata per finestra cronologica')
                n = len(all_df)
                if n >= 20:
                    labels = ['Finestra 1 (più vecchia)', 'Finestra 2', 'Finestra 3', 'Finestra 4 (più recente)']
                    all_df['_window'] = pd.qcut(all_df.index, q=4, labels=labels, duplicates='drop')
                    rows=[]
                    for window, g in all_df.groupby('_window', observed=True):
                        s = _v14_9_window_summary(g, 0.35)
                        rows.append({'Finestra':str(window), **s})
                    win_df = pd.DataFrame(rows)
                    st.dataframe(win_df.round(4), use_container_width=True, hide_index=True)
                    st.caption('Le finestre sono cronologiche e hanno numerosità OOS simile. Le metriche Top1/Top4 sono calcolate solo sulle selezioni con probabilità Top1 >=35%.')

                    st.markdown('### 2. Dettaglio per campionato e finestra')
                    rows=[]
                    for camp, gc in all_df.groupby('Campionato'):
                        if len(gc) < 20:
                            continue
                        gc = gc.sort_values('data').reset_index(drop=True)
                        gc['_window'] = pd.qcut(gc.index, q=4, labels=labels, duplicates='drop')
                        for window, g in gc.groupby('_window', observed=True):
                            s = _v14_9_window_summary(g, 0.35)
                            rows.append({'Campionato':camp, 'Finestra':str(window), **s})
                    by_league = pd.DataFrame(rows)
                    if not by_league.empty:
                        st.dataframe(by_league.round(4), use_container_width=True, hide_index=True)
                    else:
                        st.info('Campioni per campionato insufficienti per creare 4 finestre affidabili.')

                    st.markdown('### 3. Confronto prima vs ultima finestra')
                    first = _v14_9_window_summary(all_df[all_df['_window'] == labels[0]], 0.35)
                    last = _v14_9_window_summary(all_df[all_df['_window'] == labels[-1]], 0.35)
                    cmp = pd.DataFrame([{
                        'Metrica':'Top1 hit %', 'Prima finestra':first['Top1 hit %'], 'Ultima finestra':last['Top1 hit %'],
                        'Delta ultima-prima pp':last['Top1 hit %'] - first['Top1 hit %']
                    },{
                        'Metrica':'Top4 hit %', 'Prima finestra':first['Top4 hit %'], 'Ultima finestra':last['Top4 hit %'],
                        'Delta ultima-prima pp':last['Top4 hit %'] - first['Top4 hit %']
                    },{
                        'Metrica':'Prob. media Top1 %', 'Prima finestra':first['Prob. media Top1 %'], 'Ultima finestra':last['Prob. media Top1 %'],
                        'Delta ultima-prima pp':last['Prob. media Top1 %'] - first['Prob. media Top1 %']
                    },{
                        'Metrica':'Margine medio pp', 'Prima finestra':first['Margine medio pp'], 'Ultima finestra':last['Margine medio pp'],
                        'Delta ultima-prima pp':last['Margine medio pp'] - first['Margine medio pp']
                    },{
                        'Metrica':'Copertura/Prob. %', 'Prima finestra':first['Copertura/Prob. %'], 'Ultima finestra':last['Copertura/Prob. %'],
                        'Delta ultima-prima pp':last['Copertura/Prob. %'] - first['Copertura/Prob. %']
                    }])
                    st.dataframe(cmp.round(4), use_container_width=True, hide_index=True)
                    st.info('Questo confronto descrive l’eventuale drift temporale. Non trasferisce automaticamente alcuna modifica alla soglia o all’operativa.')

                    st.markdown('### 4. Export')
                    export_cols=['Campionato','data','casa','trasferta','true_combo','top1_combo','top1_prob','top2_prob','top1_hit','top4_hit']
                    export_cols=[c for c in export_cols if c in all_df.columns]
                    st.download_button(
                        '⬇️ Scarica V14.9 temporal stability CSV',
                        data=all_df[export_cols].to_csv(index=False).encode('utf-8'),
                        file_name='V14_9_temporal_stability_35pct.csv',
                        mime='text/csv', key='v14_9_dl'
                    )
                else:
                    st.warning('Numero di osservazioni OOS insufficiente per una divisione in 4 finestre.')

            if errors:
                st.warning('Campionati non completati:')
                for nome,msg in errors:
                    st.write(f'- **{nome}**: {msg}')



# 🧪 V14.10 — DRIFT DI CALIBRAZIONE TEMPORALE DELLE SELEZIONI >=35% (SOLO DIAGNOSTICO)

def _v14_10_calibration_window_summary(g, prob_threshold=0.35):
    x = g.copy()
    if 'hit_at_1' not in x.columns and 'top1_hit' in x.columns:
        x['hit_at_1'] = x['top1_hit']
    if 'top1_prob' not in x.columns:
        return None
    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    x['hit_at_1'] = pd.to_numeric(x.get('hit_at_1'), errors='coerce')
    x = x.dropna(subset=['top1_prob', 'hit_at_1'])
    s = x[x['top1_prob'] >= prob_threshold].copy()
    n = len(s)
    if n == 0:
        return {
            'N selezioni': 0,
            'Top1 hit %': np.nan,
            'Prob. media Top1 %': np.nan,
            'Gap calibrazione pp': np.nan,
            'Brier Top1': np.nan,
            'Log Loss Top1': np.nan,
        }
    y = s['hit_at_1'].astype(float).to_numpy()
    p = np.clip(s['top1_prob'].astype(float).to_numpy(), 1e-6, 1-1e-6)
    hit = float(y.mean())
    mean_p = float(p.mean())
    brier = float(np.mean((p-y)**2))
    logloss = float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))
    return {
        'N selezioni': int(n),
        'Top1 hit %': 100.0 * hit,
        'Prob. media Top1 %': 100.0 * mean_p,
        'Gap calibrazione pp': 100.0 * (hit - mean_p),
        'Brier Top1': brier,
        'Log Loss Top1': logloss,
    }


def mostra_v14_10_temporal_calibration_drift(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.10 — DRIFT DI CALIBRAZIONE TEMPORALE SOGLIA 35%')
    st.caption('Test diagnostico: misura se il rapporto tra probabilità Top1 prevista e hit Top1 realizzata cambia nel tempo sulle sole selezioni >=35%. Nessuna modifica all’operativa.')
    with st.expander('Apri V14.10 — Temporal Calibration Drift', expanded=False):
        if st.button('🔬 ESEGUI V14.10 — DRIFT CALIBRAZIONE 35%', key='v14_10_run'):
            frames=[]; errors=[]
            with st.spinner('Calcolo drift di calibrazione sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        _, d = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                        if d is not None and not d.empty:
                            frames.append(d.assign(Campionato=camp))
                    except Exception as e:
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            if frames:
                all_df = pd.concat(frames, ignore_index=True)
                all_df['data'] = pd.to_datetime(all_df['data'], errors='coerce')
                all_df = all_df.dropna(subset=['data']).sort_values('data').reset_index(drop=True)

                if len(all_df) >= 20:
                    labels = ['Finestra 1 (più vecchia)', 'Finestra 2', 'Finestra 3', 'Finestra 4 (più recente)']
                    all_df['_window'] = pd.qcut(all_df.index, q=4, labels=labels, duplicates='drop')

                    st.markdown('### 1. Calibrazione Top1 nelle selezioni >=35%')
                    rows=[]
                    for window, g in all_df.groupby('_window', observed=True):
                        s = _v14_10_calibration_window_summary(g, 0.35)
                        if s is not None:
                            s['Finestra'] = str(window)
                            rows.append(s)
                    cal_df = pd.DataFrame(rows)
                    if not cal_df.empty:
                        cols=['Finestra','N selezioni','Top1 hit %','Prob. media Top1 %','Gap calibrazione pp','Brier Top1','Log Loss Top1']
                        st.dataframe(cal_df[cols].round(4), use_container_width=True, hide_index=True)
                        st.caption('Gap calibrazione = Top1 hit % − probabilità media Top1 %. Un valore negativo indica probabilità media superiore agli hit osservati.')

                    st.markdown('### 2. Prima vs ultima finestra')
                    first = _v14_10_calibration_window_summary(all_df[all_df['_window'] == labels[0]], 0.35)
                    last = _v14_10_calibration_window_summary(all_df[all_df['_window'] == labels[-1]], 0.35)
                    if first and last and first['N selezioni'] > 0 and last['N selezioni'] > 0:
                        cmp = pd.DataFrame([
                            {'Metrica':'Top1 hit %','Prima finestra':first['Top1 hit %'],'Ultima finestra':last['Top1 hit %'],'Delta ultima-prima pp':last['Top1 hit %']-first['Top1 hit %']},
                            {'Metrica':'Prob. media Top1 %','Prima finestra':first['Prob. media Top1 %'],'Ultima finestra':last['Prob. media Top1 %'],'Delta ultima-prima pp':last['Prob. media Top1 %']-first['Prob. media Top1 %']},
                            {'Metrica':'Gap calibrazione pp','Prima finestra':first['Gap calibrazione pp'],'Ultima finestra':last['Gap calibrazione pp'],'Delta ultima-prima pp':last['Gap calibrazione pp']-first['Gap calibrazione pp']},
                            {'Metrica':'Brier Top1','Prima finestra':first['Brier Top1'],'Ultima finestra':last['Brier Top1'],'Delta ultima-prima pp':last['Brier Top1']-first['Brier Top1']},
                            {'Metrica':'Log Loss Top1','Prima finestra':first['Log Loss Top1'],'Ultima finestra':last['Log Loss Top1'],'Delta ultima-prima pp':last['Log Loss Top1']-first['Log Loss Top1']},
                        ])
                        st.dataframe(cmp.round(4), use_container_width=True, hide_index=True)

                    st.markdown('### 3. Dettaglio per campionato')
                    rows=[]
                    for camp, gc in all_df.groupby('Campionato'):
                        if len(gc) < 20:
                            continue
                        gc=gc.sort_values('data').reset_index(drop=True)
                        gc['_window']=pd.qcut(gc.index, q=4, labels=labels, duplicates='drop')
                        for window, g in gc.groupby('_window', observed=True):
                            s=_v14_10_calibration_window_summary(g,0.35)
                            if s is not None:
                                rows.append({'Campionato':camp,'Finestra':str(window),**s})
                    by_league=pd.DataFrame(rows)
                    if not by_league.empty:
                        st.dataframe(by_league.round(4),use_container_width=True,hide_index=True)

                    st.markdown('### 4. Export')
                    export_cols=['Campionato','data','casa','trasferta','true_combo','top1_combo','top1_prob','top2_prob','top1_hit','top4_hit','margin_top1_top2']
                    export_cols=[c for c in export_cols if c in all_df.columns]
                    st.download_button(
                        '⬇️ Scarica V14.10 temporal calibration CSV',
                        data=all_df[export_cols].to_csv(index=False).encode('utf-8'),
                        file_name='V14_10_temporal_calibration_35pct.csv',
                        mime='text/csv', key='v14_10_dl'
                    )
                else:
                    st.warning('Numero di osservazioni OOS insufficiente per una divisione in 4 finestre.')

            if errors:
                st.warning('Campionati non completati:')
                for nome,msg in errors:
                    st.write(f'- **{nome}**: {msg}')


# 🧪 V14.11 — CALIBRAZIONE PER FASCE DI CONFIDENZA ≥35% (SOLO DIAGNOSTICO)

def _v14_11_band_summary(g, prob_threshold=0.35):
    x = g.copy()
    if 'hit_at_1' not in x.columns and 'top1_hit' in x.columns:
        x['hit_at_1'] = x['top1_hit']
    if 'top1_prob' not in x.columns:
        return pd.DataFrame()
    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    x['hit_at_1'] = pd.to_numeric(x.get('hit_at_1'), errors='coerce')
    x = x.dropna(subset=['top1_prob', 'hit_at_1'])
    x = x[x['top1_prob'] >= prob_threshold].copy()
    if x.empty:
        return pd.DataFrame()

    bands = [0.35, 0.40, 0.45, 0.50, 0.60, 1.0000001]
    labels = ['35–40%', '40–45%', '45–50%', '50–60%', '≥60%']
    x['_band'] = pd.cut(x['top1_prob'], bins=bands, labels=labels, right=False, include_lowest=True)

    rows = []
    for band, gband in x.groupby('_band', observed=True):
        n = len(gband)
        y = gband['hit_at_1'].astype(float).to_numpy()
        p = np.clip(gband['top1_prob'].astype(float).to_numpy(), 1e-6, 1-1e-6)
        hit = float(y.mean())
        mean_p = float(p.mean())
        rows.append({
            'Fascia probabilità': str(band),
            'N': int(n),
            'Top1 hit %': 100.0 * hit,
            'Prob. media %': 100.0 * mean_p,
            'Gap calibrazione pp': 100.0 * (hit - mean_p),
            'Brier': float(np.mean((p-y)**2)),
            'Log Loss': float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p))),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out['_abs_gap_weight'] = out['N'] * out['Gap calibrazione pp'].abs()
        total_n = out['N'].sum()
        ece = float(out['_abs_gap_weight'].sum() / total_n) if total_n else np.nan
        out.attrs['ECE_pp'] = ece
    return out


def mostra_v14_11_confidence_bands(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.11 — CALIBRAZIONE PER FASCE DI CONFIDENZA ≥35%')
    st.caption('Test diagnostico: verifica dove si concentra l’errore di calibrazione tra le selezioni Top1 con probabilità ≥35%. Nessuna modifica all’operativa e nessuna nuova soglia proposta automaticamente.')
    with st.expander('Apri V14.11 — Confidence Bands', expanded=False):
        if st.button('🔬 ESEGUI V14.11 — FASCE CONFIDENZA 35%+', key='v14_11_run'):
            frames=[]; errors=[]
            with st.spinner('Calcolo calibrazione per fasce sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        _, d = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                        if d is not None and not d.empty:
                            frames.append(d.assign(Campionato=camp))
                    except Exception as e:
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            if frames:
                all_df = pd.concat(frames, ignore_index=True)
                all_df['data'] = pd.to_datetime(all_df['data'], errors='coerce')
                all_df = all_df.dropna(subset=['data']).sort_values('data').reset_index(drop=True)

                st.markdown('### 1. Storico OOS — fasce di probabilità')
                hist = _v14_11_band_summary(all_df, 0.35)
                if not hist.empty:
                    st.dataframe(hist.drop(columns=['_abs_gap_weight']).round(4), use_container_width=True, hide_index=True)
                    st.caption(f"ECE pesato sulle sole selezioni ≥35%: {hist.attrs.get('ECE_pp', np.nan):.2f} pp")
                else:
                    st.info('Nessuna selezione ≥35% disponibile.')

                st.markdown('### 2. Stagione corrente OOS — fasce di probabilità')
                if 'data' in all_df.columns:
                    latest_year = int(all_df['data'].dt.year.max())
                    current_cut = pd.Timestamp(year=latest_year, month=8, day=1)
                    curr = all_df[all_df['data'] >= current_cut].copy()
                else:
                    curr = pd.DataFrame()
                curr = _v14_11_band_summary(curr, 0.35)
                if not curr.empty:
                    st.dataframe(curr.drop(columns=['_abs_gap_weight']).round(4), use_container_width=True, hide_index=True)
                    st.caption(f"ECE pesato stagione corrente OOS (dal 1° agosto dell'anno più recente presente nei dati): {curr.attrs.get('ECE_pp', np.nan):.2f} pp")
                else:
                    st.info('Nessuna selezione ≥35% nella finestra corrente disponibile.')

                st.markdown('### 3. Confronto aggregato storico vs stagione corrente')
                def _overall_band(g, period):
                    h = _v14_11_band_summary(g, 0.35)
                    if h.empty:
                        return pd.DataFrame()
                    h = h.drop(columns=['_abs_gap_weight']).copy()
                    h.insert(0, 'Periodo', period)
                    return h
                cmp_frames=[]
                h1=_overall_band(all_df,'Tutto lo storico OOS')
                h2=_overall_band(all_df[all_df['data'] >= current_cut],'Stagione corrente OOS')
                if not h1.empty: cmp_frames.append(h1)
                if not h2.empty: cmp_frames.append(h2)
                if cmp_frames:
                    st.dataframe(pd.concat(cmp_frames, ignore_index=True).round(4), use_container_width=True, hide_index=True)

                st.markdown('### 4. Dettaglio per campionato — storico OOS')
                rows=[]
                for camp, gc in all_df.groupby('Campionato'):
                    h=_v14_11_band_summary(gc,0.35)
                    if not h.empty:
                        for _,r in h.iterrows():
                            rows.append({'Campionato':camp,**r.to_dict()})
                by_league=pd.DataFrame(rows)
                if not by_league.empty:
                    st.dataframe(by_league.drop(columns=['_abs_gap_weight'], errors='ignore').round(4),use_container_width=True,hide_index=True)

                st.markdown('### 5. Export')
                export_cols=['Campionato','data','casa','trasferta','true_combo','top1_combo','top1_prob','top2_prob','top1_hit','top4_hit','margin_top1_top2']
                export_cols=[c for c in export_cols if c in all_df.columns]
                st.download_button(
                    '⬇️ Scarica V14.11 confidence bands CSV',
                    data=all_df[export_cols].to_csv(index=False).encode('utf-8'),
                    file_name='V14_11_confidence_bands_35pct.csv',
                    mime='text/csv', key='v14_11_dl'
                )
            if errors:
                st.warning('Campionati non completati:')
                for nome,msg in errors:
                    st.write(f'- **{nome}**: {msg}')


mostra_v14_10_temporal_calibration_drift(rho_val, ewma_span_val, emivita_val)
mostra_v14_11_confidence_bands(rho_val, ewma_span_val, emivita_val)

# =====================================================================
# 🧪 V14.12-FIX — PLATT TOP1 WALK-FORWARD (SOLO DIAGNOSTICO)
# Fix robusto del blocco V14.12.
# Principi:
#   1) valutazione SOLO sulle selezioni Top1 >= 35%;
#   2) fit per singolo campionato usando esclusivamente osservazioni precedenti;
#   3) training massimo 100 osservazioni precedenti; minimo 50 per applicare PLATT;
#   4) training su tutte le Top1 storiche precedenti del campionato, mentre
#      l'universo di valutazione resta limitato alle selezioni >=35%, così
#      ogni campionato dispone di un campione di training sufficiente;
#   5) nessuna informazione della partita da predire entra nel fit;
#   6) niente np.array_split su DataFrame e indexing NumPy fragile nel ramo
#      diagnostico, per evitare il KeyError/IndexError dell'ambiente Streamlit.
# =====================================================================

V14_12_PROB_THRESHOLD = 0.35
V14_12_MIN_TRAIN = 50
V14_12_MAX_TRAIN = 100


def _v14_12_fit_platt_top1(hist_df, n_train=100):
    """Fit Platt sulla logit della Top1 usando solo osservazioni precedenti."""
    if hist_df is None or hist_df.empty:
        return None, np.nan, 'nessun training', 0

    x = hist_df.copy()
    required = ['top1_prob', 'top1_hit']
    if any(c not in x.columns for c in required):
        return None, np.nan, 'colonne mancanti', 0

    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    x['top1_hit'] = pd.to_numeric(x['top1_hit'], errors='coerce')
    x = x.dropna(subset=required).copy()
    x = x[x['top1_prob'].between(1e-6, 1.0 - 1e-6)]
    x['top1_hit'] = x['top1_hit'].astype(int)
    x = x.tail(int(min(max(n_train, V14_12_MIN_TRAIN), V14_12_MAX_TRAIN)))

    n = len(x)
    if n < V14_12_MIN_TRAIN:
        return None, np.nan, 'campione insufficiente', n

    y = np.asarray(x['top1_hit'].to_numpy(dtype=int)).reshape(-1)
    if np.unique(y).size < 2:
        return None, np.nan, 'un solo esito presente nel training', n

    p = np.asarray(x['top1_prob'].to_numpy(dtype=float)).reshape(-1)
    z = np.log(p / (1.0 - p)).reshape(-1, 1)

    model = LogisticRegression(solver='lbfgs', C=1e6, max_iter=1000)
    model.fit(z, y)

    coef_arr = np.asarray(model.coef_, dtype=float).reshape(-1)
    coef = float(coef_arr[0]) if coef_arr.size else np.nan
    if not np.isfinite(coef) or coef <= 0.0:
        return None, coef, 'fit non monotono: pendenza non positiva', n

    return model, coef, 'applicabile', n


def _v14_12_predict_platt_top1(p_raw, model):
    p_raw = float(np.clip(float(p_raw), 1e-6, 1.0 - 1e-6))
    if model is None:
        return p_raw

    z = np.asarray([[np.log(p_raw / (1.0 - p_raw))]], dtype=float)
    proba = np.asarray(model.predict_proba(z), dtype=float).reshape(-1)
    if proba.size < 2:
        return p_raw
    return float(np.clip(proba[1], 1e-6, 1.0 - 1e-6))


def _v14_12_metrics(g, period_label):
    """Metriche su un gruppo di selezioni >=35%."""
    if g is None or g.empty:
        return None

    y = np.asarray(g['top1_hit'].to_numpy(dtype=float)).reshape(-1)
    p0 = np.clip(np.asarray(g['top1_prob_raw'].to_numpy(dtype=float)).reshape(-1), 1e-6, 1.0 - 1e-6)
    p1 = np.clip(np.asarray(g['top1_prob_platt'].to_numpy(dtype=float)).reshape(-1), 1e-6, 1.0 - 1e-6)

    return {
        'Periodo': period_label,
        'N selezioni >=35%': int(len(g)),
        'N PLATT applicato': int(g['platt_applied'].sum()) if 'platt_applied' in g.columns else int(len(g)),
        'Top1 hit %': 100.0 * float(y.mean()),
        'RAW Prob. media %': 100.0 * float(p0.mean()),
        'PLATT Prob. media %': 100.0 * float(p1.mean()),
        'RAW Gap calibrazione pp': 100.0 * float(y.mean() - p0.mean()),
        'PLATT Gap calibrazione pp': 100.0 * float(y.mean() - p1.mean()),
        'RAW Brier': float(np.mean((p0 - y) ** 2)),
        'PLATT Brier': float(np.mean((p1 - y) ** 2)),
        'RAW Log Loss': float(-np.mean(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0))),
        'PLATT Log Loss': float(-np.mean(y * np.log(p1) + (1.0 - y) * np.log(1.0 - p1))),
        'Delta Brier pp': 100.0 * float(np.mean((p1 - y) ** 2) - np.mean((p0 - y) ** 2)),
        'Delta Log Loss': float(
            -np.mean(y * np.log(p1) + (1.0 - y) * np.log(1.0 - p1))
            + np.mean(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0))
        ),
    }


def _v14_12_walkforward_top1(df, n_train=100, prob_threshold=V14_12_PROB_THRESHOLD):
    """Walk-forward robusto: fit sul passato, valutazione solo Top1 >=35%."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = df.copy()
    if 'data' in x.columns:
        x['data'] = pd.to_datetime(x['data'], errors='coerce')
        x = x.sort_values(['data'], kind='mergesort').reset_index(drop=True)

    required = ['top1_prob', 'top1_hit']
    if any(c not in x.columns for c in required):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    x['top1_hit'] = pd.to_numeric(x['top1_hit'], errors='coerce')
    x = x.dropna(subset=required).copy().reset_index(drop=True)
    x['top1_hit'] = x['top1_hit'].astype(int)

    rows = []
    train_limit = int(min(max(n_train, V14_12_MIN_TRAIN), V14_12_MAX_TRAIN))

    for i in range(len(x)):
        p_raw = float(x.iloc[i]['top1_prob'])
        if not np.isfinite(p_raw):
            continue
        y = int(x.iloc[i]['top1_hit'])

        # La partita entra nella valutazione solo se supera la soglia 35%.
        selected = p_raw >= float(prob_threshold)

        prior = x.iloc[:i][['top1_prob', 'top1_hit']].copy()
        model = None
        coef = np.nan
        reason = 'non selezionata < soglia'
        n_eff = min(len(prior), train_limit)

        if selected:
            model, coef, reason, n_eff = _v14_12_fit_platt_top1(prior, n_train=train_limit)

        p_platt = _v14_12_predict_platt_top1(p_raw, model) if selected else p_raw

        # Dettaglio solo per l'universo selezionato >=35%, così il CSV è coerente
        # con V14.11 e con il test di calibrazione che stiamo approfondendo.
        if selected:
            row = {
                **{k: x.iloc[i][k] for k in ['data', 'casa', 'trasferta'] if k in x.columns},
                'top1_hit': y,
                'top1_prob_raw': p_raw,
                'top1_prob_platt': p_platt,
                'platt_applied': bool(model is not None),
                'platt_coef': coef,
                'platt_train_n': int(n_eff),
                'platt_reason': reason,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame(), pd.DataFrame()

    # Le righe precedenti alla disponibilità di almeno 50 osservazioni restano
    # nell'universo ma RAW=PLATT e sono marcate platt_applied=False.
    summary_rows = []
    all_metrics = _v14_12_metrics(out, 'Tutto lo storico OOS')
    if all_metrics is not None:
        summary_rows.append(all_metrics)

    if 'data' in out.columns and out['data'].notna().any():
        latest_year = int(pd.to_datetime(out['data'], errors='coerce').dt.year.max())
        cut = pd.Timestamp(year=latest_year, month=8, day=1)
        curr = out[pd.to_datetime(out['data'], errors='coerce') >= cut].copy()
        if not curr.empty:
            current_metrics = _v14_12_metrics(curr, 'Stagione corrente OOS')
            if current_metrics is not None:
                summary_rows.append(current_metrics)

    summary = pd.DataFrame(summary_rows)

    # Stabilità temporale: quattro finestre con split sugli INDICI, non np.array_split.
    temporal_rows = []
    n = len(out)
    if n >= 20:
        edges = np.linspace(0, n, 5, dtype=int)
        labels = ['Finestra 1 (più vecchia)', 'Finestra 2', 'Finestra 3', 'Finestra 4 (più recente)']
        for j, lab in enumerate(labels):
            a, b = int(edges[j]), int(edges[j + 1])
            ch = out.iloc[a:b].copy()
            if not ch.empty:
                met = _v14_12_metrics(ch, lab)
                if met is not None:
                    temporal_rows.append(met)

    temporal_df = pd.DataFrame(temporal_rows)
    return out, summary, temporal_df


def mostra_v14_12_top1_platt_walkforward(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.12-FIX — PLATT TOP1 WALK-FORWARD ≥35%')
    st.caption(
        'Solo diagnostico. La valutazione usa esclusivamente selezioni Top1 ≥35%. '
        'Il fit è walk-forward per singolo campionato e usa solo osservazioni precedenti; '
        'training massimo 100, minimo 50. Nessuna modifica alle probabilità operative.'
    )

    with st.expander('Apri V14.12-FIX — PLATT Top1 Walk-Forward ≥35%', expanded=False):
        ntrain = st.number_input(
            'Training PLATT Top1 (massimo)',
            V14_12_MIN_TRAIN,
            V14_12_MAX_TRAIN,
            100,
            10,
            key='v14_12_fix_train'
        )

        if st.button('🔬 ESEGUI V14.12-FIX — PLATT TOP1 ≥35%', key='v14_12_fix_run'):
            summary_frames = []
            temporal_frames = []
            detail_frames = []
            errors = []

            with st.spinner('Calcolo PLATT walk-forward Top1 ≥35% sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        res = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                        if res is None:
                            continue
                        _, det = res
                        if det is None or det.empty:
                            continue

                        wf, sm, tm = _v14_12_walkforward_top1(
                            det,
                            int(ntrain),
                            V14_12_PROB_THRESHOLD
                        )

                        if not sm.empty:
                            summary_frames.append(sm.assign(Campionato=camp))
                        if not tm.empty:
                            temporal_frames.append(tm.assign(Campionato=camp))
                        if not wf.empty:
                            detail_frames.append(wf.assign(Campionato=camp))

                    except Exception as e:
                        # Mostra il tipo e il messaggio; non nasconde il punto
                        # del nuovo blocco diagnostico dietro un errore generico.
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            if summary_frames:
                sdf = pd.concat(summary_frames, ignore_index=True)
                sdf = sdf[
                    [
                        'Campionato', 'Periodo', 'N selezioni >=35%', 'N PLATT applicato',
                        'Top1 hit %', 'RAW Prob. media %', 'PLATT Prob. media %',
                        'RAW Gap calibrazione pp', 'PLATT Gap calibrazione pp',
                        'RAW Brier', 'PLATT Brier', 'RAW Log Loss', 'PLATT Log Loss',
                        'Delta Brier pp', 'Delta Log Loss'
                    ]
                ]

                st.markdown('### 1. RAW vs PLATT — per campionato')
                st.dataframe(sdf.round(4), use_container_width=True, hide_index=True)
                st.caption('Delta Brier pp e Delta Log Loss = PLATT − RAW. Valori negativi indicano miglioramento di PLATT.')

                agg_rows = []
                for periodo, g in sdf.groupby('Periodo', sort=False):
                    weights = g['N selezioni >=35%'].astype(float)
                    total_w = float(weights.sum())
                    if total_w <= 0:
                        continue

                    def wavg(col):
                        return float(np.average(g[col].astype(float), weights=weights))

                    agg_rows.append({
                        'Periodo': periodo,
                        'N selezioni >=35%': int(weights.sum()),
                        'N PLATT applicato': int(g['N PLATT applicato'].sum()),
                        'Top1 hit %': wavg('Top1 hit %'),
                        'RAW Prob. media %': wavg('RAW Prob. media %'),
                        'PLATT Prob. media %': wavg('PLATT Prob. media %'),
                        'RAW Gap calibrazione pp': wavg('RAW Gap calibrazione pp'),
                        'PLATT Gap calibrazione pp': wavg('PLATT Gap calibrazione pp'),
                        'RAW Brier': wavg('RAW Brier'),
                        'PLATT Brier': wavg('PLATT Brier'),
                        'RAW Log Loss': wavg('RAW Log Loss'),
                        'PLATT Log Loss': wavg('PLATT Log Loss'),
                        'Delta Brier pp': wavg('Delta Brier pp'),
                        'Delta Log Loss': wavg('Delta Log Loss'),
                    })

                if agg_rows:
                    st.markdown('### 2. RAW vs PLATT — aggregato tutte le leghe')
                    st.dataframe(pd.DataFrame(agg_rows).round(4), use_container_width=True, hide_index=True)

            if temporal_frames:
                st.markdown('### 3. Stabilità temporale del confronto')
                tdf = pd.concat(temporal_frames, ignore_index=True)
                st.dataframe(tdf.round(4), use_container_width=True, hide_index=True)

            if detail_frames:
                dd = pd.concat(detail_frames, ignore_index=True)
                export_cols = [
                    'Campionato', 'data', 'casa', 'trasferta', 'top1_hit',
                    'top1_prob_raw', 'top1_prob_platt', 'platt_applied',
                    'platt_coef', 'platt_train_n', 'platt_reason'
                ]
                export_cols = [c for c in export_cols if c in dd.columns]
                st.markdown('### 4. Export dettaglio walk-forward')
                st.download_button(
                    '⬇️ Scarica V14.12-FIX PLATT Top1 dettaglio CSV',
                    data=dd[export_cols].to_csv(index=False).encode('utf-8'),
                    file_name='V14_12_FIX_top1_platt_walkforward_35pct.csv',
                    mime='text/csv',
                    key='v14_12_fix_dl'
                )

            if errors:
                st.warning('Campionati non completati:')
                for nome, msg in errors:
                    st.write(f'- **{nome}**: {msg}')


mostra_v14_12_top1_platt_walkforward(rho_val, ewma_span_val, emivita_val)

# =====================================================================
# 🧪 V14.13 — PLATT TOP1: BOOTSTRAP PAIRED + TEST DI SEGNO
# Solo diagnostico. Quantifica l'incertezza della differenza RAW vs PLATT
# sulle stesse selezioni OOS. Nessuna modifica alle probabilità operative.
# =====================================================================

V14_13_BOOTSTRAPS = 10000
V14_13_SEED = 20260930


def _v14_13_bootstrap_ci(delta_values, n_boot=V14_13_BOOTSTRAPS, seed=V14_13_SEED):
    """Bootstrap CI percentile per una differenza paired osservata per match."""
    arr = np.asarray(delta_values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi), n


def _v14_13_sign_permutation_p(delta_values, n_perm=V14_13_BOOTSTRAPS, seed=V14_13_SEED):
    """Paired random-sign permutation test della media, H0: differenza=0."""
    arr = np.asarray(delta_values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return np.nan, n
    observed = abs(float(arr.mean()))
    rng = np.random.default_rng(seed + 17)
    chunk = max(1, int(n_perm))
    hits = 0
    done = 0
    while done < int(n_perm):
        m = min(chunk, int(n_perm) - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        perm = np.abs((signs * arr).mean(axis=1))
        hits += int(np.sum(perm >= observed))
        done += m
    p = (hits + 1.0) / (float(n_perm) + 1.0)
    return float(p), n


def _v14_13_compare(g, periodo, n_boot=V14_13_BOOTSTRAPS, seed=V14_13_SEED):
    if g is None or g.empty:
        return None
    y = np.asarray(g['top1_hit'].to_numpy(dtype=float)).reshape(-1)
    raw = np.clip(np.asarray(g['top1_prob_raw'].to_numpy(dtype=float)).reshape(-1), 1e-6, 1.0 - 1e-6)
    platt = np.clip(np.asarray(g['top1_prob_platt'].to_numpy(dtype=float)).reshape(-1), 1e-6, 1.0 - 1e-6)

    d_brier = (platt - y) ** 2 - (raw - y) ** 2
    d_ll = -(y * np.log(platt) + (1.0 - y) * np.log(1.0 - platt)) + (
        y * np.log(raw) + (1.0 - y) * np.log(1.0 - raw)
    )
    # Calibrazione: gap = reale - probabilità; delta PLATT-RAW
    d_gap = 100.0 * float(np.mean(y - platt) - np.mean(y - raw))

    b_mean, b_lo, b_hi, n_b = _v14_13_bootstrap_ci(d_brier, n_boot=n_boot, seed=seed)
    l_mean, l_lo, l_hi, n_l = _v14_13_bootstrap_ci(d_ll, n_boot=n_boot, seed=seed + 1)
    p_b, _ = _v14_13_sign_permutation_p(d_brier, n_perm=n_boot, seed=seed + 2)
    p_l, _ = _v14_13_sign_permutation_p(d_ll, n_perm=n_boot, seed=seed + 3)

    return {
        'Periodo': periodo,
        'N': int(len(g)),
        'N PLATT applicato': int(g['platt_applied'].sum()) if 'platt_applied' in g.columns else int(len(g)),
        'Delta Brier medio pp': 100.0 * b_mean,
        'IC95% Delta Brier low pp': 100.0 * b_lo,
        'IC95% Delta Brier high pp': 100.0 * b_hi,
        'p permutazione Brier': p_b,
        'Delta Log Loss medio': l_mean,
        'IC95% Delta Log Loss low': l_lo,
        'IC95% Delta Log Loss high': l_hi,
        'p permutazione Log Loss': p_l,
        'Delta Gap calibrazione pp': d_gap,
        'Prob. media RAW %': 100.0 * float(raw.mean()),
        'Prob. media PLATT %': 100.0 * float(platt.mean()),
    }


def mostra_v14_13_platt_bootstrap(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.13 — PLATT TOP1: BOOTSTRAP PAIRED')
    st.caption(
        'Solo diagnostico. Usa le stesse selezioni OOS di V14.12-FIX e quantifica '
        'l\'incertezza della differenza RAW vs PLATT con bootstrap paired e test '
        'a randomizzazione dei segni. Nessuna modifica alle probabilità operative.'
    )

    with st.expander('Apri V14.13 — Bootstrap paired RAW vs PLATT', expanded=False):
        ntrain = st.number_input(
            'Training PLATT Top1 (massimo)',
            V14_12_MIN_TRAIN,
            V14_12_MAX_TRAIN,
            100,
            10,
            key='v14_13_train'
        )
        nboot = st.number_input(
            'Numero bootstrap/permutazioni',
            2000,
            50000,
            V14_13_BOOTSTRAPS,
            1000,
            key='v14_13_boots'
        )

        if st.button('🔬 ESEGUI V14.13 — BOOTSTRAP PLATT TOP1', key='v14_13_run'):
            summary_rows = []
            by_league_rows = []
            errors = []

            with st.spinner('Calcolo V14.13 sulle 5 leghe...'):
                for camp, info in CAMPIONATI_DOMESTICI.items():
                    try:
                        res = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                        if res is None:
                            continue
                        _, det = res
                        if det is None or det.empty:
                            continue
                        wf, _, _ = _v14_12_walkforward_top1(
                            det,
                            int(ntrain),
                            V14_12_PROB_THRESHOLD
                        )
                        if wf.empty:
                            continue

                        # Aggregato storico
                        met = _v14_13_compare(wf, 'Tutto lo storico OOS', n_boot=int(nboot), seed=V14_13_SEED)
                        if met is not None:
                            summary_rows.append(met)

                        # Stagione corrente: stessa convenzione V14.12 (dal 1 agosto dell'anno massimo).
                        if 'data' in wf.columns:
                            dates = pd.to_datetime(wf['data'], errors='coerce')
                            if dates.notna().any():
                                latest_year = int(dates.dt.year.max())
                                cut = pd.Timestamp(year=latest_year, month=8, day=1)
                                curr = wf[dates >= cut].copy()
                                metc = _v14_13_compare(curr, 'Stagione corrente OOS', n_boot=int(nboot), seed=V14_13_SEED + 100) if not curr.empty else None
                                if metc is not None:
                                    summary_rows.append(metc)

                        by_league = _v14_13_compare(wf, 'Tutto lo storico OOS', n_boot=int(nboot), seed=V14_13_SEED + 200)
                        if by_league is not None:
                            by_league['Campionato'] = camp
                            by_league_rows.append(by_league)
                            # dettaglio corrente per lega
                            if 'data' in wf.columns:
                                dates = pd.to_datetime(wf['data'], errors='coerce')
                                if dates.notna().any():
                                    latest_year = int(dates.dt.year.max())
                                    cut = pd.Timestamp(year=latest_year, month=8, day=1)
                                    curr = wf[dates >= cut].copy()
                                    curm = _v14_13_compare(curr, 'Stagione corrente OOS', n_boot=int(nboot), seed=V14_13_SEED + 300) if not curr.empty else None
                                    if curm is not None:
                                        curm['Campionato'] = camp
                                        by_league_rows.append(curm)

                    except Exception as e:
                        errors.append((camp, f'{type(e).__name__}: {e}'))

            if summary_rows:
                sdf = pd.DataFrame(summary_rows)
                st.markdown('### 1. Incertezza aggregata')
                st.dataframe(sdf.round(4), use_container_width=True, hide_index=True)
                st.caption(
                    'Delta Brier e Delta Log Loss = PLATT − RAW. Intervalli al 95% ottenuti '
                    'con bootstrap paired; p-value da randomizzazione dei segni. '
                    'Valori negativi della delta indicano una riduzione della metrica.'
                )
                st.info(f'Bootstrap/permutazioni richiesti: {int(nboot):,}. Seed fisso: {V14_13_SEED}.')

            if by_league_rows:
                ldf = pd.DataFrame(by_league_rows)
                st.markdown('### 2. Dettaglio per campionato')
                st.dataframe(ldf.round(4), use_container_width=True, hide_index=True)

            if errors:
                st.warning('Campionati non completati:')
                for nome, msg in errors:
                    st.write(f'- **{nome}**: {msg}')


mostra_v14_13_platt_bootstrap(rho_val, ewma_span_val, emivita_val)



# =====================================================================
# 🧪 V14.14 — SENSIBILITÀ ALLA FINESTRA DI TRAINING PLATT TOP1
# Solo diagnostico. Confronta la stessa metodologia walk-forward Top1 >=35%
# usando tre lunghezze massime di training: 50, 75, 100 osservazioni.
# Nessuna modifica alle probabilità operative.
# =====================================================================

V14_14_TRAIN_WINDOWS = [50, 75, 100]


def _v14_14_aggregate_metrics(frames, period_label):
    """Concatena i dettagli selezionati e calcola metriche pooled, esattamente match-level."""
    if not frames:
        return None
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return None
    g = pd.concat(valid, ignore_index=True)
    return _v14_12_metrics(g, period_label)


def _v14_14_current_filter(wf):
    if wf is None or wf.empty or 'data' not in wf.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(wf['data'], errors='coerce')
    if not dates.notna().any():
        return pd.DataFrame()
    latest_year = int(dates.dt.year.max())
    cut = pd.Timestamp(year=latest_year, month=8, day=1)
    return wf[dates >= cut].copy()


def mostra_v14_14_training_window_sensitivity(rho, ewma_span, emivita):
    st.divider()
    st.markdown('## 🧪 V14.14 — SENSIBILITÀ FINESTRA TRAINING PLATT TOP1')
    st.caption(
        'Solo diagnostico. Soglia Top1 >=35% fissa ex ante. Confronta il walk-forward '
        'PLATT con finestra massima 50, 75 e 100 osservazioni precedenti nello stesso '
        'campionato. Nessuna modifica alle probabilità operative.'
    )

    with st.expander('Apri V14.14 — Training Window Sensitivity', expanded=False):
        if st.button('🔬 ESEGUI V14.14 — SENSIBILITÀ TRAINING PLATT', key='v14_14_run'):
            aggregate_rows = []
            league_rows = []
            temporal_rows = []
            errors = []

            with st.spinner('Calcolo sensibilità della finestra di training sulle 5 leghe...'):
                for ntrain in V14_14_TRAIN_WINDOWS:
                    all_hist = []
                    all_curr = []

                    for camp, info in CAMPIONATI_DOMESTICI.items():
                        try:
                            res = _v14_combo_leg(info['id_fd'], rho, ewma_span, emivita, False)
                            if res is None:
                                continue
                            _, det = res
                            if det is None or det.empty:
                                continue

                            wf, _, _ = _v14_12_walkforward_top1(
                                det,
                                int(ntrain),
                                V14_12_PROB_THRESHOLD
                            )
                            if wf is None or wf.empty:
                                continue

                            mh = _v14_12_metrics(wf, 'Tutto lo storico OOS')
                            if mh is not None:
                                mh['Training max'] = int(ntrain)
                                mh['Campionato'] = camp
                                league_rows.append(mh)

                            all_hist.append(wf.assign(Campionato=camp))
                            curr = _v14_14_current_filter(wf)
                            if not curr.empty:
                                all_curr.append(curr.assign(Campionato=camp))

                        except Exception as e:
                            errors.append((camp, f'Training {ntrain}: {type(e).__name__}: {e}'))

                    mh_all = _v14_14_aggregate_metrics(all_hist, 'Tutto lo storico OOS')
                    if mh_all is not None:
                        mh_all['Training max'] = int(ntrain)
                        aggregate_rows.append(mh_all)

                    mc_all = _v14_14_aggregate_metrics(all_curr, 'Stagione corrente OOS')
                    if mc_all is not None:
                        mc_all['Training max'] = int(ntrain)
                        aggregate_rows.append(mc_all)

            if aggregate_rows:
                adf = pd.DataFrame(aggregate_rows)
                cols = [
                    'Periodo', 'Training max', 'N selezioni >=35%', 'N PLATT applicato',
                    'Top1 hit %', 'RAW Prob. media %', 'PLATT Prob. media %',
                    'RAW Gap calibrazione pp', 'PLATT Gap calibrazione pp',
                    'RAW Brier', 'PLATT Brier', 'RAW Log Loss', 'PLATT Log Loss',
                    'Delta Brier pp', 'Delta Log Loss'
                ]
                adf = adf[[c for c in cols if c in adf.columns]]
                st.markdown('### 1. Sensibilità aggregata — storico vs stagione corrente')
                st.dataframe(adf.round(4), use_container_width=True, hide_index=True)
                st.caption('Delta Brier e Delta Log Loss = PLATT − RAW. Il training usa solo osservazioni precedenti nello stesso campionato.')

            if league_rows:
                ldf = pd.DataFrame(league_rows)
                cols = [
                    'Campionato', 'Training max', 'Periodo', 'N selezioni >=35%',
                    'N PLATT applicato', 'Top1 hit %', 'RAW Prob. media %',
                    'PLATT Prob. media %', 'RAW Gap calibrazione pp',
                    'PLATT Gap calibrazione pp', 'RAW Brier', 'PLATT Brier',
                    'RAW Log Loss', 'PLATT Log Loss', 'Delta Brier pp',
                    'Delta Log Loss'
                ]
                st.markdown('### 2. Dettaglio per campionato — storico OOS')
                st.dataframe(ldf[[c for c in cols if c in ldf.columns]].round(4), use_container_width=True, hide_index=True)

            if errors:
                st.warning('Campionati non completati:')
                for nome, msg in errors:
                    st.write(f'- **{nome}**: {msg}')


mostra_v14_14_training_window_sensitivity(rho_val, ewma_span_val, emivita_val)
