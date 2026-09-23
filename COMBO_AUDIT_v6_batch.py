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

st.set_page_config(page_title="COMBO — Audit Model v5", page_icon="🔎", layout="centered")

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
st.title("🧪 COMBO — Audit Model v6")


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
            v5raw12=_v5_stats_df(v5['raw12']) if v5 else _v6_stats_df(None)
            v5cal12=_v5_stats_df(v5['cal12']) if v5 else _v6_stats_df(None)
            v5rawou=_v5_stats_df(v5['rawou']) if v5 else _v6_stats_df(None)
            v5calou=_v5_stats_df(v5['calou']) if v5 else _v6_stats_df(None)

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


st.caption("Versione separata per testare calibrazione fuori campione senza modificare v1")

st.info("Questa app è un laboratorio separato. v2 confronta il modello grezzo con una calibrazione isotonic allenata SOLO su dati temporali precedenti al test: non usa i calibratori salvati del progetto originale e non modifica v1.")

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
        "🎯 Applica calibrazione (se allenata per questo campionato)",
        value=False,
        help="In questa versione audit è disattivata di default: sulla base della verifica storica. Disattiva per confrontare prima/dopo."
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

    with st.expander("🧪 AUDIT RIGOROSO — modello grezzo", expanded=True):
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

    with st.expander("🧪 V2 — Confronto calibrazione fuori campione", expanded=True):
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

    if modello is None:
        st.warning(f"⚠️ Impossibile elaborare il match per **{partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}**.")
    else:
        st.subheader(f"📊 Analisi Match: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        if calib_1x2_info:
            st.caption(f"🎯 Probabilità 1X2 corrette con calibrazione (allenata su "
                       f"{calib_1x2_info['n_osservazioni']} osservazioni, {calib_1x2_info['timestamp']}).")

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
