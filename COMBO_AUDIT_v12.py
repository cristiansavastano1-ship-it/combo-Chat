
    path = _path_calibratore(id_fd, chiave)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None
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
    usa_platt_ou_v13 = st.checkbox(
        "🎯 V13 — PLATT solo O/U 2.5",
        value=True,
        help="Calibra solo Over/Under 2.5 con PLATT OOS; 1X2 e motore del modello restano invariati."
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

    calib_ou_v13_info = None
    if modello is not None and not is_coppa and usa_platt_ou_v13:
        modello, calib_ou_v13_info = v13_applica_platt_ou(
            modello, dati_filtrati, rho_val, ewma_span_val, emivita_val
        )

    if modello is None:
        st.warning(f"⚠️ Impossibile elaborare il match per **{partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}**.")
    else:
        st.subheader(f"📊 Analisi Match: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        if calib_1x2_info:
            st.caption(f"🎯 Probabilità 1X2 corrette con calibrazione (allenata su "
                       f"{calib_1x2_info['n_osservazioni']} osservazioni, {calib_1x2_info['timestamp']}).")
        if calib_ou_v13_info:
            st.caption(
                f"🎯 V13 PLATT O/U 2.5 attiva — {calib_ou_v13_info['n_osservazioni']} osservazioni storiche OOS; "
                f"Over raw {calib_ou_v13_info['p_raw_over']*100:.1f}% → calibrato {calib_ou_v13_info['p_cal_over']*100:.1f}%."
            )

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

# V12.1 CPU optimization: keep V11 functions intact, but do not execute the
# V11 validation automatically. Running both V11 and V12 on every Streamlit
# rerun unnecessarily doubles the heavy historical/OOS computation.
# The original V11 source remains untouched in its own file.


# =====================================================================
# V12 — CONFRONTO OOS DI METODI DI CALIBRAZIONE
# =====================================================================
# V12 è una derivazione separata dal V11.
# Non modifica il motore RAW.
#
# Metodi confrontati:
#   RAW       = probabilità del modello senza calibrazione
#   ISOTONIC  = Isotonic Regression walk-forward
#   PLATT     = Logistic/Platt scaling walk-forward
#   BETA      = Beta calibration walk-forward
#   SHRINK50  = shrinkage verso 50%, con intensità alpha stimata sul solo passato
#
# Regola anti-leakage:
# per ogni partita i calibratori sono allenati SOLO sulle osservazioni
# precedenti. La partita corrente viene valutata prima di essere aggiunta
# al training successivo.
#
# 1X2: calibrazione one-vs-rest separata per 1/X/2, poi normalizzazione a 1.
# O/U: calibrazione binaria Over; Under = 1 - Over.
#
# V12 mantiene anche il confronto economico EV >= 10% per continuità con V11,
# ma ROI/profit sono proxy storiche basate sulle quote medie dei CSV.
# =====================================================================

from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

V12_CAL_MIN_TRAIN = 100
V12_METHODS = ['RAW', 'ISOTONIC', 'PLATT', 'BETA', 'SHRINK50']



# =====================================================================
# V13 OPERATIVA — PLATT SOLO O/U 2.5
# Applicata esclusivamente alle probabilita O/U 2.5 della partita selezionata.
# Il motore V12 e il mercato 1X2 restano invariati.
# =====================================================================
@st.cache_data(ttl=3600, max_entries=10)
def v13_platt_ou_calibratore(dati_storici, rho, ewma_span, emivita):
    if dati_storici is None or dati_storici.empty:
        return None, 0
    hist = _v5_raw_history(dati_storici, rho, ewma_span, emivita)
    if hist is None or hist.empty or len(hist) < V12_CAL_MIN_TRAIN:
        return None, 0
    p = hist['po'].astype(float).tolist()
    y = (hist['esito_ou'] == 'Over').astype(int).tolist()
    cal = _v12_fit_platt(p, y)
    return cal, len(hist)


def v13_applica_platt_ou(modello, dati_storici, rho, ewma_span, emivita):
    if modello is None or dati_storici is None or dati_storici.empty:
        return modello, None
    cal, n = v13_platt_ou_calibratore(dati_storici, rho, ewma_span, emivita)
    if cal is None:
        return modello, None
    p_raw_over = (100.0 - float(modello['prob_under'][2.5])) / 100.0
    p_cal_over = _v12_apply_binary(cal, p_raw_over)
    p_cal_over = float(np.clip(p_cal_over, 0.001, 0.999))
    modello['prob_under'][2.5] = (1.0 - p_cal_over) * 100.0
    return modello, {'n_osservazioni': n, 'p_raw_over': p_raw_over, 'p_cal_over': p_cal_over}

def _v12_clip(p):
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def _v12_logloss_binary(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def _v12_fit_isotonic(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    cal.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return cal


def _v12_fit_platt(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    x = np.asarray(p, dtype=float).reshape(-1, 1)
    yy = np.asarray(y, dtype=int)
    # Platt scaling: logistic regression on the raw probability.
    cal = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    cal.fit(x, yy)
    return cal


def _v12_fit_beta(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    x = np.column_stack([np.log(p), np.log1p(-p)])
    yy = np.asarray(y, dtype=int)
    # Beta calibration in the canonical logistic form:
    # logit(p_cal) = a*log(p) + b*log(1-p) + c.
    cal = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    cal.fit(x, yy)
    return cal


def _v12_apply_binary(cal, p):
    if cal is None:
        return float(p)
    p = _v12_clip(p)
    if isinstance(cal, tuple) and cal and cal[0] == 'shrink50':
        alpha = float(cal[1])
        return float(np.clip(0.5 + alpha*(p-0.5), 0.001, 0.999))
    if isinstance(cal, tuple) and cal and cal[0] == 'beta':
        model = cal[1]
        x = np.array([[np.log(p), np.log1p(-p)]], dtype=float)
        return float(np.clip(model.predict_proba(x)[0,1], 0.001, 0.999))
    # Isotonic or Platt
    if isinstance(cal, IsotonicRegression):
        return float(np.clip(cal.predict([p])[0], 0.001, 0.999))
    return float(np.clip(cal.predict_proba(np.array([[p]], dtype=float))[0,1], 0.001, 0.999))


def _v12_fit_shrink50(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # alpha=0 -> 50%; alpha=1 -> RAW.
    def objective(alpha):
        pc = np.clip(0.5 + alpha*(p-0.5), 1e-6, 1-1e-6)
        return _v12_logloss_binary(pc, y)

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded',
                          options={'xatol': 1e-4})
    alpha = float(np.clip(res.x if res.success else 1.0, 0.0, 1.0))
    return ('shrink50', alpha)


def _v12_fit_method(name, p, y):
    if name == 'ISOTONIC':
        return _v12_fit_isotonic(p, y)
    if name == 'PLATT':
        return _v12_fit_platt(p, y)
    if name == 'BETA':
        fitted = _v12_fit_beta(p, y)
        return ('beta', fitted) if fitted is not None else None
    if name == 'SHRINK50':
        return _v12_fit_shrink50(p, y)
    return None


def _v12_apply_1x2(rawp, models):
    if models is None or not all(models.get(s) is not None for s in ['1','X','2']):
        return rawp.copy()
    cp = {s: _v12_apply_binary(models[s], rawp[s]) for s in ['1','X','2']}
    tot = sum(cp.values())
    if tot <= 0:
        return rawp.copy()
    return {s: cp[s]/tot for s in ['1','X','2']}


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def _v12_build_paired(dati_completi, rho, ewma_span, emivita,
                      soglia_ev=0.10, warmup=100):
    """Costruisce un dataset OOS paired per RAW + 4 calibratori."""
    hist = _v5_raw_history(dati_completi, rho, ewma_span, emivita)
    if hist.empty:
        return None

    rows12, rowsou = [], []

    prior12_p = {'1': [], 'X': [], '2': []}
    prior12_y = {'1': [], 'X': [], '2': []}
    prior_ou_p, prior_ou_y = [], []

    for j, r in hist.iterrows():
        if j < warmup:
            prior12_p['1'].append(float(r.p1)); prior12_y['1'].append(int(r.esito12 == '1'))
            prior12_p['X'].append(float(r.px)); prior12_y['X'].append(int(r.esito12 == 'X'))
            prior12_p['2'].append(float(r.p2)); prior12_y['2'].append(int(r.esito12 == '2'))
            prior_ou_p.append(float(r.po)); prior_ou_y.append(int(r.esito_ou == 'Over'))
            continue

        raw12 = {'1': float(r.p1), 'X': float(r.px), '2': float(r.p2)}

        models12 = {
            method: {
                s: _v12_fit_method(method, prior12_p[s], prior12_y[s])
                for s in ['1','X','2']
            }
            for method in V12_METHODS if method != 'RAW'
        }
        probs12 = {'RAW': raw12.copy()}
        for method in V12_METHODS:
            if method == 'RAW':
                continue
            probs12[method] = _v12_apply_1x2(raw12, models12[method])

        rq = {'1': float(r.q1), 'X': float(r.qx), '2': float(r.q2)}
        row12 = {
            'data': r.data, 'esito': r.esito12,
            'p1_raw': raw12['1'], 'px_raw': raw12['X'], 'p2_raw': raw12['2']
        }

        for method in V12_METHODS:
            pp = probs12[method]
            choice = max(pp, key=pp.get)
            ev = pp[choice] * rq[choice] - 1.0
            bet = bool(ev >= soglia_ev)
            profit = ((rq[choice]-1.0) if choice == r.esito12 else -1.0) if bet else 0.0
            row12.update({
                f'{method.lower()}_p1': pp['1'],
                f'{method.lower()}_px': pp['X'],
                f'{method.lower()}_p2': pp['2'],
                f'{method.lower()}_conf': pp[choice],
                f'{method.lower()}_choice': choice,
                f'{method.lower()}_hit': bool(choice == r.esito12),
                f'{method.lower()}_ev': ev,
                f'{method.lower()}_in': bet,
                f'{method.lower()}_profit': profit,
                f'{method.lower()}_quota': rq[choice],
            })
        rows12.append(row12)

        rawou = {'Over': float(r.po), 'Under': float(r.pu)}
        ou_models = {
            method: _v12_fit_method(method, prior_ou_p, prior_ou_y)
            for method in V12_METHODS if method != 'RAW'
        }
        probsou = {'RAW': rawou.copy()}
        for method in V12_METHODS:
            if method == 'RAW':
                continue
            po_cal = _v12_apply_binary(ou_models[method], rawou['Over'])
            probsou[method] = {'Over': po_cal, 'Under': 1.0-po_cal}

        oq = {'Over': float(r.qo), 'Under': float(r.qu)}
        rowou = {'data': r.data, 'esito': r.esito_ou,
                  'po_raw': rawou['Over'], 'pu_raw': rawou['Under']}

        for method in V12_METHODS:
            pp = probsou[method]
            choice = max(pp, key=pp.get)
            ev = pp[choice] * oq[choice] - 1.0
            bet = bool(ev >= soglia_ev)
            profit = ((oq[choice]-1.0) if choice == r.esito_ou else -1.0) if bet else 0.0
            rowou.update({
                f'{method.lower()}_po': pp['Over'],
                f'{method.lower()}_pu': pp['Under'],
                f'{method.lower()}_conf': pp[choice],
                f'{method.lower()}_choice': choice,
                f'{method.lower()}_hit': bool(choice == r.esito_ou),
                f'{method.lower()}_ev': ev,
                f'{method.lower()}_in': bet,
                f'{method.lower()}_profit': profit,
                f'{method.lower()}_quota': oq[choice],
            })
        rowsou.append(rowou)

        # IMPORTANT: current observation enters training only AFTER evaluation.
        prior12_p['1'].append(float(r.p1)); prior12_y['1'].append(int(r.esito12 == '1'))
        prior12_p['X'].append(float(r.px)); prior12_y['X'].append(int(r.esito12 == 'X'))
        prior12_p['2'].append(float(r.p2)); prior12_y['2'].append(int(r.esito12 == '2'))
        prior_ou_p.append(float(r.po)); prior_ou_y.append(int(r.esito_ou == 'Over'))

    return {'1X2': pd.DataFrame(rows12), 'O/U 2.5': pd.DataFrame(rowsou)}


def _v12_metriche(df, merc, common_only=False):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df.raw_in & df.cal_dummy].copy() if False else df.copy()
    if common_only:
        mask = np.ones(len(df), dtype=bool)
        for m in V12_METHODS:
            mask &= df[f'{m.lower()}_in'].to_numpy(dtype=bool)
        d = df.loc[mask].copy()
    if d.empty:
        return pd.DataFrame()

    out = []
    for method in V12_METHODS:
        key = method.lower()
        hits = d[f'{key}_hit'].to_numpy(dtype=bool)
        n = len(d)
        probs = []
        true_probs = []
        brier = []
        logloss = []
        for _, r in d.iterrows():
            if merc == '1X2':
                ps = np.array([r[f'{key}_p1'], r[f'{key}_px'], r[f'{key}_p2']], dtype=float)
                y = np.array([r.esito == '1', r.esito == 'X', r.esito == '2'], dtype=float)
                true_p = float(ps[int(np.argmax(y))])
                brier.append(float(np.sum((ps-y)**2)))
            else:
                po = float(r[f'{key}_po'])
                ps = {'Over': po, 'Under': 1-po}
                true_p = ps[r.esito]
                brier.append(float((po-(r.esito == 'Over'))**2 +
                                   ((1-po)-(r.esito == 'Under'))**2))
            true_probs.append(true_p)
            logloss.append(-np.log(_v12_clip(true_p)))

        profit = d[f'{key}_profit'].to_numpy(dtype=float)
        bets = d[f'{key}_in'].to_numpy(dtype=bool)
        nb = int(bets.sum())
        out.append({
            'metodo': method,
            'n': n,
            'accuracy_pct': float(hits.mean()*100),
            'brier': float(np.mean(brier)),
            'log_loss': float(np.mean(logloss)),
            'calibration_mae_pp': float(np.mean(np.abs(
                np.asarray(true_probs)*100 -
                np.asarray(hits, dtype=float)*100
            ))),
            'bets_ev10': nb,
            'profit_ev10_u': float(profit.sum()),
            'roi_ev10_pct': float(100*profit.sum()/nb) if nb else np.nan
        })
    return pd.DataFrame(out)


def _v12_band_table(df, merc, common_only=False):
    if df is None or df.empty:
        return pd.DataFrame()
    if common_only:
        mask = np.ones(len(df), dtype=bool)
        for m in V12_METHODS:
            mask &= df[f'{m.lower()}_in'].to_numpy(dtype=bool)
        d = df.loc[mask].copy()
    else:
        d = df.copy()
    if d.empty:
        return pd.DataFrame()

    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0000001]
    labels = ['0-40%','40-50%','50-60%','60-70%','70-80%','80-100%']
    out = []

    for method in V12_METHODS:
        key = method.lower()
        conf = np.asarray(d[f'{key}_conf'], dtype=float)
        hit = np.asarray(d[f'{key}_hit'], dtype=bool)
        idx = np.digitize(conf, bins[1:-1], right=False)
        for k, label in enumerate(labels):
            mask = idx == k
            n = int(mask.sum())
            if not n:
                continue
            p = conf[mask]
            h = hit[mask]
            profit = np.asarray(d.loc[mask, f'{key}_profit'], dtype=float)
            bets = np.asarray(d.loc[mask, f'{key}_in'], dtype=bool)
            nb = int(bets.sum())
            out.append({
                'metodo': method,
                'fascia': label,
                'n_opportunita': n,
                'prob_media': float(p.mean()*100),
                'frequenza_reale': float(h.mean()*100),
                'errore_cal_pp': float(h.mean()*100-p.mean()*100),
                'bets_ev10': nb,
                'roi_ev10_pct': float(100*profit.sum()/nb) if nb else np.nan,
                'profit_ev10_u': float(profit.sum())
            })
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=10)
def _v12_build_all(warmup=100):
    per_metriche = []
    per_bands = []
    dfs = {'1X2': {}, 'O/U 2.5': {}}
    errors = []

    for camp, info in CAMPIONATI_DOMESTICI.items():
        try:
            dati = carica_dati_campionato(info['id_fd'])
            paired = _v12_build_paired(
                dati, rho_val, ewma_span_val, emivita_val, 0.10, warmup
            )
            if paired is None:
                raise ValueError('dati OOS vuoti')

            for merc, df in paired.items():
                dfs[merc][camp] = df
                for common in [False, True]:
                    m = _v12_metriche(df, merc, common)
                    if not m.empty:
                        m.insert(0, 'campionato', camp)
                        m.insert(1, 'mercato', merc)
                        m.insert(2, 'universo',
                                  'COMMON BETS' if common else 'ALL OOS')
                        per_metriche.append(m)

                    b = _v12_band_table(df, merc, common)
                    if not b.empty:
                        b.insert(0, 'campionato', camp)
                        b.insert(1, 'mercato', merc)
                        b.insert(2, 'universo',
                                  'COMMON BETS' if common else 'ALL OOS')
                        per_bands.append(b)
        except Exception as e:
            errors.append((camp, f'{type(e).__name__}: {e}'))

    metrics = pd.concat(per_metriche, ignore_index=True) if per_metriche else pd.DataFrame()
    bands = pd.concat(per_bands, ignore_index=True) if per_bands else pd.DataFrame()
    return metrics, bands, errors


def mostra_v12_validation():
    st.divider()
    st.markdown('## 🧬 V12.1 — CONFRONTO 5 METODI DI CALIBRAZIONE OOS · CPU OPTIMIZED')
    st.caption(
        'RAW vs Isotonic vs Platt vs Beta vs Shrink50. Cache attiva e V11 non viene eseguito automaticamente. '
        'Tutti i calibratori sono walk-forward: per ogni partita usano solo il passato. '
        '1X2 e O/U 2.5 sono valutati separatamente.'
    )

    c1, c2 = st.columns(2)
    with c1:
        wu = st.number_input('Warm-up V12', 50, 300, 100, 10, key='v12_warm')
    with c2:
        st.metric('Metodi', '5')

    st.info(
        'Brier e Log Loss più bassi indicano probabilità più accurate. '
        'Errore calibrazione vicino a 0 indica migliore allineamento tra probabilità '
        'e frequenza osservata. ROI/profit sono proxy storiche sulle selezioni EV≥10%. '
        'Le fasce alte possono avere campioni piccoli.'
    )

    if st.button('🧬 ESEGUI V12 — CONFRONTO 5 LEGHE', key='v12_run'):
        try:
            with st.spinner('V12.1 in esecuzione: 5 metodi × 5 leghe... primo run può essere pesante; i successivi usano cache.'):
                metrics, bands, errors = _v12_build_all(int(wu))

            if not metrics.empty:
                st.markdown('### 1. Metriche OOS per lega')
                st.dataframe(
                    metrics[
                        ['campionato','mercato','universo','metodo','n',
                         'accuracy_pct','brier','log_loss',
                         'calibration_mae_pp','bets_ev10',
                         'roi_ev10_pct','profit_ev10_u']
                    ],
                    use_container_width=True, hide_index=True
                )
                st.download_button(
                    '⬇️ Scarica V12 metriche',
                    data=metrics.to_csv(index=False).encode('utf-8'),
                    file_name='V12_metriche_metodi.csv',
                    mime='text/csv', key='v12_dl_metrics'
                )

            if not bands.empty:
                st.markdown('### 2. Reliability per fasce di probabilità')
                st.dataframe(bands, use_container_width=True, hide_index=True)
                st.download_button(
                    '⬇️ Scarica V12 fasce',
                    data=bands.to_csv(index=False).encode('utf-8'),
                    file_name='V12_reliability_fasce.csv',
                    mime='text/csv', key='v12_dl_bands'
                )

            if errors:
                st.error('Campionati non completati:')
                for nome, msg in errors:
                    st.write(f'- **{nome}**: {msg}')

        except Exception as e:
            st.error(f'Errore V12: {type(e).__name__}: {e}')


try:
    mostra_v12_validation()
except Exception as _v12_err:
    st.error(f"V12 non disponibile: {type(_v12_err).__name__}: {_v12_err}")
