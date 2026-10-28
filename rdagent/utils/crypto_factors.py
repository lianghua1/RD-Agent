"""Cryptocurrency factor library.

Written for a 7x24, price-and-volume-only market. Deliberate choices:

- **No ``$vwap``**: crypto OHLCV has no VWAP column, so any factor built on it
  would fail at execution time.
- **Log returns / log volume**: crypto returns are fat-tailed and volume spans
  orders of magnitude across coins, so raw levels are unstable.
- **Short windows (1-30 bars)**: crypto regimes shift far faster than equities;
  60+ day windows mostly encode stale regimes.
- **No fundamentals**: there are no shares outstanding, industry or financial
  statements; factors must be derivable from OHLCV alone.
- Only Qlib built-in operators that are already exercised by ``ALPHA20`` /
  ``ALPHA158`` are used, so the expression engine accepts every entry.

``CRYPTO_BASE`` is the default base-feature set injected into the R&D loop;
``CRYPTO_ALL`` is the full library available to the model.
"""

CRYPTO_ALL: dict[str, str] = {
    # -- candle structure ---------------------------------------------------
    "C_BODY": "($close-$open)/$open",
    "C_RANGE": "($high-$low)/$open",
    "C_BODY_RATIO": "($close-$open)/($high-$low+1e-12)",
    "C_CLOSE_POS": "(2*$close-$high-$low)/($high-$low+1e-12)",
    "C_UPPER_WICK": "($high-Greater($open,$close))/$open",
    "C_LOWER_WICK": "(Less($open,$close)-$low)/$open",
    "C_GAP": "$open/Ref($close,1)-1",
    # -- return momentum / reversal ----------------------------------------
    # All return factors use the ``current/ref - 1`` convention so a positive
    # value always means the price rose over the window (as in ``R_RET1`` and
    # ``R_LRET*``). The upstream ALPHA158 ``ROC`` uses the opposite
    # ``Ref/$close - 1`` form, which would silently flip every momentum sign.
    "R_RET1": "$close/Ref($close,1)-1",
    "R_ROC5": "$close/Ref($close,5)-1",
    "R_ROC10": "$close/Ref($close,10)-1",
    "R_ROC20": "$close/Ref($close,20)-1",
    "R_ROC30": "$close/Ref($close,30)-1",
    "R_LRET5": "Log($close/Ref($close,5))",
    "R_LRET10": "Log($close/Ref($close,10))",
    "R_LRET20": "Log($close/Ref($close,20))",
    "R_ACCEL": "($close/Ref($close,5)-1)-(Ref($close,5)/Ref($close,10)-1)",
    # -- volatility ---------------------------------------------------------
    "V_STD5": "Std(Log($close/Ref($close,1)),5)",
    "V_STD10": "Std(Log($close/Ref($close,1)),10)",
    "V_STD20": "Std(Log($close/Ref($close,1)),20)",
    "V_STD30": "Std(Log($close/Ref($close,1)),30)",
    "V_RANGE5": "Mean(($high-$low)/$close,5)",
    "V_RANGE10": "Mean(($high-$low)/$close,10)",
    "V_HL10": "Std(Log($high/($low+1e-12)),10)",
    "V_VOLOFVOL10": "Std(Std(Log($close/Ref($close,1)),5),10)",
    # -- trend --------------------------------------------------------------
    "T_MA5": "Mean($close,5)/$close-1",
    "T_MA10": "Mean($close,10)/$close-1",
    "T_MA20": "Mean($close,20)/$close-1",
    "T_MA30": "Mean($close,30)/$close-1",
    "T_SLOPE10": "Slope($close,10)/$close",
    "T_SLOPE20": "Slope($close,20)/$close",
    "T_RSQR10": "Rsquare($close,10)",
    "T_RSQR20": "Rsquare($close,20)",
    "T_DIST_HIGH20": "$close/(Max($high,20)+1e-12)-1",
    "T_DIST_LOW20": "$close/(Min($low,20)+1e-12)-1",
    # -- volume level / liquidity ------------------------------------------
    "Q_LVOL": "Log($volume+1)",
    "Q_VMA5": "Mean($volume,5)/($volume+1e-12)",
    "Q_VMA20": "Mean($volume,20)/($volume+1e-12)",
    "Q_VSTD10": "Std($volume,10)/(Mean($volume,10)+1e-12)",
    "Q_VOLZ20": "($volume-Mean($volume,20))/(Std($volume,20)+1e-12)",
    "Q_VPCORR10": "Corr(Log($volume+1),$close,10)",
    "Q_VRCORR10": "Corr(Log($volume+1),Log($close/Ref($close,1)),10)",
    "Q_AMIHUD10": "Mean(Abs($close/Ref($close,1)-1)/($volume+1e-12),10)",
    # -- volume structure ---------------------------------------------------
    "Q_UPVOL5": "Sum(Greater($close-Ref($close,1),0)*$volume,5)/(Sum($volume,5)+1e-12)",
    "Q_DNVOL5": "Sum(Greater(Ref($close,1)-$close,0)*$volume,5)/(Sum($volume,5)+1e-12)",
    "Q_UPVOL20": "Sum(Greater($close-Ref($close,1),0)*$volume,20)/(Sum($volume,20)+1e-12)",
    "Q_DNVOL20": "Sum(Greater(Ref($close,1)-$close,0)*$volume,20)/(Sum($volume,20)+1e-12)",
}

#: Default base features handed to the loop. A diverse, cheap subset: one or two
#: representatives per category so the first rounds have something to combine.
CRYPTO_BASE: dict[str, str] = {
    key: CRYPTO_ALL[key]
    for key in (
        "C_BODY",
        "C_RANGE",
        "C_CLOSE_POS",
        "C_GAP",
        "R_RET1",
        "R_ROC5",
        "R_ROC20",
        "R_LRET10",
        "R_ACCEL",
        "V_STD10",
        "V_STD20",
        "V_RANGE10",
        "V_VOLOFVOL10",
        "T_MA20",
        "T_SLOPE20",
        "T_RSQR20",
        "T_DIST_HIGH20",
        "Q_LVOL",
        "Q_VMA20",
        "Q_VOLZ20",
        "Q_VRCORR10",
        "Q_AMIHUD10",
    )
}

__all__ = ["CRYPTO_ALL", "CRYPTO_BASE"]
