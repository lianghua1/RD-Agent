import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv

ALPHA20 = {
    "RESI5": "Resi($close, 5)/$close",
    "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "RSQR5": "Rsquare($close, 5)",
    "KLEN": "($high-$low)/$open",
    "RSQR10": "Rsquare($close, 10)",
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "ROC60": "Ref($close, 60)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "VSTD5": "Std($volume, 5)/($volume+1e-12)",
    "RSQR60": "Rsquare($close, 60)",
    "CORR60": "Corr($close, Log($volume+1), 60)",
    "WVMA60": "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
    "STD5": "Std($close, 5)/$close",
    "RSQR20": "Rsquare($close, 20)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "KLOW": "(Less($open, $close)-$low)/$open",
}

ALPHA158 = {
    "KMID": "($close-$open)/$open",
    "KLEN": "($high-$low)/$open",
    "KMID2": "($close-$open)/($high-$low+1e-12)",
    "KUP": "($high-Greater($open, $close))/$open",
    "KUP2": "($high-Greater($open, $close))/($high-$low+1e-12)",
    "KLOW": "(Less($open, $close)-$low)/$open",
    "KLOW2": "(Less($open, $close)-$low)/($high-$low+1e-12)",
    "KSFT": "(2*$close-$high-$low)/$open",
    "KSFT2": "(2*$close-$high-$low)/($high-$low+1e-12)",
    "OPEN0": "$open/$close",
    "HIGH0": "$high/$close",
    "LOW0": "$low/$close",
    "VWAP0": "$vwap/$close",
    "ROC5": "Ref($close, 5)/$close",
    "ROC10": "Ref($close, 10)/$close",
    "ROC20": "Ref($close, 20)/$close",
    "ROC30": "Ref($close, 30)/$close",
    "ROC60": "Ref($close, 60)/$close",
    "MA5": "Mean($close, 5)/$close",
    "MA10": "Mean($close, 10)/$close",
    "MA20": "Mean($close, 20)/$close",
    "MA30": "Mean($close, 30)/$close",
    "MA60": "Mean($close, 60)/$close",
    "STD5": "Std($close, 5)/$close",
    "STD10": "Std($close, 10)/$close",
    "STD20": "Std($close, 20)/$close",
    "STD30": "Std($close, 30)/$close",
    "STD60": "Std($close, 60)/$close",
    "BETA5": "Slope($close, 5)/$close",
    "BETA10": "Slope($close, 10)/$close",
    "BETA20": "Slope($close, 20)/$close",
    "BETA30": "Slope($close, 30)/$close",
    "BETA60": "Slope($close, 60)/$close",
    "RSQR5": "Rsquare($close, 5)",
    "RSQR10": "Rsquare($close, 10)",
    "RSQR20": "Rsquare($close, 20)",
    "RSQR30": "Rsquare($close, 30)",
    "RSQR60": "Rsquare($close, 60)",
    "RESI5": "Resi($close, 5)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "RESI20": "Resi($close, 20)/$close",
    "RESI30": "Resi($close, 30)/$close",
    "RESI60": "Resi($close, 60)/$close",
    "MAX5": "Max($high, 5)/$close",
    "MAX10": "Max($high, 10)/$close",
    "MAX20": "Max($high, 20)/$close",
    "MAX30": "Max($high, 30)/$close",
    "MAX60": "Max($high, 60)/$close",
    "MIN5": "Min($low, 5)/$close",
    "MIN10": "Min($low, 10)/$close",
    "MIN20": "Min($low, 20)/$close",
    "MIN30": "Min($low, 30)/$close",
    "MIN60": "Min($low, 60)/$close",
    "QTLU5": "Quantile($close, 5, 0.8)/$close",
    "QTLU10": "Quantile($close, 10, 0.8)/$close",
    "QTLU20": "Quantile($close, 20, 0.8)/$close",
    "QTLU30": "Quantile($close, 30, 0.8)/$close",
    "QTLU60": "Quantile($close, 60, 0.8)/$close",
    "QTLD5": "Quantile($close, 5, 0.2)/$close",
    "QTLD10": "Quantile($close, 10, 0.2)/$close",
    "QTLD20": "Quantile($close, 20, 0.2)/$close",
    "QTLD30": "Quantile($close, 30, 0.2)/$close",
    "QTLD60": "Quantile($close, 60, 0.2)/$close",
    "RANK5": "Rank($close, 5)",
    "RANK10": "Rank($close, 10)",
    "RANK20": "Rank($close, 20)",
    "RANK30": "Rank($close, 30)",
    "RANK60": "Rank($close, 60)",
    "RSV5": "($close-Min($low, 5))/(Max($high, 5)-Min($low, 5)+1e-12)",
    "RSV10": "($close-Min($low, 10))/(Max($high, 10)-Min($low, 10)+1e-12)",
    "RSV20": "($close-Min($low, 20))/(Max($high, 20)-Min($low, 20)+1e-12)",
    "RSV30": "($close-Min($low, 30))/(Max($high, 30)-Min($low, 30)+1e-12)",
    "RSV60": "($close-Min($low, 60))/(Max($high, 60)-Min($low, 60)+1e-12)",
    "IMAX5": "IdxMax($high, 5)/5",
    "IMAX10": "IdxMax($high, 10)/10",
    "IMAX20": "IdxMax($high, 20)/20",
    "IMAX30": "IdxMax($high, 30)/30",
    "IMAX60": "IdxMax($high, 60)/60",
    "IMIN5": "IdxMin($low, 5)/5",
    "IMIN10": "IdxMin($low, 10)/10",
    "IMIN20": "IdxMin($low, 20)/20",
    "IMIN30": "IdxMin($low, 30)/30",
    "IMIN60": "IdxMin($low, 60)/60",
    "IMXD5": "(IdxMax($high, 5)-IdxMin($low, 5))/5",
    "IMXD10": "(IdxMax($high, 10)-IdxMin($low, 10))/10",
    "IMXD20": "(IdxMax($high, 20)-IdxMin($low, 20))/20",
    "IMXD30": "(IdxMax($high, 30)-IdxMin($low, 30))/30",
    "IMXD60": "(IdxMax($high, 60)-IdxMin($low, 60))/60",
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "CORR30": "Corr($close, Log($volume+1), 30)",
    "CORR60": "Corr($close, Log($volume+1), 60)",
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORD20": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 20)",
    "CORD30": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 30)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "CNTP5": "Mean($close>Ref($close, 1), 5)",
    "CNTP10": "Mean($close>Ref($close, 1), 10)",
    "CNTP20": "Mean($close>Ref($close, 1), 20)",
    "CNTP30": "Mean($close>Ref($close, 1), 30)",
    "CNTP60": "Mean($close>Ref($close, 1), 60)",
    "CNTN5": "Mean($close<Ref($close, 1), 5)",
    "CNTN10": "Mean($close<Ref($close, 1), 10)",
    "CNTN20": "Mean($close<Ref($close, 1), 20)",
    "CNTN30": "Mean($close<Ref($close, 1), 30)",
    "CNTN60": "Mean($close<Ref($close, 1), 60)",
    "CNTD5": "Mean($close>Ref($close, 1), 5)-Mean($close<Ref($close, 1), 5)",
    "CNTD10": "Mean($close>Ref($close, 1), 10)-Mean($close<Ref($close, 1), 10)",
    "CNTD20": "Mean($close>Ref($close, 1), 20)-Mean($close<Ref($close, 1), 20)",
    "CNTD30": "Mean($close>Ref($close, 1), 30)-Mean($close<Ref($close, 1), 30)",
    "CNTD60": "Mean($close>Ref($close, 1), 60)-Mean($close<Ref($close, 1), 60)",
    "SUMP5": "Sum(Greater($close-Ref($close, 1), 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
    "SUMP10": "Sum(Greater($close-Ref($close, 1), 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)",
    "SUMP20": "Sum(Greater($close-Ref($close, 1), 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
    "SUMP30": "Sum(Greater($close-Ref($close, 1), 0), 30)/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)",
    "SUMP60": "Sum(Greater($close-Ref($close, 1), 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)",
    "SUMN5": "Sum(Greater(Ref($close, 1)-$close, 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
    "SUMN10": "Sum(Greater(Ref($close, 1)-$close, 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)",
    "SUMN20": "Sum(Greater(Ref($close, 1)-$close, 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
    "SUMN30": "Sum(Greater(Ref($close, 1)-$close, 0), 30)/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)",
    "SUMN60": "Sum(Greater(Ref($close, 1)-$close, 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)",
    "SUMD5": "(Sum(Greater($close-Ref($close, 1), 0), 5)-Sum(Greater(Ref($close, 1)-$close, 0), 5))/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)",
    "SUMD10": "(Sum(Greater($close-Ref($close, 1), 0), 10)-Sum(Greater(Ref($close, 1)-$close, 0), 10))/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)",
    "SUMD20": "(Sum(Greater($close-Ref($close, 1), 0), 20)-Sum(Greater(Ref($close, 1)-$close, 0), 20))/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
    "SUMD30": "(Sum(Greater($close-Ref($close, 1), 0), 30)-Sum(Greater(Ref($close, 1)-$close, 0), 30))/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)",
    "SUMD60": "(Sum(Greater($close-Ref($close, 1), 0), 60)-Sum(Greater(Ref($close, 1)-$close, 0), 60))/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)",
    "VMA5": "Mean($volume, 5)/($volume+1e-12)",
    "VMA10": "Mean($volume, 10)/($volume+1e-12)",
    "VMA20": "Mean($volume, 20)/($volume+1e-12)",
    "VMA30": "Mean($volume, 30)/($volume+1e-12)",
    "VMA60": "Mean($volume, 60)/($volume+1e-12)",
    "VSTD5": "Std($volume, 5)/($volume+1e-12)",
    "VSTD10": "Std($volume, 10)/($volume+1e-12)",
    "VSTD20": "Std($volume, 20)/($volume+1e-12)",
    "VSTD30": "Std($volume, 30)/($volume+1e-12)",
    "VSTD60": "Std($volume, 60)/($volume+1e-12)",
    "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "WVMA10": "Std(Abs($close/Ref($close, 1)-1)*$volume, 10)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 10)+1e-12)",
    "WVMA20": "Std(Abs($close/Ref($close, 1)-1)*$volume, 20)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 20)+1e-12)",
    "WVMA30": "Std(Abs($close/Ref($close, 1)-1)*$volume, 30)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 30)+1e-12)",
    "WVMA60": "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
    "VSUMP5": "Sum(Greater($volume-Ref($volume, 1), 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
    "VSUMP10": "Sum(Greater($volume-Ref($volume, 1), 0), 10)/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
    "VSUMP20": "Sum(Greater($volume-Ref($volume, 1), 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
    "VSUMP30": "Sum(Greater($volume-Ref($volume, 1), 0), 30)/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)",
    "VSUMP60": "Sum(Greater($volume-Ref($volume, 1), 0), 60)/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)",
    "VSUMN5": "Sum(Greater(Ref($volume, 1)-$volume, 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
    "VSUMN10": "Sum(Greater(Ref($volume, 1)-$volume, 0), 10)/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
    "VSUMN20": "Sum(Greater(Ref($volume, 1)-$volume, 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
    "VSUMN30": "Sum(Greater(Ref($volume, 1)-$volume, 0), 30)/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)",
    "VSUMN60": "Sum(Greater(Ref($volume, 1)-$volume, 0), 60)/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)",
    "VSUMD5": "(Sum(Greater($volume-Ref($volume, 1), 0), 5)-Sum(Greater(Ref($volume, 1)-$volume, 0), 5))/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)",
    "VSUMD10": "(Sum(Greater($volume-Ref($volume, 1), 0), 10)-Sum(Greater(Ref($volume, 1)-$volume, 0), 10))/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
    "VSUMD20": "(Sum(Greater($volume-Ref($volume, 1), 0), 20)-Sum(Greater(Ref($volume, 1)-$volume, 0), 20))/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
    "VSUMD30": "(Sum(Greater($volume-Ref($volume, 1), 0), 30)-Sum(Greater(Ref($volume, 1)-$volume, 0), 30))/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)",
    "VSUMD60": "(Sum(Greater($volume-Ref($volume, 1), 0), 60)-Sum(Greater(Ref($volume, 1)-$volume, 0), 60))/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)",
}


def crypto_mode_enabled() -> bool:
    """Whether the crypto scenario is active (``CRYPTO_MODE=true``).

    Read straight from the environment so this module stays dependency-free and
    cannot introduce an import cycle with the scenario packages.
    """

    return os.environ.get("CRYPTO_MODE", "").strip().lower() == "true"


def get_base_factors() -> dict[str, str]:
    """Base feature library for the current market.

    Crypto -> our own OHLCV-only crypto library. Otherwise the original
    A-share ``ALPHA20``, unchanged.
    """

    if crypto_mode_enabled():
        from rdagent.utils.crypto_factors import CRYPTO_BASE

        return CRYPTO_BASE
    return ALPHA20


def get_all_factors() -> dict[str, str]:
    """Full feature library for the current market."""

    if crypto_mode_enabled():
        from rdagent.utils.crypto_factors import CRYPTO_ALL

        return CRYPTO_ALL
    return ALPHA158


_TFW = FBWorkspace()  # test feature workspace
TEST_FEATURE_CODE = """
import qlib  
from qlib.data import D  

qlib.init()  
expressions = {experessions}
df = D.features(["SH600000"], expressions, start_time="2008-01-01", end_time="2020-08-31")
"""

TEST_FEATURE_CODE_CRYPTO = """
import qlib
from qlib.data import D

qlib.init(provider_uri={provider_uri}, region="cn")
expressions = {experessions}
df = D.features([{instrument}], expressions, start_time={start}, end_time={end})
assert not df.empty, "feature validation returned no rows"
"""


def _crypto_validation_target() -> tuple[str, str, str, str] | None:
    """Provider/instrument/date-range the crypto validation should exercise.

    The A-share probe replays ``SH600000`` on the default ``cn_data`` provider,
    which does not exist in the crypto deployment; validating crypto base
    features against it always failed, silently dropping user
    ``base_factors.json`` entries. Resolve the provider the same way the run
    does (``CRYPTO_BIN_ROOT`` / ``CRYPTO_DATASET_NAME`` overrides included) and
    pick a real instrument from that provider.
    """

    from rdagent.scenarios.qlib.crypto.conf import CRYPTO_SETTINGS

    provider = Path(CRYPTO_SETTINGS.provider_uri)
    instruments_file = provider / "instruments" / "all.txt"
    try:
        codes = [
            line.split("\t")[0].strip()
            for line in instruments_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return None
    if not codes:
        return None
    instrument = "BTCUSDT" if "BTCUSDT" in codes else codes[0]
    start = "2022-01-01"
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        meta = json.loads((provider / "crypto_meta.json").read_text(encoding="utf-8"))
        start = str(meta.get("start") or start)
        end = str(meta.get("end") or end)
    except (OSError, ValueError):
        pass
    return str(provider), instrument, start, end


def validate_qlib_features(expressions: list[str]) -> bool:
    if crypto_mode_enabled():
        target = _crypto_validation_target()
        if target is None:
            return False
        provider_uri, instrument, start, end = target
        code = TEST_FEATURE_CODE_CRYPTO.format(
            experessions=str(expressions),
            provider_uri=repr(provider_uri),
            instrument=repr(instrument),
            start=repr(start),
            end=repr(end),
        )
    else:
        code = TEST_FEATURE_CODE.format(experessions=str(expressions))
    _TFW.inject_files(**{"test_fea.py": code})

    qlib_env = QlibCondaEnv(conf=QlibCondaConf())
    qlib_env.prepare()
    res = _TFW.run(
        env=qlib_env,
        entry="python test_fea.py",
    )
    return res.exit_code == 0
