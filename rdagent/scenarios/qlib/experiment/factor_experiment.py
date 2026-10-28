from copy import deepcopy
from pathlib import Path

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.crypto.conf import (
    CRYPTO_DERIVATIVES_NOTE,
    CRYPTO_MARKET_NOTE,
    CRYPTO_SETTINGS,
    CRYPTO_SINGLE_ASSET_NOTE,
    crypto_enabled,
)
from rdagent.scenarios.qlib.crypto.runtime import crypto_prompt_args
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class QlibFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")
        self.stdout = ""
        self.base_features: dict[str, str] = (
            {}
        )  # Qlib features in operator form, e.g., "RESI5": "Resi($close, 5)/$close"
        self.base_feature_codes: dict[str, str] = {}  # Qlib features in code form


def _provider_instrument_count() -> int:
    path = Path(CRYPTO_SETTINGS.provider_uri) / "instruments" / "all.txt"
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


class QlibFactorScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self._background = deepcopy(
            T(".prompts:qlib_factor_background").r(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        if crypto_enabled():
            self._background += "\n\n" + CRYPTO_MARKET_NOTE
            if CRYPTO_SETTINGS.include_derivatives:
                self._background += "\n\n" + CRYPTO_DERIVATIVES_NOTE
            if _provider_instrument_count() == 1:
                self._background += "\n\n" + CRYPTO_SINGLE_ASSET_NOTE
        self._source_data = deepcopy(get_data_folder_intro())
        # The upstream output-format example uses A-share tickers (SH600000 /
        # SZ300059). In crypto mode that leaks the wrong market into the
        # hypothesis prompt and the model writes A-share theses for a crypto
        # dataset, so swap in a crypto example instead.
        self._output_format = deepcopy(
            T(
                ".prompts:qlib_factor_output_format_crypto"
                if crypto_enabled()
                else ".prompts:qlib_factor_output_format"
            ).r()
        )
        self._interface = deepcopy(T(".prompts:qlib_factor_interface").r())
        self._strategy = deepcopy(T(".prompts:qlib_factor_strategy").r())
        self._simulator = deepcopy(T(".prompts:qlib_factor_simulator").r())
        self._rich_style_description = deepcopy(T(".prompts:qlib_factor_rich_style_description").r())
        setting_args = (
            crypto_prompt_args()
            if crypto_enabled()
            else {
                "train_start": FACTOR_PROP_SETTING.train_start,
                "train_end": FACTOR_PROP_SETTING.train_end,
                "valid_start": FACTOR_PROP_SETTING.valid_start,
                "valid_end": FACTOR_PROP_SETTING.valid_end,
                "test_start": FACTOR_PROP_SETTING.test_start,
                "test_end": FACTOR_PROP_SETTING.test_end,
            }
        )
        self._experiment_setting = deepcopy(T(".prompts:qlib_factor_experiment_setting").r(**setting_args))

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        """A static scenario describer"""
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.get_source_data_desc(task)}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your factor:
{self.simulator}
"""

    def get_runtime_environment(self):
        factor_env = get_factor_env()
        stdout = get_runtime_environment_by_env(env=factor_env)
        return stdout
