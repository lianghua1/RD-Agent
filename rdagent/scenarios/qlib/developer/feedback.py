import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

from rdagent.scenarios.qlib.experiment.backtest_config import position_mode

BASE_METRICS = [
    "IC",
    "Rank IC",
    "Long-Short Ann Return",
    "Long-Short Ann Sharpe",
]
MODE_METRICS = {
    "long_only": [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.max_drawdown",
    ],
    "short_only": [
        "short_only.annualized_return",
        "short_only.sharpe",
        "short_leg.annualized_return",
        "short_leg.sharpe",
        "short_carry_cost_sum",
        "funding_pnl_sum",
        "turnover_cost_mean",
    ],
    "long_short": [
        "long_short.annualized_return",
        "long_short.sharpe",
        "long_short.max_drawdown",
        "long_leg.annualized_return",
        "short_leg.annualized_return",
        "short_carry_cost_sum",
        "funding_pnl_sum",
        "turnover_cost_mean",
    ],
}


def important_metrics(mode: str | None = None) -> list[str]:
    mode = mode or position_mode()
    report_freq = os.environ.get("CRYPTO_REPORT_FREQ", "1day").strip() or "1day"
    try:
        n_instruments = int(os.environ.get("FACTOR_N_INSTRUMENTS", "0") or "0")
    except ValueError:
        n_instruments = 0
    # Cross-sectional IC is undefined with one instrument; showing NaN IC/Rank
    # IC to the judge makes a valid time-series result look like a failure.
    base_metrics = [] if n_instruments == 1 else BASE_METRICS
    long_only_metrics = [
        f"{report_freq}.excess_return_with_cost.annualized_return",
        f"{report_freq}.excess_return_with_cost.information_ratio",
        f"{report_freq}.excess_return_with_cost.max_drawdown",
        # Single-instrument time-series simulator (used by BTC-only runs).
        "strategy.annualized_return",
        "strategy.sharpe",
        "strategy.max_drawdown",
        "long_only.annualized_return",
        "long_only.sharpe",
        "long_only.max_drawdown",
        "long_leg.annualized_return",
        "long_leg.sharpe",
        "turnover_cost_mean",
    ]
    if mode == "long_only":
        return base_metrics + long_only_metrics
    return base_metrics + MODE_METRICS.get(mode, MODE_METRICS["long_only"])


def process_results(current_result, sota_result, mode: str | None = None):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Filter the combined DataFrame to retain only the important metrics.
    # A metric can be missing (e.g. a backtest that only reports IC). Degrade the
    # report instead of raising a KeyError in the middle of the loop.
    available_metrics = [metric for metric in important_metrics(mode) if metric in combined_df.index]
    if not available_metrics:
        return "No important metrics available in backtest results."
    filtered_combined_df = combined_df.loc[available_metrics]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            try:
                results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
            except (TypeError, ValueError):
                results.append(f"{metric} of Current Result is {current}, of SOTA Result is {sota}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


def select_important_metrics(result, mode: str | None = None):
    """Subset a Qlib result Series to the important metrics that actually exist.

    Falls back to the full series when none of the expected keys are present, so
    the prompt still receives evidence rather than an empty value. This keeps
    both the A-share and crypto paths free of KeyError crashes.
    """

    if result is None:
        return None
    available = [metric for metric in important_metrics(mode) if metric in result.index]
    return result.loc[available] if available else result


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        active_mode = position_mode()
        combined_result = process_results(current_result, sota_result, active_mode)

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor"), position_mode=active_mode
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(), position_mode=active_mode
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
            panel_facts=os.environ.get("FACTOR_PANEL_FACTS", "").strip() or "not provided",
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(), position_mode=position_mode()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=select_important_metrics(SOTA_experiment.result) if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=select_important_metrics(exp.result) if exp.result is not None else "execution failed",
        )

        # Single LLM call. The upstream code called the model twice and discarded
        # the first response, doubling cost for no benefit.
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
