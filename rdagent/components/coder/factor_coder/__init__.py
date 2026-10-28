from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.scenarios.qlib.crypto.conf import CRYPTO_SETTINGS, crypto_enabled
from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
from rdagent.components.coder.factor_coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario


class FactorCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        # The knowledge base needs an embedding endpoint. When the configured
        # gateway does not offer one (e.g. a chat-only OpenAI-compatible proxy),
        # crypto mode can run without it: with_knowledge=False skips the RAG
        # retrieval entirely instead of failing on a missing embedding model.
        if crypto_enabled() and not CRYPTO_SETTINGS.enable_knowledge:
            kwargs.setdefault("with_knowledge", False)

        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)

    def develop(self, exp: Experiment) -> Experiment:
        try:
            exp = super().develop(exp)
        finally:
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                es = self.evolve_agent.evolving_trace[-1]
                exp.prop_dev_feedback = es.feedback
        return exp
