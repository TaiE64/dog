# Register custom actor-critic classes so that rsl_rl's OnPolicyRunner
# can find them via eval(class_name).
# The runner imports from rsl_rl.modules, so we inject our class there.
import rsl_rl.runners.on_policy_runner as _runner_module

from .alam_actor_critic import ALaMActorCritic

_runner_module.ALaMActorCritic = ALaMActorCritic
