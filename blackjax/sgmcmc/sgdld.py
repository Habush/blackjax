"""Public API for the Stochastic gradient Discrete Langevin Dynamics kernel."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.types import PRNGKey, PyTree
from blackjax.sgmcmc.sgld import SGLDState

__all__ = ["kernel"]

def kernel(grad_estimator_fn: Callable) -> Callable:

    def diff_fn(state, step_size):

        theta = jax.tree_util.tree_map(lambda x, g: 0.5*(g)*(2.*x - 1) - 0.5*step_size,
                                       state.position, state.logprob_grad)

        # theta_ravel, _ = ravel_pytree(theta)
        theta = jnp.where(jnp.isnan(theta), -jnp.inf, theta)
        return jnp.exp(theta) / (jnp.exp(theta) + 1)


    def one_step(
            rng_key: PRNGKey, state: SGLDState, data_batch: PyTree, step_size: float
    ) -> SGLDState:
        _, key_rmh = jax.random.split(rng_key)

        step, *diffusion_state = state
        u = jax.random.uniform(key_rmh, shape=state.position.shape)
        p_state = diff_fn(diffusion_state, step_size)
        ind = jnp.array(u <= p_state)

        pos_new = (1. - diffusion_state.position)*ind + diffusion_state.position*(1. - ind)
        grad_state_new = grad_estimator_fn(pos_new, data_batch)

        return SGLDState(step+1, pos_new, grad_state_new)


    return one_step