# Author: Abdulrahman S. Omar <hsamireh@gmail.com>

import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Tuple
from blackjax.types import PRNGKey, PyTree
from blackjax.mcmc.mala import kernel as ckernel
from blackjax.mcmc.dmala import kernel as dkernel
from blackjax.mcmc.mala import MALAState, MALAInfo
from numpyro.infer import MixedHMC
from blackjax.mcmc.diffusion import generate_gaussian_noise
from jax.flatten_util import ravel_pytree

__all__ = ["MixedMALAState", "MixedMALAInfo", "init", "kernel"]

class MixedMALAPosition(NamedTuple):
    discrete_position: PyTree
    contin_position: PyTree

class MixedMALAState(NamedTuple):
    """Holds info about the discrete and the continuous r.vs in the mixed support"""

    position: MixedMALAPosition

    logprob: float

    discrete_logprob_grad: PyTree
    contin_logprob_grad: PyTree

class MixedMALAInfo(NamedTuple):
    """Additional information on the MALA transition.

        This additional information can be used for debugging or computing
        diagnostics.

        acceptance_probability
            The acceptance probability of the transition.
        is_accepted
            Whether the proposed position was accepted or the original position
            was returned.

    """

    discrete_acc_prob: float
    discrete_is_accepted: bool

    contin_acc_prob: float
    contin_is_accepted: bool

# We assume the log probability function takes discrete variable as its 1st arg and the contin as its 2nd arg
def init(disc_position: PyTree, contin_position: PyTree, logprob_fn: Callable,
         disc_grad_fn: Callable, contin_grad_fn: Callable) -> MixedMALAState:

    logprob = logprob_fn(disc_position, contin_position)
    disc_logprob_grad = disc_grad_fn(disc_position, contin_position)

    contin_logprob_grad = contin_grad_fn(disc_position, contin_position)

    return MixedMALAState(MixedMALAPosition(disc_position, contin_position), logprob,
                                                    disc_logprob_grad, contin_logprob_grad)



def kernel():


    def contin_transition_probability(state, new_state, step_size):
        """Transition probability to go from `state` to `new_state`"""
        theta = jax.tree_util.tree_map(
            lambda new_x, x, g: new_x - x - step_size * g,
            new_state.position,
            state.position,
            state.logprob_grad,
        )
        theta_ravel, _ = ravel_pytree(theta)
        return -0.25 * (1.0 / step_size) * jnp.dot(theta_ravel, theta_ravel)


    EPS = 1e-10

    def diff_fn(state, step_size):

        theta = jax.tree_util.tree_map(lambda x, g: 0.5*(g)*(2.*x - 1) - 0.5*step_size,
                                       state.position, state.logprob_grad)

        # theta_ravel, _ = ravel_pytree(theta)
        theta = jnp.where(jnp.isnan(theta), -jnp.inf, theta)
        return jnp.exp(theta) / (jnp.exp(theta) + 1)


    def take_discrete_step(rng_key: PRNGKey, disc_state: MALAState, contin_state: MALAState,
                           logprob_fn: Callable, disc_grad_fn: Callable,
                           step_size: float) -> MALAState:

        _, key_rmh = jax.random.split(rng_key)
        u = jax.random.uniform(key_rmh, shape=disc_state.position.shape)
        p_cur = diff_fn(disc_state, step_size)
        ind = jnp.array(u < p_cur) * 1
        probs_cur = p_cur*ind + (1. - p_cur) * (1. - ind)
        lp_forward = jnp.sum(jnp.log(probs_cur + EPS), axis=-1)

        pos_new = (1. - disc_state.position)*ind + disc_state.position*(1. - ind)
        logprob_new = logprob_fn(disc_state.position, contin_state.position)
        logprob_grad_new = disc_grad_fn(disc_state.position, contin_state.position)
        new_state = MALAState(pos_new, logprob_new, logprob_grad_new)
        p_new = diff_fn(new_state, step_size)
        probs_new = p_new*ind + (1. - p_new)*(1. - ind)
        lp_reverse = jnp.sum(jnp.log(probs_new+EPS), axis=-1)

        delta = (new_state.logprob
                - disc_state.logprob
                + lp_reverse
                - lp_forward)

        delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        p_accept = jnp.clip(jnp.exp(delta), a_max=1)

        do_accept = jax.random.bernoulli(key_rmh, p_accept)

        return jax.lax.cond(
            do_accept,
            lambda _: new_state,
            lambda _: disc_state,
            operand=None,
        )


    def take_contin_step(rng_key: PRNGKey, disc_state: MALAState, contin_state: MALAState,
                           logprob_fn: Callable, contin_grad_fn: Callable,
                           step_size: float) -> MALAState:

        _, key_rmh = jax.random.split(rng_key)
        noise = generate_gaussian_noise(rng_key, contin_state.position)
        new_position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            contin_state.position,
            contin_state.logprob_grad,
            noise,
        )

        logprob_new = logprob_fn(disc_state.position, new_position)
        logprob_grad_new = contin_grad_fn(disc_state.position, new_position)
        new_state = MALAState(new_position, logprob_new, logprob_grad_new)

        delta = (new_state.logprob
                - contin_state.logprob
                + contin_transition_probability(new_state, contin_state, step_size)
                - contin_transition_probability(contin_state, new_state, step_size)
        )
        delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        p_accept = jnp.clip(jnp.exp(delta), a_max=1)

        do_accept = jax.random.bernoulli(key_rmh, p_accept)

        return jax.lax.cond(
            do_accept,
            lambda _: new_state,
            lambda _: contin_state,
            operand=None,
        )

    def one_step(
            rng_key: PRNGKey, state: MixedMALAState, logprob_fn: Callable,
            disc_grad_fn: Callable, contin_grad_fn: Callable,
            discrete_step_size: float, contin_step_size: float, L: int
    ) -> MixedMALAState:


        # Evolve each variable in tandem and combine the results
        disc_state = MALAState(state.position.discrete_position, state.logprob, state.discrete_logprob_grad)
        contin_state = MALAState(state.position.contin_position, state.logprob, state.contin_logprob_grad)

        # Take steps for the discrete variable
        new_disc_state = jax.lax.fori_loop(0, L,
            lambda ii, state: take_discrete_step(
                rng_key, disc_state, contin_state, logprob_fn, disc_grad_fn, discrete_step_size),
                disc_state)
        # Take steps for the contin variable
        new_contin_state = jax.lax.fori_loop(0, L,
             lambda ii, state: take_contin_step(
                 rng_key, disc_state, contin_state, logprob_fn, contin_grad_fn, contin_step_size),
             contin_state)

        new_state = MixedMALAState(MixedMALAPosition(new_disc_state.position, new_contin_state.position),
                                   new_contin_state.logprob,
                                   new_disc_state.logprob_grad, new_contin_state.logprob_grad)


        return new_state


    return one_step