# Author: Abdulrahman S. Omar <hsamireh@gmail.com>

import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Tuple
from blackjax.types import PRNGKey, PyTree
from blackjax.mcmc.mala import kernel as ckernel
from blackjax.mcmc.dmala import kernel as dkernel
from blackjax.mcmc.mala import MALAState, MALAInfo
from numpyro.infer import MixedHMC

__all__ = ["MixedMALAState", "MixedMALAInfo", "init", "kernel"]

class MixedMALAPosition(NamedTuple):
    discrete_position: PyTree
    contin_position: PyTree

class MixedMALAState(NamedTuple):
    """Holds info about the discrete and the continuous r.vs in the mixed support"""

    position: MixedMALAPosition

    discrete_logprob: float
    contin_logprob: float

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


def init(position: PyTree, logprob_fn: Callable) -> MixedMALAState:
    disc_position, contin_position = jax.tree_util.tree_flatten(position)[0]
    logprob = lambda z, x : logprob_fn((z, x))
    disc_logprob, disc_logprob_grad = jax.value_and_grad(logprob, argnums=0)(disc_position,
                                                                                contin_position)
    contin_logprob, contin_logprob_grad = jax.value_and_grad(logprob, argnums=1)(disc_position,
                                                                                    contin_position)

    return MixedMALAState(MixedMALAPosition(disc_position, contin_position), disc_logprob, contin_logprob,
                                                    disc_logprob_grad, contin_logprob_grad)



def kernel():

    discrete_step_fn = dkernel()
    contin_step_fn = ckernel()

    def one_step(
            rng_key: PRNGKey, state: MixedMALAState, logprob_fn: Callable,
            discrete_step_size: float, contin_step_size: float, L: int
    ) -> Tuple[MixedMALAState, MixedMALAInfo]:
        logprob = lambda z, x : logprob_fn((z, x))
        # Evolve each variable in tandem and combine the results
        discrete_logprob_fn = lambda x : logprob(x, state.position.contin_position)
        disc_logprob, disc_logprob_grad = jax.value_and_grad(discrete_logprob_fn)(state.position.discrete_position)
        discrete_state = MALAState(state.position.discrete_position, disc_logprob, disc_logprob_grad)

        # Take steps for the discrete variable
        new_disc_state = jax.lax.fori_loop(0, L,
            lambda ii, state: discrete_step_fn(rng_key, discrete_state, discrete_logprob_fn, discrete_step_size)[0],
            discrete_state)
        # new_disc_state, disc_info = discrete_step_fn(rng_key, discrete_state, discrete_logprob_fn, discrete_step_size)

        contin_logprob_fn = lambda x : logprob(new_disc_state.position, x)
        contin_logprob, contin_logprob_grad = jax.value_and_grad(contin_logprob_fn)(state.position.contin_position)

        contin_state = MALAState(state.position.contin_position, contin_logprob, contin_logprob_grad)
        # Take steps for the contin variable
        new_contin_state = jax.lax.fori_loop(0, L,
             lambda ii, state: contin_step_fn(rng_key, contin_state, contin_logprob_fn, contin_step_size)[0],
             contin_state)
        # new_contin_state, contin_info = contin_step_fn(rng_key, contin_state, contin_logprob_fn, contin_step_size)

        # disc_logprob, disc_logprob_grad = jax.value_and_grad(logprob, argnums=0)(new_disc_state.position,
        #                                                                             new_contin_state.position)
        # contin_logprob, contin_logprob_grad = jax.value_and_grad(logprob, argnums=1)(new_disc_state.position,
        #                                                                                 new_contin_state.position)

        new_state = MixedMALAState(MixedMALAPosition(new_disc_state.position, new_contin_state.position),
                                   new_disc_state.logprob, new_contin_state.logprob,
                                   new_disc_state.logprob_grad, new_contin_state.logprob_grad)

        # new_info = MixedMALAInfo(disc_info.acceptance_probability, disc_info.is_accepted,
        #                          contin_info.acceptance_probability, contin_info.is_accepted)

        new_info = MixedMALAInfo(1., True,
                                1., True)

        return new_state, new_info


    return one_step