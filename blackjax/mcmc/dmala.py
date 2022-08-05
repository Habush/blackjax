# Author: Abdulrahman S. Omar<hsamireh@gmal.com>
# Implement the Discrete Metropolis-Adjusted Langevin Algorithm (DMALA) - https://arxiv.org/abs/2206.09914

from typing import Callable, Tuple, ClassVar
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import distrax
from blackjax.types import PRNGKey, PyTree
from blackjax.mcmc.mala import MALAState, MALAInfo
from blackjax.mcmc.diffusion import DiffusionState, overdamped_langevin
import numpy as np
from jax.experimental.host_callback import id_print

def jax_prng_key():
    return jax.random.PRNGKey(np.random.randint(int(1e5)))

def init(position: PyTree, logprob_fn: Callable) -> MALAState:
    grad_fn = jax.value_and_grad(logprob_fn)
    logprob, logprob_grad = grad_fn(position)
    return MALAState(position, logprob, logprob_grad)

def print_tap(msg):
    def print_msg(arg):
        print(msg + arg)

    return print_msg

def kernel():
    """
    Build a DMALA kernel for binary variables - based on https://arxiv.org/abs/2206.09914 - Algorithm 2

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """
    EPS = 1e-10

    def diff_fn(state, step_size):

        theta = jax.tree_util.tree_map(lambda x, g: -0.5*(g)*(2.*x - 1) - (1./(2.*step_size)),
                                       state.position, state.logprob_grad)

        return jax.nn.sigmoid(theta)

    def one_step(
        rng_key: PRNGKey, state: MALAState, logprob_fn: Callable, step_size: float
    ) -> Tuple[MALAState, MALAInfo]:

        _, key_rmh, key_accept = jax.random.split(rng_key, 3)
        grad_fn = jax.grad(logprob_fn)
        theta_cur = state.position

        u = jax.random.uniform(key_rmh, shape=state.position.shape)
        p_curr = diff_fn(state, step_size)
        ind = jnp.array(u < p_curr)
        pos_new = (1. - theta_cur)*ind + theta_cur*(1. - ind)
        probs = p_curr*ind + (1. - p_curr) * (1. - ind)
        lp_forward = jnp.sum(jnp.log(probs+EPS), axis=-1)

        logprob_new = logprob_fn(pos_new)
        logprob_grad_new = grad_fn(pos_new)
        new_state = MALAState(pos_new, logprob_new, logprob_grad_new)
        p_new = diff_fn(new_state, step_size)
        probs_new = p_new*ind + (1. - p_new)*(1. - ind)
        lp_reverse = jnp.sum(jnp.log(probs_new + EPS), axis=-1)
        delta = (new_state.logprob
                 - state.logprob
                 + lp_reverse
                 - lp_forward)

        u2 = jax.random.uniform(key_accept, shape=theta_cur.shape)
        a = u2 < jnp.exp(delta)
        theta_new = pos_new * a + theta_cur * (1. - a)
        logprob_theta_new = logprob_fn(theta_new)
        grad_theta_new = grad_fn(theta_new)
        new_state = MALAState(pos_new, logprob_theta_new, grad_theta_new)
        info = MALAInfo(jnp.mean(a), jnp.mean(a) > 0)

        return (new_state, info)


    return one_step