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

        theta = jax.tree_util.tree_map(lambda x, g: 0.5*(g)*(2.*x - 1) - 0.5*step_size,
                               state.position, state.logprob_grad)

        # theta_ravel, _ = ravel_pytree(theta)
        theta = jnp.where(jnp.isnan(theta), -jnp.inf, theta)
        return jnp.exp(theta) / (jnp.exp(theta) + 1)

    def transition_probability(state, new_state, p_state, p_state_new, ind):

        q_state = p_state*ind + (1 - p_state)*(1. - ind)
        q_state_new = p_state_new*ind + (1 - p_state_new)*(1 - ind)

        lq_state = jnp.sum(q_state, axis=-1)

        lq_state_new = jnp.sum(q_state_new, axis=-1)



        delta = jnp.exp(new_state.logprob - state.logprob) + (lq_state_new / lq_state)
        # delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        return delta

    # def proposal_distribution(pos, new_pos, grad, step_size):
    #
    #     probs =  0.5*grad[0]*(new_pos - pos) - 0.5*((new_pos - pos)**2)*step_size
    #     print(f"Probs: {probs}")
    #     print(f"Probs shape: {probs.shape}")
    #     proposal_dist = distrax.Categorical(probs=jax.nn.softmax(probs))
    #
    #     return proposal_dist

    def one_step(
        rng_key: PRNGKey, state: MALAState, logprob_fn: Callable, step_size: float
    ) -> Tuple[MALAState, MALAInfo]:

        _, key_rmh = jax.random.split(rng_key)
        grad_fn = jax.value_and_grad(logprob_fn)
        val_theta, grad_theta = grad_fn(state.position)
        # id_print(val_theta, a="Called from disc step")
        u = jax.random.uniform(key_rmh, shape=state.position.shape)
        theta = DiffusionState(state.position, val_theta, grad_theta)
        p_theta = diff_fn(theta, step_size)
        ind = jnp.array(u < p_theta) * 1
        pos_new = (1. - theta.position)*ind + theta.position*(1. - ind)

        # id_print(pos_new, a="pos_new")
        probs = p_theta*ind + (1. - p_theta) * (1. - ind)
        lp_forward = jnp.sum(jnp.log(probs+EPS), axis=-1)
        val_theta_new, grad_theta_new = grad_fn(pos_new)
        # print(f"val_theta_new: {val_theta_new}, grad_theta_new: {grad_theta_new}")
        theta_new = DiffusionState(pos_new, val_theta_new, grad_theta_new)
        p_theta_new = diff_fn(theta_new, step_size)
        # id_print(p_theta_new, a="p_theta_new")
        probs = p_theta_new*ind + (1. - p_theta_new)*(1. - ind)
        lp_reverse = jnp.sum(jnp.log(probs+EPS), axis=-1)
        # id_print((val_theta, val_theta_new), a="val_theta, val_theta_new")
        m_term = (theta_new.logprob - theta.logprob)
        lp_term = lp_reverse - lp_forward
        la = jnp.exp(m_term + lp_term)
        # id_print((m_term, lp_term, la), a="m_term, lp_term, la")
        _, key_rmh2 = jax.random.split(key_rmh)
        u2 = jax.random.uniform(key_rmh2, shape=la.shape)
        # id_print(u2, a="u2")
        a = (u2 < la) * 1
        # id_print(a, a="accept_p")
        pos_new = pos_new * a[:,None] + theta.position * (1. - a[:,None])
        val_theta_new, grad_theta_new = grad_fn(pos_new)
        new_state = MALAState(*DiffusionState(pos_new, val_theta_new, grad_theta_new))
        info = MALAInfo(jnp.mean(u2), a > 0)
        return (new_state, info)

    # def one_step(
    #         rng_key: PRNGKey, state: MALAState, logprob_fn: Callable, step_size: float
    # ) -> Tuple[MALAState, MALAInfo]:
    #
    #     grad_fn = jax.value_and_grad(logprob_fn)
    #     integrator = overdamped_langevin(grad_fn)
    #     key_integrator, key_rmh = jax.random.split(rng_key)
    #     theta = state.position
    #     _, grad_theta = grad_fn(theta)
    #     theta_new = integrator(key_integrator, state, step_size).position
    #     proposal_dist = proposal_distribution(theta, theta_new, grad_theta, step_size)
    #     theta_new = proposal_dist.sample(seed=key_rmh, sample_shape=theta.shape)*1.
    #     val_theta_new, grad_theta_new = grad_fn(theta_new)
    #     new_state = MALAState(theta_new, val_theta_new, grad_theta_new)
    #     new_proposal_dist = proposal_distribution(theta_new, theta, grad_theta_new, step_size)
    #
    #     delta = (new_state.logprob - state.logprob + new_proposal_dist.log_prob(theta)
    #                 - proposal_dist.log_prob(theta_new))
    #
    #     delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
    #     p_accept = jnp.clip(jnp.exp(delta), a_max=1)
    #
    #     u = jax.random.uniform(key_rmh)
    #     do_accept = u < p_accept
    #
    #     info = MALAInfo(p_accept, do_accept)
    #
    #     return jax.lax.cond(
    #         do_accept,
    #         lambda _: (new_state, info),
    #         lambda _: (state, info),
    #         operand=None,
    #     )
    return one_step