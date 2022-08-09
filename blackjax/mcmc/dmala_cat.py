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

def jax_prng_key():
    return jax.random.PRNGKey(np.random.randint(int(1e5)))

def init(position: PyTree, logprob_fn: Callable) -> MALAState:
    grad_fn = jax.value_and_grad(logprob_fn)
    logprob, logprob_grad = grad_fn(position)
    return MALAState(position, logprob, logprob_grad)

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

    def diff_fn_multi_dim(theta, theta_delta, step_size):
        return jax.tree_util.tree_map(lambda x_cur, x_delta, g_cur: 0.5*g_cur*(x_delta - x_cur)
                                                             - 0.5*((x_delta - x_cur)**2)*step_size,
                               theta.position, theta_delta.position, theta.logprob_grad)


    def one_step(
            rng_key: PRNGKey, state: MALAState, logprob_fn: Callable, step_size: float
    ) -> Tuple[MALAState, MALAInfo]:
        constant = 1.
        grad_fn = jax.value_and_grad(logprob_fn)
        integrator = overdamped_langevin(grad_fn)
        key_integrator, key_rmh = jax.random.split(rng_key)
        u = jax.random.uniform(key_rmh)

        theta_cur = state.position
        val_theta, grad_theta = grad_fn(theta_cur)

        theta_delta = integrator(key_integrator, DiffusionState(*state), step_size)

        forward_delta = diff_fn_multi_dim(state, theta_delta, step_size)
        # make sure we dont choose to stay where we are!
        forward_logits = forward_delta - constant * theta_cur
        cd_forward = distrax.OneHotCategorical(logits=forward_logits.reshape(theta_cur.shape[0], -1))
        theta_delta = cd_forward.sample(seed=jax_prng_key())
        # compute probability of sampling this change
        lp_forward = cd_forward.log_prob(theta_delta)

        val_theta_delta, grad_theta_delta = grad_fn(theta_delta)
        reverse_delta = diff_fn_multi_dim(DiffusionState(theta_delta, val_theta_delta, grad_theta_delta), state, step_size)
        reverse_logits = reverse_delta - constant * theta_delta
        cd_reverse = distrax.OneHotCategorical(logits=reverse_logits.reshape(theta_delta.shape[0], -1))
        # get dims that changed
        lp_reverse = cd_reverse.log_prob(theta_cur)

        transition_p = (val_theta_delta - val_theta) + (lp_reverse - lp_forward)
        transition_p = jnp.where(jnp.isnan(transition_p), -jnp.inf, transition_p)
        transition_p = jnp.clip(jnp.exp(transition_p), a_max=1)

        do_accept = u < transition_p

        new_state = MALAState(theta_delta, val_theta_delta, grad_theta_delta)
        info = MALAInfo(jnp.mean(transition_p), do_accept)

        return jax.lax.cond(
        do_accept,
        lambda _: (new_state, info),
        lambda _: (state, info),
        operand=None,
    )


    # def one_step(
    #     rng_key: PRNGKey, state: MALAState, logprob_fn: Callable, step_size: float
    # ) -> Tuple[MALAState, MALAInfo]:
    #     constant = 1.
    #     grad_fn = jax.value_and_grad(logprob_fn)
    #     integrator = overdamped_langevin(grad_fn)
    #     key_integrator, key_rmh = jax.random.split(rng_key)
    #     u = jax.random.uniform(key_rmh, shape=state.position.shape)
    #
    #     theta_cur = state.position
    #     val_theta, grad_theta = grad_fn(theta_cur)
    #
    #     theta_delta = integrator(key_integrator, DiffusionState(*state), step_size)
    #
    #     forward_delta = diff_fn_multi_dim(state, theta_delta, step_size)
    #     # make sure we dont choose to stay where we are!
    #     forward_logits = forward_delta - constant * theta_cur
    #     cd_forward = distrax.OneHotCategorical(logits=forward_logits.reshape(theta_cur.shape[0], -1))
    #     changes = cd_forward.sample(seed=jax_prng_key())
    #     # compute probability of sampling this change
    #     lp_forward = cd_forward.log_prob(changes)
    #     #reshape to (bs, dim, nout)
    #     changes_r = changes.reshape(theta_cur.shape)
    #     # get binary indicator (bs, dim) indicating which dim was changed
    #     changed_ind = jnp.sum(changes_r, axis=-1)
    #     # mask out unchanged dim and add in the change
    #     theta_delta = theta_cur * (1. - changed_ind[:,:,None]) + changes_r
    #
    #     val_theta_delta, grad_theta_delta = grad_fn(theta_delta)
    #     reverse_delta = diff_fn_multi_dim(DiffusionState(theta_delta, val_theta_delta, grad_theta_delta), state, step_size)
    #     reverse_logits = reverse_delta - constant * theta_delta
    #     cd_reverse = distrax.OneHotCategorical(logits=reverse_logits.reshape(theta_delta.shape[0], -1))
    #     # get dims that changed
    #     reverse_changes = theta_cur * changed_ind[:, :, None]
    #     lp_reverse = cd_reverse.log_prob(reverse_changes.reshape(theta_delta.shape[0], -1))
    #
    #     transition_p = (val_theta_delta - val_theta) + (lp_reverse - lp_forward)
    #     transition_p = jnp.where(jnp.isnan(transition_p), -jnp.inf, transition_p)
    #     transition_p = jnp.clip(jnp.exp(transition_p), a_max=1)
    #
    #     do_accept = u < transition_p
    #
    #     theta = theta_delta*do_accept[:,None,None] + theta_cur*(1. - do_accept[:,None,None])
    #     val_theta, grad_theta = grad_fn(theta)
    #     new_state = MALAState(theta, val_theta, grad_theta)
    #     info = MALAInfo(jnp.mean(transition_p), do_accept)
    #     return (new_state, info)

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