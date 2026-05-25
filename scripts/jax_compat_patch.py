"""Restore jax.device_put_replicated for brax compat with JAX 0.10+.

brax 0.14.2 uses jax.device_put_replicated which was removed in JAX 0.10.
This shim adds a leading batch dim (size=1) so brax's _unpmap can squeeze(0).

Install: copy to site-packages/ and create jax_compat_patch.pth with content:
    import jax_compat_patch
"""
import jax

def _device_put_replicated(val, devices):
    import jax.numpy as jnp
    def _rep(x):
        x = jnp.asarray(x)
        return jnp.expand_dims(x, axis=0)
    return jax.tree.map(_rep, val)

if not hasattr(jax, "device_put_replicated"):
    jax.device_put_replicated = _device_put_replicated
