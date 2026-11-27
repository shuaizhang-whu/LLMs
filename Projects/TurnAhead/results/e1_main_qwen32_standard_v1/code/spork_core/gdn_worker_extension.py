"""D3 RPCs plus the opt-in reference GDN worker hook."""
from .gdn_invariant_worker import register_worker_hook, runtime_status
from .vllm029_d3 import SporkWorkerExtension

register_worker_hook()


class ReferenceGdnWorkerExtension(SporkWorkerExtension):
    def spork_gdn_status(self):
        return runtime_status()
