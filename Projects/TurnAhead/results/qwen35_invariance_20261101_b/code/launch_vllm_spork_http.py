#!/usr/bin/env python3
"""SPORK-HTTP launcher for installed vLLM 0.29 (Qwen3 completion API).

Use a worker extension instead of fork-dependent class patches. The underlying
ngram proposer and vLLM rejection sampler remain in place. Single evaluator,
TP supported, V1 runner and synchronous scheduling required for D3 accounting.
"""
import os
import sys
import hashlib
from pathlib import Path
from importlib.metadata import version

ROOT = str(Path(__file__).resolve().parent)
STARTUP_SOURCE_SHA256 = {
    name: hashlib.sha256((Path(ROOT) / name).read_bytes()).hexdigest()
    for name in ("launch_vllm_spork_http.py", "spork_core/vllm029_d3.py")
}
def startup_option(name):
    for i, value in enumerate(sys.argv[1:], 1):
        if value == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None

STARTUP_SERVING = {
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "tensor_parallel_size": startup_option("--tensor-parallel-size"),
    "max_model_len": startup_option("--max-model-len"),
    "max_num_seqs": startup_option("--max-num-seqs"),
    "max_num_batched_tokens": startup_option("--max-num-batched-tokens"),
}
sys.path.insert(0, ROOT)
os.environ['PYTHONPATH'] = ROOT + os.pathsep + os.environ.get('PYTHONPATH','')
os.environ.setdefault('VLLM_USE_V2_MODEL_RUNNER','0')
if not version('vllm').startswith('0.29.'):
    raise RuntimeError('This compatibility launcher targets vLLM 0.29.x only')

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.launchers.api_server import entry


def select_worker(results):
    # The TP ranks that do not own the drafter are inactive; do not sum replicas.
    for r in results:
        if r.get('status')=='ok' or r.get('patched'):
            return {**r,'workers':results}
    for r in results:
        if r.get('has_drafter') or r.get('status')=='error':
            return {**r,'workers':results}
    return {'status':'error','error':'no_drafter','workers':results}


async def set_tokens(request: Request):
    body = await request.json()
    rid,tokens,plen = body.get('request_id'),body.get('draft_tokens'),body.get('prompt_len')
    if not isinstance(rid,str) or not isinstance(tokens,list) or not tokens or not all(type(t) is int and t>=0 for t in tokens) or type(plen) is not int or plen<0:
        return JSONResponse({'status':'error','error':'invalid_payload'},status_code=400)
    results = await request.app.state.engine_client.collective_rpc('spork_set_draft_tokens',args=(rid,tokens,plen))
    return JSONResponse(select_worker(results))


async def status(request: Request):
    results = await request.app.state.engine_client.collective_rpc('spork_get_status')
    result = select_worker(results)
    # Expose only the serving setting needed to prevent collecting against an
    # old/default server. This is not a claim that output invariance was tested.
    result['server_config'] = {
        'vllm_version': version('vllm'),
        'batch_invariant_requested': os.environ.get('VLLM_BATCH_INVARIANT', '0') == '1',
        'prefix_cache_reset_available': True,
        'project_root': ROOT,
        'startup_source_sha256': STARTUP_SOURCE_SHA256,
        'tool_call_open_token_id': int(os.environ.get('SPORK_TOOL_CALL_OPEN_TOKEN_ID', '151657')),
        'experiment_profile': os.environ.get('SPORK_EXPERIMENT_PROFILE', 'qwen3_32b'),
        'startup_serving': STARTUP_SERVING,
        'gdn_execution': {
            'decode_kernel': os.environ.get('VLLM_GDN_DECODE_KERNEL', 'cuda'),
            'packed_recurrent_decode': os.environ.get('VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE', '1'),
        },
    }
    return JSONResponse(result)


async def clear(request: Request):
    results = await request.app.state.engine_client.collective_rpc('spork_clear')
    return JSONResponse(select_worker(results))


async def reset_prefix_cache(request: Request):
    # vLLM 0.29 returns False while live requests still hold cache blocks.
    # Never abort running requests or reset external connector state here.
    success = await request.app.state.engine_client.reset_prefix_cache(False, False)
    return JSONResponse({'success': bool(success)})


original_build_app = entry.build_app

def build_app(*args,**kwargs):
    app = original_build_app(*args,**kwargs)
    app.add_api_route('/spork/set_tokens',set_tokens,methods=['POST'])
    app.add_api_route('/spork/status',status,methods=['GET'])
    app.add_api_route('/spork/clear',clear,methods=['POST'])
    app.add_api_route('/spork/reset_prefix_cache',reset_prefix_cache,methods=['POST'])
    return app

entry.build_app = build_app

if __name__=='__main__':
    if not any(a.startswith('--worker-extension-cls') for a in sys.argv):
        sys.argv += ['--worker-extension-cls','spork_core.vllm029_d3.SporkWorkerExtension']
    if '--no-async-scheduling' not in sys.argv:
        sys.argv += ['--no-async-scheduling']
    entry.main()
