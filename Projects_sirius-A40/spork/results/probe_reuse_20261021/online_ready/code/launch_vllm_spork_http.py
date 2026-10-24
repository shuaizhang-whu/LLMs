#!/usr/bin/env python3
"""SPORK-HTTP launcher for installed vLLM 0.29 (Qwen3 completion API).

Use a worker extension instead of fork-dependent class patches. The underlying
ngram proposer and vLLM rejection sampler remain in place. Single evaluator,
TP supported, V1 runner and synchronous scheduling required for D3 accounting.
"""
import os
import sys
from pathlib import Path
from importlib.metadata import version

ROOT = str(Path(__file__).resolve().parent)
sys.path.insert(0, ROOT)
os.environ['PYTHONPATH'] = ROOT + os.pathsep + os.environ.get('PYTHONPATH','')
os.environ.setdefault('VLLM_USE_V2_MODEL_RUNNER','0')
if not version('vllm').startswith('0.29.'):
    raise RuntimeError('This compatibility launcher targets vLLM 0.29.x only')

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.launchers.api_server import entry
from spork_core.draft_reuse import REUSE_MODES


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
    reuse_mode = body.get('reuse_mode', 'single')
    if reuse_mode not in REUSE_MODES:
        return JSONResponse({'status':'error','error':'invalid_reuse_mode'},status_code=400)
    if not isinstance(rid,str) or not isinstance(tokens,list) or not tokens or not all(type(t) is int and t>=0 for t in tokens) or type(plen) is not int or plen<0:
        return JSONResponse({'status':'error','error':'invalid_payload'},status_code=400)
    results = await request.app.state.engine_client.collective_rpc('spork_set_draft_tokens',args=(rid,tokens,plen,reuse_mode))
    return JSONResponse(select_worker(results))


async def status(request: Request):
    results = await request.app.state.engine_client.collective_rpc('spork_get_status')
    result = select_worker(results)
    # Expose only the serving setting needed to prevent collecting against an
    # old/default server. This is not a claim that output invariance was tested.
    result['server_config'] = {
        'vllm_version': version('vllm'),
        'batch_invariant_requested': os.environ.get('VLLM_BATCH_INVARIANT', '0') == '1',
        'd3_reuse_modes': list(REUSE_MODES),
    }
    return JSONResponse(result)


async def clear(request: Request):
    results = await request.app.state.engine_client.collective_rpc('spork_clear')
    return JSONResponse(select_worker(results))


original_build_app = entry.build_app

def build_app(*args,**kwargs):
    app = original_build_app(*args,**kwargs)
    app.add_api_route('/spork/set_tokens',set_tokens,methods=['POST'])
    app.add_api_route('/spork/status',status,methods=['GET'])
    app.add_api_route('/spork/clear',clear,methods=['POST'])
    return app

entry.build_app = build_app

if __name__=='__main__':
    if not any(a.startswith('--worker-extension-cls') for a in sys.argv):
        sys.argv += ['--worker-extension-cls','spork_core.vllm029_d3.SporkWorkerExtension']
    if '--no-async-scheduling' not in sys.argv:
        sys.argv += ['--no-async-scheduling']
    entry.main()
