"""D3 adapter for vLLM 0.29's synchronous V1 ngram runner.

Keeps vLLM's proposer and rejection sampler; substitutes a single draft on the
bound main request only. Verification counts exclude the bonus/correction token.
No GPU imports: the binding and accounting can be tested on CPU.
"""
from __future__ import annotations
import inspect


class D3Bridge:
    def __init__(self, runner):
        if getattr(runner, 'use_async_scheduling', False):
            raise RuntimeError('D3 0.29 smoke requires --no-async-scheduling')
        self.runner = runner
        self.drafter = runner.drafter
        original = self.drafter.propose
        if list(inspect.signature(original).parameters)[:2] != ['num_speculative_tokens', 'sampled_token_ids']:
            raise RuntimeError('Unsupported ngram proposer signature; expected vLLM 0.29')
        self.original_propose = original
        self.original_bookkeeping = runner._bookkeeping_sync
        self.states = {}
        self.pending = {}
        self.counts = dict(set_count=0, inject_count=0, draft_token_count=0,
                           verified_injections=0, verified_draft_tokens=0,
                           accepted_draft_tokens=0, dropped_injections=0,
                           prefix_mismatches=0, accounting_errors=0)
        self.last_inject_info = None
        self.last_verification = None
        self.drafter.propose = self.propose
        runner._bookkeeping_sync = self.bookkeeping

    def register(self, request_id, draft_tokens, prompt_len):
        if not request_id or not draft_tokens or prompt_len < 0:
            return {'status':'error','error':'invalid_request_or_draft'}
        # Completion API adds cmpl- and prompt index; the engine may add a UUID.
        external = 'cmpl-' + request_id + '-0'
        ids = [r for r in self.runner.requests if r == external or r.startswith(external + '-')]
        if len(ids) != 1:
            return {'status':'error','error':'main_request_not_active_or_ambiguous','matches':len(ids)}
        internal = ids[0]
        # Do not discard a previously proposed draft awaiting verification.
        self.states[internal] = {'tokens':list(draft_tokens),'prompt_len':prompt_len,'fired':False}
        self.counts['set_count'] += 1
        return {'status':'ok','request_id':request_id,'engine_request_id':internal,'n_tokens':len(draft_tokens)}

    def snapshot(self):
        return {**self.counts,'patched':True,'has_drafter':True,
                'active_requests':len(self.states),'pending_verifications':len(self.pending),
                'last_inject_info':self.last_inject_info,'last_verification':self.last_verification}

    def clear(self):
        self.counts['dropped_injections'] += len(self.pending)
        self.pending.clear()
        self.states.clear()

    def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        drafts = self.original_propose(num_speculative_tokens, sampled_token_ids,
                                       num_tokens_no_spec, token_ids_cpu, slot_mappings=slot_mappings)
        if num_speculative_tokens <= 0:
            return drafts
        for i, rid in enumerate(self.runner.input_batch.req_ids):
            state = self.states.get(rid)
            if not state or state['fired'] or not sampled_token_ids[i]:
                continue
            end = int(num_tokens_no_spec[i])
            seq = token_ids_cpu[i, state['prompt_len']:end].tolist()
            positions = [j for j,t in enumerate(seq) if t == 151657]
            if not positions:
                continue
            body = seq[positions[-1]+1:]
            draft = state['tokens']
            # Align to the whole already-generated body, not just its common prefix.
            if body != draft[:len(body)]:
                state['fired'] = True
                self.counts['prefix_mismatches'] += 1
                continue
            k = min(num_speculative_tokens, self.drafter.k, self.drafter.max_model_len-end)
            proposal = draft[len(body):len(body)+max(0,k)]
            state['fired'] = True
            if not proposal:
                continue
            drafts[i] = proposal
            self.pending[rid] = proposal
            self.counts['inject_count'] += 1
            self.counts['draft_token_count'] += len(proposal)
            self.last_inject_info = {'engine_request_id':rid,'batch_row':i,'draft_tokens':len(proposal),'body_tokens_already_generated':len(body)}
        return drafts

    def bookkeeping(self, scheduler_output, *args, **kwargs):
        output = self.original_bookkeeping(scheduler_output, *args, **kwargs)
        # 0.29 _bookkeeping_sync returns CPU samples in [3], request IDs in [5].
        samples, req_ids = output[3], output[5]
        for i,rid in enumerate(req_ids):
            proposed = self.pending.pop(rid,None)
            if proposed is None:
                continue
            scheduled = scheduler_output.scheduled_spec_decode_tokens.get(rid,[])
            returned = samples[i]
            if not scheduled or not returned:
                self.counts['dropped_injections'] += 1
                continue
            accepted = len(returned)-1  # exclude bonus/correction token
            if (scheduled != proposed[:len(scheduled)] or len(scheduled)>len(proposed)
                    or not 0 <= accepted <= len(scheduled) or returned[:accepted] != scheduled[:accepted]):
                self.counts['accounting_errors'] += 1
                continue
            self.counts['verified_injections'] += 1
            self.counts['verified_draft_tokens'] += len(scheduled)
            self.counts['accepted_draft_tokens'] += accepted
            self.last_verification = {'engine_request_id':rid,'scheduled_draft_tokens':len(scheduled),
                                      'accepted_draft_tokens':accepted,'returned_tokens_including_bonus':len(returned)}
        return output


class SporkWorkerExtension:
    """Imported by vLLM on every worker, independent of fork/spawn mode."""
    def spork_set_draft_tokens(self, request_id, draft_tokens, prompt_len=0):
        runner = self.model_runner
        if getattr(runner,'drafter',None) is None:
            return {'status':'inactive','has_drafter':False}
        if not hasattr(runner,'_spork_d3_bridge'):
            runner._spork_d3_bridge = D3Bridge(runner)
        return runner._spork_d3_bridge.register(request_id,draft_tokens,prompt_len)

    def spork_get_status(self):
        runner = self.model_runner
        bridge = getattr(runner,'_spork_d3_bridge',None)
        if bridge:
            return bridge.snapshot()
        return {'patched':False,'has_drafter':getattr(runner,'drafter',None) is not None}

    def spork_clear(self):
        bridge = getattr(self.model_runner,'_spork_d3_bridge',None)
        if bridge:
            bridge.clear()
        return {'status':'cleared',**self.spork_get_status()}
