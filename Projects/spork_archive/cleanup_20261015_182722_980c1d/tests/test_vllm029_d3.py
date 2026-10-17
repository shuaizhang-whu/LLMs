"""CPU-only tests of D3 request routing and verifier accounting."""
import unittest
from types import SimpleNamespace
import numpy as np
from spork_core.vllm029_d3 import D3Bridge, SporkWorkerExtension

class Drafter:
    k=4
    max_model_len=100
    def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        return [[90] for _ in sampled_token_ids]

class Runner:
    use_async_scheduling=False
    def __init__(self):
        self.rid='cmpl-main-0-abcd1234'
        self.requests={self.rid:object(),'cmpl-probe-0-eeeeeeee':object()}
        self.input_batch=SimpleNamespace(req_ids=['cmpl-probe-0-eeeeeeee',self.rid])
        self.drafter=Drafter()
        self.output=None
    def _bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        return self.output

class D3Tests(unittest.TestCase):
    def setUp(self):
        self.r=Runner();self.b=D3Bridge(self.r)
        self.tokens=np.array([[1,151657,10,0,0],[1,151657,10,0,0]])
    def register(self):
        return self.b.register('main',[10,11,12,13,14],1)
    def propose(self,k=2):
        return self.r.drafter.propose(k,[[10],[10]],np.array([3,3]),self.tokens)
    def test_stable_id_does_not_inject_probe_row(self):
        self.assertEqual(self.register()['status'],'ok')
        self.assertEqual(self.propose(),[[90],[11,12]])
        self.assertEqual(self.b.snapshot()['draft_token_count'],2)
        self.assertEqual(self.propose(),[[90],[90]])
    def test_reordered_batch_keeps_binding(self):
        self.register();self.r.input_batch.req_ids.reverse()
        self.assertEqual(self.propose(),[[11,12],[90]])
    def test_generated_body_mismatch_falls_back(self):
        self.register();self.tokens[1,2]=999
        self.assertEqual(self.propose(),[[90],[90]])
        self.assertEqual(self.b.snapshot()['prefix_mismatches'],1)
    def test_no_active_request_is_not_success(self):
        self.assertEqual(self.b.register('missing',[10],1)['status'],'error')
        self.assertEqual(self.b.snapshot()['set_count'],0)
    def verify(self,scheduled,returned):
        self.r.output=(None,None,None,[returned],None,[self.r.rid],None,[])
        self.r._bookkeeping_sync(SimpleNamespace(scheduled_spec_decode_tokens={self.r.rid:scheduled}))
    def test_acceptance_excludes_bonus_token(self):
        self.register();self.propose();self.verify([11,12],[11,12,77])
        self.assertEqual(self.b.snapshot()['accepted_draft_tokens'],2)
        self.assertEqual(self.b.snapshot()['verified_injections'],1)
    def test_rejection_can_accept_zero(self):
        self.register();self.propose();self.verify([11,12],[88])
        self.assertEqual(self.b.snapshot()['accepted_draft_tokens'],0)
        self.assertEqual(self.b.snapshot()['verified_draft_tokens'],2)
    def test_scheduler_shortens_draft(self):
        self.register();self.propose();self.verify([11],[11,66])
        self.assertEqual(self.b.snapshot()['accepted_draft_tokens'],1)
    def test_unrelated_scheduled_tokens_not_counted(self):
        self.register();self.propose();self.verify([51,52],[51,52,77])
        self.assertEqual(self.b.snapshot()['accepted_draft_tokens'],0)
        self.assertEqual(self.b.snapshot()['accounting_errors'],1)
    def test_clear_removes_state(self):
        self.register();self.propose();self.b.clear()
        self.assertEqual(self.b.snapshot()['active_requests'],0)
        self.assertEqual(self.b.snapshot()['pending_verifications'],0)
        self.assertEqual(self.propose(),[[90],[90]])
    def test_fresh_worker_reports_zero_state_without_installing_bridge(self):
        worker=SporkWorkerExtension();worker.model_runner=Runner()
        status=worker.spork_get_status()
        self.assertTrue(status['has_drafter'])
        self.assertFalse(status['patched'])
        self.assertEqual(status['active_requests'],0)
        self.assertEqual(status['pending_verifications'],0)
        self.assertEqual(status['verified_injections'],0)
        self.assertFalse(hasattr(worker.model_runner,'_spork_d3_bridge'))
        self.assertEqual(worker.spork_clear()['status'],'cleared')

if __name__=='__main__':unittest.main()
