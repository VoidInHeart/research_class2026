import unittest
import numpy as np

from intentexec.compare import costs, jev_step
from intentexec.experiment import infer_one, policy_decisions
from intentexec.jev import Jev
from intentexec.metrics import call_parts, paired_interval


class Gate:
    def encode(self,texts):
        return np.asarray([[t == "previous turns plus current user query"] for t in texts])
    def predict_encoded(self, features, texts=None):
        return np.tile([.35, .65], (len(features), 1)), ~features[:, 0]
    def predict(self, texts):
        return self.predict_encoded(self.encode(texts))


class RouterSpy:
    def choose(self, x):
        assert x.shape[1] == 2
        return np.ones(len(x), dtype=int)


class Pool:
    def __init__(self):
        self.calls = []
    def execute(self, executor_id, text):
        self.calls.append((executor_id, text))
        return {"service":"Restaurants", "method":"Find", "arguments":{"city":"NYC"}}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.bundle = dict(recognizer=Gate(), experts=["first","second"], base="base",
                           known=["intent A","intent B"], routers={"matrix":RouterSpy()})
        self.pool = Pool()
        self.text = "previous turns plus current user query"

    def test_single_execution_and_bypass(self):
        result = infer_one(self.bundle,self.pool,"matrix",self.text)
        self.assertEqual(self.pool.calls, [("second",self.text)])
        self.assertEqual(result["intent_result"], {"intent A":.35, "intent B":.65})
        self.bundle["routers"].clear()
        oos = infer_one(self.bundle,self.pool,"matrix","unseen request")
        self.assertEqual((oos["intent_result"], self.pool.calls[-1]), ("OOS",("base","unseen request")))
        self.assertEqual(oos["route_ms"],0)

    def test_batch_and_online_choose_same_executors(self):
        public = [{"id":str(i), "text":t} for i,t in enumerate([self.text,"OOD"])]
        rows = policy_decisions(self.bundle,public)
        for row, decision in zip(public,rows):
            for name in ("always_base","matrix","matrix_no_gate"):
                self.assertEqual(decision["selected"][name],
                                 infer_one(self.bundle,self.pool,name,row["text"])["executor"])

    def test_jev_contract_and_no_call_on_bypass(self):
        sent = []
        def transport(payload):
            sent.append(payload)
            return {"answers":{"executor":{"choice":"second","type":"choice"}}, "usage":{"input_tokens":10}}
        api = Jev("test-double",transport=transport)
        opts = {"first":"A", "second":"B"}
        for text in [self.text,"outside"]:
            jev_step(self.bundle,self.pool,api,opts,False,text)
        self.assertEqual(len(sent),1)
        self.assertNotIn(self.text,sent[0]["state"])
        self.assertIn("intent B", sent[0]["state"])
        self.assertEqual(self.pool.calls[-1], ("base","outside"))
        jev_step(self.bundle,self.pool,api,dict(opts,base="fallback"),True,"outside")
        self.assertEqual(sent[-1]["state"], "outside")
        failing=Jev("mock",transport=lambda _:{"answers":{"executor":{"choice":"unlisted"}}})
        failed=jev_step(self.bundle,self.pool,failing,opts,True,self.text)
        self.assertTrue(failed["failed_decision"])
        self.assertIsNone(failed["call"])
        self.assertEqual(len(self.pool.calls),3)

    def test_quality_and_cost(self):
        call={"service":"x","method":"f","arguments":{"city":"San Jose"}}
        self.assertEqual(call_parts(dict(call,arguments={"city":"  san   jose"}),call), (True,True,True))
        self.assertFalse(call_parts(dict(call, arguments={}),call)[0])
        a=[{"id":"1","dialogue_id":"d","correct":True}]
        self.assertEqual(paired_interval(a,a,repeats=5)["ci95"],[0,0])
        usage = [{"total_ms":3600,"correct":True,"api_called":True,
                  "usage":{"input_tokens":100,"output_tokens":10}}]
        charge=costs(usage,1,4,10,.05)
        self.assertAlmostEqual(charge["estimated_total_usd_per_1000"],1.55)
        self.assertIsNone(costs(usage,1,None,None,None)["estimated_total_usd_per_1000"])
        self.assertIsNone(costs([dict(usage[0],failed_decision=True)],1,4,10,.05)["estimated_api_usd"])
