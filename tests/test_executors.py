"""Grounded retrieval, generation prompt isolation, and optional real PEFT smoke test."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from intentexec.executors import EXPERTS, function_catalog, make_pool, specialist_rows, valid_call
from intentexec.lora import catalog_text, collate, prompt_ids, training_example


SCHEMA = [{"service_name": "Dining", "slots": [
    {"name": "city", "possible_values": ["Boston", "Chicago"]}],
    "intents": [{"name": "Find", "description": "Find restaurants",
                 "required_slots": ["city"], "optional_slots": {}},
                {"name": "Book", "description": "Book a reservation",
                 "required_slots": ["city"], "optional_slots": {}}]}]


def row(i, method, text):
    return {"id": str(i), "text": text, "intent": method, "gold": {
        "service": "Dining", "method": method, "arguments": {"city": "Boston"}}}


ROWS = [row(1, "Find", "USER: Find restaurants in Boston"),
        row(2, "Book", "USER: Reserve a restaurant in Boston"),
        row(3, "Book", "USER: I'd like to book\nSYSTEM: Which city\nUSER: Boston"),
        row(4, "Find", "USER: Show me an eatery in Boston")]


class CharacterTokenizer:
    """Makes every character one token for exact context budgeting assertions."""
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nassistant>"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class ExecutionTests(unittest.TestCase):
    def test_fit_only_grounding_overrides_retrieved_parameter(self):
        pool = make_pool(ROWS, SCHEMA, {"backend": "retrieval"})
        answer = pool.execute("base", "USER: Find restaurants in Chicago")
        self.assertEqual(answer, {"service": "Dining", "method": "Find", "arguments": {"city":"Chicago"}})
        with self.assertRaises(KeyError):
            pool.execute("invalid", "USER: Find restaurants")

    def test_shared_budget_with_specialist_preference(self):
        sets = [specialist_rows(ROWS, e, limit=2) for e in EXPERTS]
        self.assertEqual([len(s) for s in sets], [2, 2, 2])
        self.assertEqual({r["intent"] for r in sets[0]}, {"Find"})
        self.assertEqual({r["intent"] for r in sets[1]}, {"Book"})
        self.assertTrue(any("\n" in r["text"] for r in sets[2]))

    def test_invalid_generated_call_rejected(self):
        catalog = function_catalog(SCHEMA)
        for val in (None, {"service":[],"method":"Book","arguments":{}},
                    {"service":"Dining","method":"Buy","arguments":{}},
                    {"service":"Dining","method":"Book","arguments":{"wrong_key":"x"}}):
            self.assertFalse(valid_call(val, catalog))
        self.assertTrue(valid_call(ROWS[0]["gold"], catalog))

    def test_prompt_reserves_budget_and_mask_covers_only_response(self):
        tok, tools = CharacterTokenizer(), catalog_text(SCHEMA)
        base = "USER: hello"
        max_new = 90
        exact = len(prompt_ids(tok, base, tools, 2000, max_new)) + max_new
        prefix = prompt_ids(tok, "SYSTEM: hidden-old\n"+base, tools, exact, max_new)
        text = "".join(map(chr, prefix))
        self.assertNotIn("hidden-old", text)
        self.assertIn("Dining.Book", text)
        self.assertIn(base, text)
        with self.assertRaises(ValueError):
            prompt_ids(tok, base*10, tools, exact, max_new)
        ex = training_example(tok, {**ROWS[0], "text":base}, tools, exact, max_new)
        self.assertEqual(ex["labels"][:len(prefix)], [-100]*len(prefix))
        self.assertEqual(ex["labels"][len(prefix):], ex["input_ids"][len(prefix):])
        self.assertEqual(ex["labels"][-1], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional tensor backend")
    def test_padding_is_unsupervised(self):
        batch = collate([{"input_ids":[1,2],"labels":[-100,2]},
                         {"input_ids":[1,3,4],"labels":[-100,3,4]}], 0)
        self.assertEqual(batch["labels"][0].tolist(), [-100,2,-100])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1,1,0])

    @unittest.skipUnless(importlib.util.find_spec("peft"), "optional PEFT dependencies")
    def test_real_adapters_fit_reopen_disable_and_generate(self):
        import torch
        from tokenizers import Tokenizer, pre_tokenizers
        from tokenizers.models import WordLevel
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from intentexec.lora import train_pool
        with tempfile.TemporaryDirectory() as tmp:
            base, output = Path(tmp)/"base", Path(tmp)/"adapters"
            raw = Tokenizer(WordLevel({"[UNK]":0,"[EOS]":1,"city":2,"Boston":3,
                                       "{":4,"}":5,"Book":6,"Find":7}, unk_token="[UNK]"))
            raw.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]")
            tokenizer.chat_template = "{{ messages[0]['content'] }}\nAssistant:"
            tokenizer.save_pretrained(base)
            torch.manual_seed(17)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_layer=1, n_head=1,
                                    n_embd=8, n_positions=256, eos_token_id=1, bos_token_id=1))
            model.save_pretrained(base)
            cfg = {"base_path":str(base),"output":str(output),"seed":1118,"device":"cpu",
                   "per_expert":1,"rank":2,"epochs":1,"batch_size":1,"learning_rate":0.001,
                   "max_length":256,"max_new_tokens":64}
            manifest = train_pool(ROWS, ["Find","Book"], SCHEMA, cfg)
            self.assertEqual([entry["steps"] for entry in manifest["training"].values()], [1,1,1])
            pool = make_pool(ROWS, SCHEMA, {**cfg, "max_new_tokens":2,
                                            "adapters_dir":str(output),"backend":"lora"})
            self.assertEqual(set(pool.model.peft_config), set(EXPERTS))
            x = torch.tensor([[2,3]])
            pool.execute("transaction", ROWS[0]["text"])
            pool.execute("base", ROWS[0]["text"])
            with pool.model.disable_adapter(), torch.no_grad():
                torch.testing.assert_close(pool.model(x).logits, model.eval()(x).logits)
            with self.assertRaisesRegex(ValueError, "provenance"):
                make_pool(ROWS[:-1], SCHEMA, {**cfg,"adapters_dir":str(output),"backend":"lora"})
