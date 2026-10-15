"""官方IRRA Tokenizer CPU检查；不加载图像/文本编码网络。"""
import unittest
from dataclasses import replace
from contracts import AttributeSpan,TextExample
from constraints import candidates
from tokenizer_probe import TokenizerProbe,official_probe
from test_policy import ByteTokenizer,example

class TokenizerTests(unittest.TestCase):
    def test_probe_exact_official_tokenize_for_plain_unicode_and_truncated(self):
        import importlib.util,sys
        from pathlib import Path
        root=Path("/home/lzf/Attack/IRRA-main/IRRA-main")
        sys.path.insert(0,str(root))
        spec=importlib.util.spec_from_file_location("h_v1_official_datasets_bases",root/"datasets/bases.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        probe=official_probe()
        for caption in ("A coat.","A cоat.","hello "*100):
            self.assertEqual(probe.inspect(caption).model_tokens,
                             tuple(module.tokenize(caption,probe.tokenizer).tolist()))
    def test_byte_probe_EOS_padding_and_truncate(self):
        p=TokenizerProbe(ByteTokenizer())
        self.assertEqual(len(p.inspect("coat").model_tokens),77)
        self.assertEqual(p.inspect("coat").model_tokens[0],1001)
        r=p.inspect("a"*80)
        self.assertTrue(r.truncated);self.assertEqual(r.model_tokens[-1],1002)
    def test_tokenizer_neutral_candidates_masked(self):
        class Neutral(ByteTokenizer):
            def encode(self,text):return [1,2,3]
        self.assertEqual(candidates(example(),"coat",0,set(),TokenizerProbe(Neutral())),{})
    def test_new_truncation_masked(self):
        x=example("o"+"b"*74,0,1)
        self.assertFalse(TokenizerProbe(ByteTokenizer()).inspect(x.caption).truncated)
        self.assertEqual(candidates(x,x.caption,0,set(),TokenizerProbe(ByteTokenizer())),{})
    def test_existing_truncated_caption_has_no_edits(self):
        x=example("coat"+"b"*80,0,4)
        self.assertEqual(candidates(x,x.caption,0,set(),TokenizerProbe(ByteTokenizer())),{})
    def test_official_byte_BPE_changes_and_lengths(self):
        probe=official_probe()
        audit=probe.pair_report("A person wears a coat.","A person wears a cоat.")
        self.assertTrue(audit["model_tokens_changed"])
        self.assertFalse(audit["new_truncation"])
        self.assertEqual(len(audit["after"]["model_tokens"]),77)
        self.assertGreater(audit["after"]["utf8_length"],audit["before"]["utf8_length"])
    def test_official_candidate_support_nonempty(self):
        probe=official_probe();support=candidates(example(),"coat",0,set(),probe)
        self.assertIn(1,support)
        for pos,rs in support.items():
            for r in rs:self.assertTrue(probe.legal_change("coat","coat"[:pos]+r+"coat"[pos+1:]))

if __name__=="__main__":unittest.main()
