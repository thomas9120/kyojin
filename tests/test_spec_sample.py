# CPU tests of exllamav3/generator/spec_sample.py (no GPU, no model).   python tests/test_spec_sample.py
#  1. law() of a step-by-step ComboSampler stack == an fp64 reference (softmax, sort, top-K, top-P cumsum <= p), padding gets 0
#  2. fused_law() == a slow reference of the fused kernel's truncation, with tie groups
#  3. draft d ~ q, verify(): emitted tokens follow p (chi-square), keep rate == sum min(p, q), never outside p's support
#  4. eligible(): sampled ComboSampler jobs qualify, greedy / masked / forced jobs do not
import os, sys, math, random, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.generator import spec_sample as ss
from exllamav3.generator.sampler import ComboSampler, ArgmaxSampler

def steps_prm(sampler, vs):
    return ("steps", sampler.steps[:-1], vs)

def ref_steps(l, T, k, tp):
    pr = torch.softmax(l.double() / T, 0); sp, si = torch.sort(pr, descending = True)
    if k: sp[k:] = 0; sp = sp / sp.sum()
    if tp < 1: m = torch.cumsum(sp, 0) <= tp; m[0] = True; sp = sp * m
    out = torch.zeros_like(pr); out[si] = sp
    return (out / out.sum()).float()

def ref_fused(l, prm):
    inv_t, inv_tf, filt, k, tp = prm
    v, order = torch.sort(l.double(), descending = True); v = v.tolist(); order = order.tolist()
    kept = [True] * len(v)
    if filt & 1 and 0 < k < len(v): kept = [x >= v[k - 1] for x in v]
    if filt & 2 and tp < 1:
        w = [math.exp((x - v[0]) * inv_tf) if kk else 0.0 for x, kk in zip(v, kept)]; z = sum(w); acc = 0.0; c = len(v) - 1
        for i, x in enumerate(w):
            acc += x
            if acc > tp * z: c = i; break
        kept = [kk and (x > v[c] or x == v[0]) for x, kk in zip(v, kept)]
    pw = [math.exp((x - v[0]) * inv_t) if kk else 0.0 for x, kk in zip(v, kept)]; z = sum(pw)
    out = [0.0] * len(v)
    for i, x in zip(order, pw): out[i] = x / z
    return torch.tensor(out)

class FakeJob:
    def __init__(self, sampler, **kw):
        self.sampler = sampler; self.filters = []; self.forced_ids = None; self.return_probs = False
        self.return_top_tokens = 0; self.device_logit_mask = None; self.sequences = [None]
        self.generator = type("G", (), {"tokenizer": type("T", (), {"actual_vocab_size": 100})()})()
        self.__dict__.update(kw)

class TestSpecSample(unittest.TestCase):

    def test_steps_law(self):
        g = torch.Generator().manual_seed(1)
        for T, k, tp in ((1.0, 0, 0.95), (1.0, 20, 0.95), (0.7, 0, 0.9)):
            prm = steps_prm(ComboSampler(temperature = T, top_k = k, top_p = tp), 300)
            for _ in range(100):
                l = torch.randn(310, generator = g) * (0.5 + 3 * torch.rand(1, generator = g))
                p = ss.law(l, prm)
                self.assertEqual(float(p[300:].abs().max()), 0.0)
                self.assertLess(float((p[:300] - ref_steps(l[:300], T, k, tp)).abs().max()), 1e-6)

    def test_fused_law_ties(self):
        g = torch.Generator().manual_seed(2)
        for prm in ((1.0, 1.0, 2, 0, 0.95), (1.0, 1.0, 3, 20, 0.95), (1 / 0.7, 1 / 0.7, 2, 0, 0.9)):
            for _ in range(100):
                l = (torch.randn(300, generator = g) * 2).half().float()
                l[torch.randint(0, 300, (40,), generator = g)] = l[torch.randint(0, 300, (1,), generator = g)]
                self.assertLess(float((ss.law(l, ("fused", prm, 300)) - ref_fused(l, prm)).abs().max()), 1e-6)

    def test_exact_in_distribution(self):
        g = torch.Generator().manual_seed(3); rng = random.Random(4)
        V, N = 48, 40000
        for prm in (steps_prm(ComboSampler(temperature = 1.0, top_p = 0.95), V), ("fused", (1.0, 1.0, 3, 20, 0.95), V)):
            lt = torch.randn(V, generator = g) * 2.0; ld = lt + torch.randn(V, generator = g) * 1.2
            p = ss.law(lt, prm); q = ss.law(ld, prm)
            cnt = torch.zeros(V); keep = 0
            for _ in range(N):
                d = int(ss.draw(q, rng.random())); x = int(ss.verify(lt, q, d, prm, rng)); cnt[x] += 1; keep += (x == d)
            m = p > 0
            self.assertEqual(int(cnt[~m].sum()), 0)
            e = p[m] * N; chi = float(((cnt[m] - e) ** 2 / e).sum()); dof = int(m.sum()) - 1
            self.assertLess((chi - dof) / math.sqrt(2 * dof), 4.0)
            self.assertLess(abs(keep / N - float(torch.minimum(p, q).sum())), 0.01)

    def test_a_position_is_verified_only_when_it_is_reached(self):
        # The round records one law per draft position, but the verify of position i runs when the
        # consume loop reaches i (generator._spec_verify_position): an earlier rejection leaves the
        # rest of the window unverified, un-drawn and unretained.
        from exllamav3.generator.generator import _spec_verify_position
        g = torch.Generator().manual_seed(5)
        V, W = 64, 6
        prm = steps_prm(ComboSampler(temperature = 1.0, top_p = 0.95), V)
        logits = torch.randn(1, W, V + 4, generator = g)
        laws = [ss.law(logits[0, i, :], prm) for i in range(W - 1)]       # the last position is the bonus
        drafts = torch.tensor([[int(ss.draw(q, 0.5)) for q in laws] + [0]])
        # the spec path draws from the job's forked stream (F9): the target stream is untouched
        job = type("J", (), {"rng": random.Random(7), "spec_rng": random.Random(7)})()
        mirror = random.Random(7)
        sr = (job, prm, list(laws))

        for i in range(W - 1):
            next_token, k_t, k_p, prob = _spec_verify_position(logits, sr, i, drafts, 0)
            self.assertEqual(next_token.shape, (1, 1))
            self.assertIsNone(k_t or k_p or prob)
            p = ss.law(logits[0, i, :], prm)
            self.assertGreater(float(p[int(next_token)]), 0.0)           # never outside p's support
            self.assertIsNone(sr[2][i])                                 # the law is released as consumed
            mirror.random()                                             # keep test
            mirror.random()                                             # residual draw
        self.assertEqual(job.spec_rng.getstate(), mirror.getstate())     # two uniforms per position, no more
        self.assertEqual(job.rng.getstate(), random.Random(7).getstate())  # ... and none from the target stream

        # a rejection at position 0 leaves the laws of 1.. untouched: nothing was computed for them
        sr2 = (type("J", (), {"rng": random.Random(7), "spec_rng": random.Random(7)})(), prm, list(laws))
        _spec_verify_position(logits, sr2, 0, drafts, 0)
        self.assertTrue(all(q is not None for q in sr2[2][1:]))

    def test_draw_never_returns_a_zero_mass_index(self):
        # F4: a u within one ULP of 1.0 used to fall through the searchsorted clamp to the last index,
        # which may carry no mass. Sweep the whole unit interval over a law with a zero tail.
        p = torch.tensor([0.5, 0.5] + [0.0] * 62)
        for u in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0 - 1e-7, 1.0 - 5e-9, 1.0]:
            self.assertGreater(float(p[int(ss.draw(p, u))]), 0.0)
        for u in (i / 1000 for i in range(1001)):
            self.assertGreater(float(p[int(ss.draw(p, u))]), 0.0)

    def test_verify_rejects_an_impossible_draft(self):
        # F4: q[d] == 0 with p[d] > 0 kept unconditionally (0 < p[d]); now it rejects and the residual
        # draw decides. p = [0.5, 0.3, 0.2], q one-hot at 1, d = 0: the old code emits 0 every time, the
        # residual emits 0 w.p. 5/7 and 2 w.p. 2/7, so any 2 at all proves the guard.
        import math as _math
        plain = ("fused", (1.0, 1.0, 0, 0, 1.0), 3)
        onehot1 = ("fused", (1.0, 1.0, 3, 1, 1.0), 3)
        lt = torch.tensor([_math.log(5), _math.log(3), _math.log(2)])
        q = ss.law(torch.tensor([0.0, 10.0, 0.0]), onehot1)
        self.assertEqual(q.tolist(), [0.0, 1.0, 0.0])
        rng = random.Random(11)
        n2 = sum(int(ss.verify(lt, q, 0, plain, rng)) == 2 for _ in range(2000))
        self.assertGreater(n2, 100)     # residual rate 2/7 -> ~571; the old code gives exactly 0
        self.assertLess(n2, 1000)

    def test_spec_path_leaves_the_target_stream_alone(self):
        # F9: draws and verifies consume only the forked stream, so spec on/off give identical target seeds
        from exllamav3.generator.generator import _spec_verify_position
        g = torch.Generator().manual_seed(9)
        V = 32
        prm = steps_prm(ComboSampler(temperature = 1.0, top_p = 0.95), V)
        logits = torch.randn(1, 2, V, generator = g)
        q = ss.law(logits[0, 0, :], prm)
        job = type("J", (), {"rng": random.Random(13), "spec_rng": random.Random(13)})()
        ss.draw(q, job.spec_rng.random())
        _spec_verify_position(logits, (job, prm, [q]), 0, torch.tensor([[int(ss.draw(q, 0.5)), 0]]), 0)
        self.assertEqual(job.rng.getstate(), random.Random(13).getstate())
        self.assertNotEqual(job.spec_rng.getstate(), random.Random(13).getstate())

    def test_eligible(self):
        def kind(job):
            # eligible() returns the law params or None; these jobs must be eligible
            prm = ss.eligible(job)
            assert prm is not None
            return prm[0]
        self.assertEqual(kind(FakeJob(ComboSampler(temperature = 1.0, top_p = 0.95))), "steps")
        self.assertEqual(kind(FakeJob(ComboSampler(temperature = 1.0, top_k = 20, top_p = 0.95))), "steps")
        self.assertIsNone(ss.eligible(FakeJob(ArgmaxSampler())))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 0.0))))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 1.0, rep_p = 1.1))))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 1.0), device_logit_mask = torch.zeros(1))))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 1.0), forced_ids = [1])))

if __name__ == "__main__":
    unittest.main()
