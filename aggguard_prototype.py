"""AggGuard prototype + preliminary experiments (synthetic HR corpus, rule-based parser, no LLM in the loop)."""
import numpy as np, json, time, itertools
from sklearn.feature_extraction.text import TfidfVectorizer

DEPTS = ["Engineering", "Design", "Sales", "Finance", "People"]
OFFICES = ["Austin", "Seattle", "NYC", "Remote"]
SAL_LO, SAL_HI = 60_000, 260_000
RANGE = SAL_HI - SAL_LO


# ------------------------------------------------------------------ corpus
def make_corpus(seed, N=240):
    rng = np.random.default_rng(seed)
    # team sizes: 4 deliberately tiny teams (1,2,3,4) + 4 planted all-on-leave teams + normal teams
    tiny = [1, 2, 3, 4]
    planted = [5, 6, 7, 8]
    normal = []
    rem = N - sum(tiny) - sum(planted)
    while rem > 0:
        s = int(rng.integers(8, 19))
        s = min(s, rem) if rem - s >= 8 or rem - s == 0 else rem
        normal.append(s)
        rem -= s
    sizes = tiny + planted + normal
    kinds = ["tiny"] * 4 + ["planted"] * 4 + ["normal"] * len(normal)
    team_dept = [DEPTS[i % len(DEPTS)] for i in rng.permutation(len(sizes))]
    people = []
    pid = 0
    for t, (sz, kd) in enumerate(zip(sizes, kinds)):
        for _ in range(sz):
            lvl = int(rng.integers(1, 6))
            base = 70_000 + 22_000 * lvl + (15_000 if team_dept[t] == "Engineering" else 0)
            sal = float(np.clip(rng.normal(base, 9_000), SAL_LO, SAL_HI)).__round__(-2) if False else float(round(np.clip(rng.normal(base, 9_000), SAL_LO, SAL_HI), -2))
            people.append(dict(id=pid, name=f"Emp{pid:03d}", team=f"T{t:02d}", dept=team_dept[t], office=OFFICES[int(rng.integers(0, 4))],
                               level=lvl, gender=("F" if rng.random() < 0.45 else "M"), hire=int(rng.choice(range(2019, 2027), p=[.12, .14, .15, .15, .14, .12, .1, .08])),
                               salary=sal, leave=int(kd == "planted" or rng.random() < 0.08), kind=kd))
            pid += 1
    return people


def arrays(people):
    A = {k: np.array([p[k] for p in people]) for k in people[0]}
    return A


def chunk_text(p):
    return (f"Employee record. Name: {p['name']}. Department: {p['dept']}. Team: {p['team']}. Office: {p['office']}. "
            f"Level: L{p['level']}. Hired: {p['hire']}. Gender: {p['gender']}. Salary: ${int(p['salary'])}. On leave: {'yes' if p['leave'] else 'no'}.")


# ------------------------------------------------------------------ queries
class Q:
    __slots__ = ("attr", "op", "mask", "parent", "named")

    def __init__(self, attr, op, mask, parent=None, named=False):
        self.attr, self.op, self.mask, self.parent, self.named = attr, op, mask, parent, named

    @property
    def size(self):
        return int(self.mask.sum())


def mask_bits(mask):
    return int("".join("1" if b else "0" for b in mask), 2)


class Engine:
    def __init__(self, A):
        self.A = A

    def exact(self, q):
        v = self.A["salary"] if q.attr == "salary" else self.A["leave"].astype(float)
        x = v[q.mask]
        if q.op == "avg": return float(x.mean())
        if q.op == "count": return float(x.sum())
        if q.op == "max": return float(x.max())
        if q.op == "min": return float(x.min())
        if q.op == "lookup": return float(x[0])
        raise ValueError


# ------------------------------------------------------------------ defenses
class Defense:
    name = "none"

    def __init__(self, A, k=5, rng=None):
        self.A, self.k, self.eng = A, k, Engine(A)
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.t = {}

    def answer(self, q):  # -> (value or None, reason)
        return self.eng.exact(q), "ok"


class ACL(Defense):  # aggregate-only analyst role
    name = "ACL"

    def answer(self, q):
        if q.op == "lookup" or q.named: return None, "acl"
        return self.eng.exact(q), "ok"


class SingleTurn(ACL):  # SD-RAG-style: per-query individual-value filter, no memory
    name = "Single-turn"

    def answer(self, q):
        v, r = super().answer(q)
        if v is None: return v, r
        if q.size == 1: return None, "single-turn"
        return v, "ok"


class DPStyle(Defense):
    name = "DP-style"

    def __init__(self, A, k=5, rng=None, eps=1.0):
        self.EPS_Q, self.BUDGET = eps, 10.0 * eps
        super().__init__(A, k, rng)

    def reset(self):
        self.spent = 0.0

    def answer(self, q):
        if self.spent + self.EPS_Q > self.BUDGET + 1e-9: return None, "budget"
        self.spent += self.EPS_Q
        v = self.eng.exact(q)
        if q.attr == "salary":
            scale = RANGE / (q.size * self.EPS_Q) if q.op == "avg" else RANGE / self.EPS_Q
        else:
            scale = 1 / self.EPS_Q
        return v + float(self.rng.laplace(0, scale)), "ok"


class AggGuard(Defense):
    """ACL + small-cell (+ pairwise overlap audit) (+ exact linear auditor) (+ l-diversity)."""
    name = "AggGuard"

    def __init__(self, A, k=5, audit=True, rank=True, ldiv=False, cell=True, rng=None):
        self.audit, self.rank, self.ldiv, self.cell = audit, rank, ldiv, cell
        super().__init__(A, k, rng)

    def reset(self):
        self.sets = []  # bitmasks of answered sets (per attr)
        self.rows = {"salary": [], "leave": []}
        self.t = dict(cell=[], audit=[], rank=[], ldiv=[])

    def _timed(self, key, fn):
        t0 = time.perf_counter(); r = fn(); self.t[key].append(time.perf_counter() - t0); return r

    def answer(self, q):
        if self.cell and self._timed("cell", lambda: q.size < self.k): return None, "cell"
        if self.ldiv and q.attr == "leave" and q.op in ("count",):
            if self._timed("ldiv", lambda: len(np.unique(self.A["leave"][q.mask])) < 2): return None, "ldiv"
        b = mask_bits(q.mask)
        if self.audit:
            def pw():
                for (bb, a) in self.sets:
                    if a == q.attr and bb != b and 0 < (bb ^ b).bit_count() < self.k: return True
                return False
            if self._timed("audit", pw): return None, "audit"
        sumtype = q.op in ("avg", "count")
        if self.rank and sumtype:
            def rk():
                M = np.array(self.rows[q.attr] + [q.mask.astype(float)])
                _, s, Vt = np.linalg.svd(M, full_matrices=False)
                r = int((s > 1e-9).sum())
                P = (Vt[:r] ** 2).sum(0)
                return bool((P > 1 - 1e-8).any())
            if self._timed("rank", rk): return None, "rank"
        self.sets.append((b, q.attr))
        if sumtype: self.rows[q.attr].append(q.mask.astype(float))
        return self.eng.exact(q), "ok"


def configs(A, k, rng):
    return {
        "No guard": Defense(A, k, rng),
        "Access control": ACL(A, k, rng),
        "Differential privacy": DPStyle(A, k, rng, eps=1.0),
        "Small-group suppression": AggGuard(A, k, audit=False, rank=False, cell=True, rng=rng),
        "Query auditing": AggGuard(A, k, audit=True, rank=True, cell=False, rng=rng),
        "Both (ours)": AggGuard(A, k, audit=True, rank=True, cell=True, rng=rng),
        "Both + l-diversity": AggGuard(A, k, audit=True, rank=True, ldiv=True, cell=True, rng=rng),
    }


# ------------------------------------------------------------------ attacks
def close(est, true, binary):
    return abs(est - true) < 0.5 if binary else abs(est - true) <= 0.05 * true


def atk_small(D, A, rng, n=60):
    """F1: single-person group via tiny team or unique quasi-identifier combination."""
    N = len(A["id"])
    keys = list(zip(A["dept"], A["office"], A["gender"], A["hire"]))
    from collections import Counter
    c = Counter(keys)
    uniq = [i for i in range(N) if c[keys[i]] == 1]
    res = []
    for _ in range(n):
        t = int(rng.choice(uniq))
        mask = np.zeros(N, bool); mask[t] = True
        attr, op = [("salary", "avg"), ("salary", "max"), ("salary", "min"), ("leave", "count")][int(rng.integers(0, 4))]
        D.reset(); v, _ = D.answer(Q(attr, op, mask))
        true = A["salary"][t] if attr == "salary" else A["leave"][t]
        res.append(v is not None and close(v, true, attr == "leave"))
    return res


def atk_diff(D, A, rng, n=60):
    """F2: differencing -- group G vs G minus one person isolated by attribute filters."""
    N = len(A["id"]); res = []
    tries = 0
    while len(res) < n and tries < 10000:
        tries += 1
        d = rng.choice(DEPTS); G = A["dept"] == d
        if rng.random() < .5: G = G & (A["office"] == rng.choice(OFFICES))
        if G.sum() < D.k + 2: continue
        t = int(rng.choice(np.flatnonzero(G)))
        excl = (A["office"] == A["office"][t]) & (A["gender"] == A["gender"][t]) & (A["hire"] == A["hire"][t]) & (A["level"] == A["level"][t])
        G2 = G & ~excl
        if (G & excl).sum() != 1 or G2.sum() < D.k: continue
        attr = "salary" if rng.random() < .7 else "leave"
        D.reset()
        a1, _ = D.answer(Q(attr, "avg" if attr == "salary" else "count", G))
        a2, _ = D.answer(Q(attr, "avg" if attr == "salary" else "count", G2))
        if a1 is None or a2 is None: res.append(False); continue
        est = (a1 * G.sum() - a2 * G2.sum()) if attr == "salary" else (a1 - a2)
        true = A["salary"][t] if attr == "salary" else A["leave"][t]
        res.append(close(est, true, attr == "leave"))
    return res


def atk_triangle(D, A, rng, n=40):
    """F3: 3-query attack that evades the pairwise overlap rule: Q1+Q3-Q2 = 2*x_t."""
    N = len(A["id"]); res = []
    teams = [tm for tm in np.unique(A["team"]) if (A["team"] == tm).sum() >= D.k]
    keys = list(zip(A["dept"], A["office"], A["gender"], A["hire"]))
    from collections import Counter
    c = Counter(keys); uniq = [i for i in range(N) if c[keys[i]] == 1]
    tries = 0
    while len(res) < n and tries < 5000:
        tries += 1
        X, Y = rng.choice(teams, 2, replace=False)
        a = A["team"] == X; b = A["team"] == Y
        t = int(rng.choice(uniq))
        if a[t] or b[t]: continue
        tm = np.zeros(N, bool); tm[t] = True
        S1, S2, S3 = a | tm, a | b, b | tm
        D.reset()
        out = []
        for S in (S1, S2, S3):
            v, _ = D.answer(Q("salary", "avg", S)); out.append(v)
        if any(v is None for v in out): res.append(False); continue
        est = (out[0] * S1.sum() + out[2] * S3.sum() - out[1] * S2.sum()) / 2
        res.append(close(est, A["salary"][t], False))
    return res


def atk_direct(D, A, rng, n=30):
    N = len(A["id"]); res = []
    for _ in range(n):
        t = int(rng.integers(0, N)); m = np.zeros(N, bool); m[t] = True
        D.reset(); v, _ = D.answer(Q("salary", "lookup", m, named=True))
        res.append(v is not None and close(v, A["salary"][t], False))
    return res


def atk_homog(D, A, rng, n=None):
    """F5: homogeneity -- group >= k people, everyone shares the sensitive value (all on leave)."""
    N = len(A["id"]); res = []
    for tm in np.unique(A["team"][A["kind"] == "planted"]):
        m = A["team"] == tm
        for _ in range(3):
            D.reset(); v, _ = D.answer(Q("leave", "count", m))
            res.append(v is not None and abs(v - m.sum()) < 0.5)
    return res


FAMILIES = [("Small group (1 person)", atk_small), ("Differencing (2 queries)", atk_diff), ("Triangle (3 queries)", atk_triangle),
            ("Direct extraction", atk_direct), ("Homogeneous group", atk_homog)]


# ------------------------------------------------------------------ benign sessions
def benign_groups(A, rng):
    N = len(A["id"])
    fam = rng.choice(["dept", "office", "level", "dept_office", "dept_level", "hire", "team", "tinyteam"], p=[.14, .14, .12, .18, .14, .1, .1, .08])
    if fam == "dept": m = A["dept"] == rng.choice(DEPTS)
    elif fam == "office": m = A["office"] == rng.choice(OFFICES)
    elif fam == "level": m = A["level"] == int(rng.integers(1, 6))
    elif fam == "dept_office": m = (A["dept"] == rng.choice(DEPTS)) & (A["office"] == rng.choice(OFFICES))
    elif fam == "dept_level": m = (A["dept"] == rng.choice(DEPTS)) & (A["level"] == int(rng.integers(1, 6)))
    elif fam == "hire": m = A["hire"] == int(rng.integers(2019, 2027))
    elif fam == "team": m = A["team"] == rng.choice(np.unique(A["team"][A["kind"] == "normal"]))
    else: m = A["team"] == rng.choice(np.unique(A["team"][A["kind"] == "tiny"]))
    parent = (A["dept"] == A["dept"][np.flatnonzero(m)[0]]) if m.sum() else m
    return m, parent


def benign_eval(A, k, rng, widen, n_sessions=40, per=8, which=None, override=None):
    cfgs = override if override is not None else configs(A, k, np.random.default_rng(99))
    if which: cfgs = {n: cfgs[n] for n in which}
    out = {n: dict(safe=0, ans=0, correct=0, refused=0, small=0, small_ans=0, relerr=[]) for n in cfgs}
    stage_t = {"cell": [], "audit": [], "rank": []}
    for s in range(n_sessions):
        qs = []
        if s % 2 == 0:  # drill-down analyst session (nested, overlapping groups)
            d = rng.choice(DEPTS); base = A["dept"] == d
            subs = [base & (A["office"] == o) for o in OFFICES] + [base & (A["level"] >= 3), base & (A["level"] < 3), base & (A["hire"] <= 2023),
                    base & (A["hire"] >= 2024), base & (A["gender"] == "F"), base & (A["gender"] == "M"), base & (A["hire"] <= 2024), base & (A["hire"] <= 2025)]
            pick = [base] + [subs[i] for i in rng.permutation(len(subs))[:per - 1]]
            for m in pick:
                if m.sum() == 0: continue
                attr, op = [("salary", "avg"), ("salary", "avg"), ("leave", "count"), ("salary", "max")][int(rng.integers(0, 4))]
                qs.append((Q(attr, op, m, parent=base), m.sum() >= k))
        else:
            for _ in range(per):
                m, par = benign_groups(A, rng)
                if m.sum() == 0: continue
                attr, op = [("salary", "avg"), ("salary", "avg"), ("leave", "count"), ("salary", "max")][int(rng.integers(0, 4))]
                qs.append((Q(attr, op, m, parent=par), m.sum() >= k))
        for name, D in cfgs.items():
            D.reset(); o = out[name]
            for q, safe in qs:
                qq = q
                v, why = D.answer(qq)
                if v is None and not safe and widen and isinstance(D, AggGuard) and D.cell:
                    qq = Q(q.attr, q.op, q.parent)  # widen team -> department
                    v, why = D.answer(qq)
                if safe:
                    o["safe"] += 1
                    if v is None: o["refused"] += 1
                    else:
                        o["ans"] += 1
                        tr = Engine(A).exact(q)
                        err = abs(v - tr) / (abs(tr) if tr else 1) if q.op != "count" else abs(v - tr)
                        o["relerr"].append(err)
                        if (q.op == "count" and err < .5) or (q.op != "count" and err <= .01): o["correct"] += 1
                else:
                    o["small"] += 1
                    if v is not None: o["small_ans"] += 1
        # collect latencies
    ag = cfgs.get("Both (ours)")
    return out, ag


def side_finding(A, rng, m=10, n_q=300):
    """Plain top-m RAG with an idealised reader: groups answered from only the m retrieved chunks."""
    people = [dict(zip(A.keys(), vals)) for vals in zip(*A.values())]
    texts = [chunk_text(p) for p in people]
    vec = TfidfVectorizer().fit(texts); X = vec.transform(texts)
    rows = []
    for _ in range(n_q):
        m_, _ = benign_groups(A, rng)
        if m_.sum() == 0: continue
        i0 = np.flatnonzero(m_)[0]
        # query text mentions the defining attributes of the group
        parts = []
        for key, lab in (("team", "Team"), ("dept", "Department"), ("office", "Office"), ("level", "Level"), ("hire", "Hired")):
            vals = np.unique(A[key][m_])
            if len(vals) == 1 and (A[key] == vals[0]).sum() != len(A["id"]):
                parts.append(f"{lab}: {'L' if key == 'level' else ''}{vals[0]}")
        if not parts: continue
        qtxt = "average salary " + " ".join(parts)
        t0 = time.perf_counter()
        sims = (X @ vec.transform([qtxt]).T).toarray().ravel()
        top = np.argsort(-sims)[:m]
        t_ret = time.perf_counter() - t0
        got = [i for i in top if m_[i]]
        tr = A["salary"][m_].mean()
        if not got: est, nest = None, 0
        else: est, nest = float(A["salary"][got].mean()), len(got)
        rows.append((int(m_.sum()), est, tr, nest, t_ret))
    return rows


def run(seed, k=5):
    rng = np.random.default_rng(seed)
    people = make_corpus(seed); A = arrays(people)
    asr = {}
    for fam, fn in FAMILIES:
        asr[fam] = {}
        for name in configs(A, k, np.random.default_rng(seed)):
            cfgs = configs(A, k, np.random.default_rng(seed + 1))
            r = fn(cfgs[name], A, np.random.default_rng(seed + 7))
            asr[fam][name] = (int(sum(r)), len(r))
    ben, ag = benign_eval(A, k, np.random.default_rng(seed + 3), widen=True)
    side = side_finding(A, np.random.default_rng(seed + 5))
    lat = {kk: float(np.mean(v)) * 1e6 if v else 0.0 for kk, v in ag.t.items()}
    return asr, ben, side, lat


if __name__ == "__main__":
    import sys
    seeds = [11, 12, 13, 14, 15]
    all_asr, all_ben, all_side, all_lat = [], [], [], []
    for s in seeds:
        a, b, sd, lt = run(s)
        all_asr.append(a); all_ben.append(b); all_side += sd; all_lat.append(lt)
        print("seed", s, "done", flush=True)
    # aggregate
    names = list(all_asr[0][FAMILIES[0][0]].keys())
    res = {"asr": {}, "benign": {}, "side": {}, "latency_us": {}}
    for fam, _ in FAMILIES:
        res["asr"][fam] = {}
        for n in names:
            succ = sum(a[fam][n][0] for a in all_asr); tot = sum(a[fam][n][1] for a in all_asr)
            res["asr"][fam][n] = dict(rate=succ / tot, succ=succ, n=tot)
    for n in names:
        agg = dict(safe=0, ans=0, correct=0, refused=0, small=0, small_ans=0); rel = []
        for b in all_ben:
            for key in agg: agg[key] += b[n][key]
            rel += b[n]["relerr"]
        res["benign"][n] = dict(over_refusal=agg["refused"] / agg["safe"], accuracy=agg["correct"] / agg["safe"],
                                 median_relerr=float(np.median(rel)) if rel else None, small_answered=agg["small_ans"] / max(agg["small"], 1),
                                 n_safe=agg["safe"], n_small=agg["small"])
    # side finding buckets
    buckets = [(1, 4), (5, 9), (10, 19), (20, 39), (40, 999)]
    for lo, hi in buckets:
        rows = [r for r in all_side if lo <= r[0] <= hi]
        if not rows: continue
        ok = [r for r in rows if r[1] is not None and abs(r[1] - r[2]) / r[2] <= .01]
        errs = [abs(r[1] - r[2]) / r[2] for r in rows if r[1] is not None]
        cover = np.mean([r[3] / r[0] for r in rows])
        res["side"][f"{lo}-{hi if hi < 999 else '+'}"] = dict(n=len(rows), exact=len(ok) / len(rows), median_err=float(np.median(errs)) if errs else None, coverage=float(cover))
    res["side_overall"] = dict(n=len(all_side), exact=float(np.mean([1 if (r[1] is not None and abs(r[1] - r[2]) / r[2] <= .01) else 0 for r in all_side])),
                               retr_ms=float(np.mean([r[4] for r in all_side]) * 1e3))
    res["latency_us"] = {kk: float(np.mean([l[kk] for l in all_lat])) for kk in all_lat[0]}
    # k sweep: over-refusal / widen for AggGuard full
    ks = {}
    for k in (3, 5, 8, 12):
        ors, sm = [], []
        for s in seeds:
            A = arrays(make_corpus(s))
            b, _ = benign_eval(A, k, np.random.default_rng(s + 3), widen=True, which=["Both (ours)", "Small-group suppression"])
            ors.append(b["Both (ours)"]["refused"] / b["Both (ours)"]["safe"]); sm.append(b["Both (ours)"]["small_ans"] / max(b["Both (ours)"]["small"], 1))
        ks[k] = dict(over_refusal=float(np.mean(ors)), small_answered=float(np.mean(sm)))
    res["ksweep"] = ks
    sweep = {}
    for eps in (0.5, 1, 5, 20, 100):
        succ = tot = 0; corr = safe = 0; per_fam = {}
        for s in seeds:
            A = arrays(make_corpus(s))
            for fam, fn in FAMILIES:
                D = DPStyle(A, 5, np.random.default_rng(s + 1), eps=eps)
                r = fn(D, A, np.random.default_rng(s + 7))
                per_fam.setdefault(fam, []).append(float(np.mean(r)))
            ov = {"dp": DPStyle(A, 5, np.random.default_rng(99), eps=eps)}
            b, _ = benign_eval(A, 5, np.random.default_rng(s + 3), widen=False, override=ov)
            corr += b["dp"]["correct"]; safe += b["dp"]["safe"]
        fam_rates = {f: float(np.mean(v)) for f, v in per_fam.items()}
        sweep[str(eps)] = dict(asr_mean=float(np.mean(list(fam_rates.values()))), asr=fam_rates, accuracy=corr / safe)
    res["dp_sweep"] = sweep
    # mean ASR across families for every guard (for the frontier chart)
    res["asr_mean"] = {n: float(np.mean([res["asr"][f][n]["rate"] for f, _ in FAMILIES])) for n in names}
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
