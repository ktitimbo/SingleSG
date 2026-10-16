// Fits of the 2026-07-06 spin-resolved separation data (spin_resolved_sep_20260706.jld2, exported to CSV).
// Run: node run_fits.js      (writes results.json and the CSV files used by the report)
const fs = require('fs');
const M = require('./cqd_model.js');
const D = M.loadData(__dirname + '/spin_resolved_sep_20260706.csv');
const N = D.I.length;
const out = { data: D, N };

const chiVec = (model, y, s) => model.reduce((c, m, i) => c + ((m - y[i]) / s[i]) ** 2, 0);
// cache of responses at the data currents, keyed by D1|f|dir
const cache = new Map();
const R = (D1, f, dir) => { const k = `${D1}|${f}|${dir}`; if (!cache.has(k)) cache.set(k, M.responses(D1, f, dir, D.I)); return cache.get(k); };
const chiPar = (D1, f, dir) => { const r = R(D1, f, dir); return chiVec(r.pts.map(q => M.dPar(r, q)), D.par, D.sPar); };
const chiAnti = (D1, f, dir, wf, p) => { const r = R(D1, f, dir); return chiVec(r.pts.map(q => M.dAnti(r, q, M.W[wf](q.D0, p))), D.anti, D.sAnti); };
// best weight parameter for given D1 (golden section on [0, pmax])
const bestW = (D1, f, dir, wf, pmax = 5) => M.golden(p => chiAnti(D1, f, dir, wf, p), 0, pmax, 60);
// curvature error of a 1-D chi2 at its minimum: sigma = sqrt(2 / chi'')
// robust version for wide parameter ranges: log-grid scan, then golden section around the best node
const bestWrobust = (D1, f, dir, wf) => { const xs = [...Array(61)].map((_, i) => -3 + i * 0.1); let ib = 0, b = Infinity;
  xs.forEach((x, i) => { const c = chiAnti(D1, f, dir, wf, 10 ** x); if (c < b) { b = c; ib = i; } });
  return M.golden(p => chiAnti(D1, f, dir, wf, p), 10 ** xs[Math.max(0, ib - 1)], 10 ** xs[Math.min(60, ib + 1)], 60); };
const curvErr = (fn, x) => { const h = Math.max(1e-4, 1e-3 * Math.abs(x)); return Math.sqrt(2 / ((fn(x + h) - 2 * fn(x) + fn(x - h)) / (h * h))); };

// ---------- Strategy 2: parallel only, then borrow k_i ----------
out.strategy2 = [];
for (const f of [1, 7]) for (const dir of ['baseline', 5.7571]) {
  const chi = D1 => chiPar(D1, f, dir);
  const [D1, c0] = M.minimizeD1(chi);
  const [lo, hi] = M.profileInterval(chi, D1, c0);
  const r = R(D1, f, dir);
  const res = { f, dir, D1, ki: M.ki(D1), kiLo: M.ki(lo), kiHi: M.ki(hi), chi2Par: c0, DIR: r.DIR, Z0: r.Z0,
    alpha0: D1 / (M.I_REF * f), chi2AntiStrict: chiAnti(D1, f, dir, 'none', 0) };
  for (const wf of ['lin', 'exp']) {
    const [p, c] = bestW(D1, f, dir, wf);
    res[wf] = { p, err: curvErr(q => chiAnti(D1, f, dir, wf, q), p), chi2: c };
  }
  out.strategy2.push(res);
  console.log(`S2 f=${f} dir=${dir}: D1=${D1.toFixed(4)} k_i=${res.ki.toExponential(3)} [${res.kiLo.toExponential(2)},${res.kiHi.toExponential(2)}] chi2par=${c0.toFixed(2)} DIR=${r.DIR.toFixed(3)} | strict ${res.chi2AntiStrict.toFixed(1)} | kappa=${res.lin.p.toFixed(4)}+-${res.lin.err.toFixed(4)} chi2=${res.lin.chi2.toFixed(2)} | RLC=${res.exp.p.toFixed(4)}+-${res.exp.err.toFixed(4)} chi2=${res.exp.chi2.toFixed(2)}`);
}

// ---------- Strategy 1: both sets together (D1 and RLC free) ----------
out.strategy1 = [];
for (const f of [1, 7]) {
  const prof = D1 => { const [p, ca] = bestW(D1, f, 'baseline', 'exp'); return chiPar(D1, f, 'baseline') + ca; };
  const [D1, c0] = M.minimizeD1(prof);
  const [lo, hi] = M.profileInterval(prof, D1, c0);
  const [RLC] = bestW(D1, f, 'baseline', 'exp');
  const res = { f, D1, ki: M.ki(D1), kiLo: M.ki(lo), kiHi: M.ki(hi), RLC, chi2: c0, chi2Par: chiPar(D1, f, 'baseline'),
    DIR: R(D1, f, 'baseline').DIR };
  out.strategy1.push(res);
  console.log(`S1 f=${f}: D1=${D1.toFixed(4)} k_i=${res.ki.toExponential(3)} [${res.kiLo.toExponential(2)},${res.kiHi.toExponential(2)}] RLC=${RLC.toFixed(3)} chi2=${c0.toFixed(2)} (${2*N} pts)`);
}

// ---------- Strategy 3: antiparallel only ----------
const D1grid = [0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0];
out.strategy3 = [];
for (const f of [1, 7]) {
  const pure = D1grid.map(D1 => chiAnti(D1, f, 'baseline', 'none', 0));
  const withW = D1grid.map(D1 => bestWrobust(D1, f, 'baseline', 'exp')[1]);
  out.strategy3.push({ f, D1: D1grid, ki: D1grid.map(M.ki), chi2Pure: pure, chi2WithRLC: withW });
  console.log(`S3 f=${f}: pure ${pure.map(v => v.toFixed(1)).join(' ')}`);
  console.log(`        withRLC ${withW.map(v => v.toFixed(2)).join(' ')}`);
}

// ---------- Strategy 4 / sensitivity: scan the SG0 divisor f ----------
out.fscan = [];
for (const f of [1, 2, 3, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10]) {
  const [D1p, cp] = M.minimizeD1(D1 => chiPar(D1, f, 'baseline'));
  const prof = D1 => chiPar(D1, f, 'baseline') + bestW(D1, f, 'baseline', 'exp')[1];
  const [D1b, cb] = M.minimizeD1(prof);
  out.fscan.push({ f, kiPar: M.ki(D1p), chi2Par: cp, kiBoth: M.ki(D1b), chi2Both: cb });
  console.log(`f=${f}: parallel-only k_i=${M.ki(D1p).toExponential(3)} chi2=${cp.toFixed(2)} | both k_i=${M.ki(D1b).toExponential(3)} chi2=${cb.toFixed(2)}`);
}

// ---------- slope capacity (why large f fails) ----------
out.capacity = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0].map(D1 => {
  const g = M.geometry(D1, 'baseline'), h = 1e-4;
  const gcap = D1 * (M.Zpar(h, D1, g.rho) - g.Z0) / h / g.Z0;
  return { D1, DIR: g.DIR, Z0: g.Z0, g: gcap };
});

// ---------- CSV outputs ----------
const grid = Array.from({ length: 371 }, (_, i) => +(i * 0.01).toFixed(2));
const fx = v => (Math.abs(v) < 5e-5 ? 0 : v).toFixed(4);
for (const s of out.strategy2.filter(s => s.dir === 'baseline')) {
  const r = M.responses(s.D1, s.f, 'baseline', grid);
  const rows = ['I0_A,dD_parallel_px,dD_antiparallel_px,dD_antiparallel_exp_px,dD_antiparallel_pureCQD_px,Dc0,w_ret'];
  r.pts.forEach(q => { const w = M.W.lin(q.D0, s.lin.p);
    rows.push([q.I0.toFixed(2), fx(M.dPar(r, q)), fx(M.dAnti(r, q, w)), fx(M.dAnti(r, q, M.W.exp(q.D0, s.exp.p))),
      fx(M.dAnti(r, q, 0)), q.D0.toFixed(5), w.toFixed(5)].join(',')); });
  fs.writeFileSync(`${__dirname}/fit_curves_f${s.f}_20260706.csv`, rows.join('\n') + '\n');
  // values at the data currents
  const rd = R(s.D1, s.f, 'baseline');
  s.atData = rd.pts.map(q => ({ I0: q.I0, par: M.dPar(rd, q), anti: M.dAnti(rd, q, M.W.lin(q.D0, s.lin.p)), strict: M.dAnti(rd, q, 0) }));
}
const s7 = out.strategy2.find(s => s.f === 7 && s.dir === 'baseline');
fs.writeFileSync(`${__dirname}/fit_params_f7_20260706.csv`, ['quantity,value,note',
  `f,7,SG0 field divisor (fixed input; not fitted)`,
  `D_c1,${s7.D1.toFixed(5)},fitted to the parallel set only (pure CQD)`,
  `k_i,${s7.ki.toExponential(4)},profile Delta-chi2=1 interval ${s7.kiLo.toExponential(3)} to ${s7.kiHi.toExponential(3)}`,
  `DIR,${s7.DIR.toFixed(4)},solved from the 13-px baseline`,
  `chi2_parallel,${s7.chi2Par.toFixed(3)},${N} points; per-point sigmas from the data file`,
  `kappa,${s7.lin.p.toFixed(4)},antiparallel linear weight w = kappa*Dc0 (k_i fixed); +- ${s7.lin.err.toFixed(4)}`,
  `chi2_antiparallel_linear,${s7.lin.chi2.toFixed(3)},${N} points`,
  `RLC,${s7.exp.p.toFixed(4)},antiparallel exponential weight w = 1-exp(-RLC*Dc0) (k_i fixed); +- ${s7.exp.err.toFixed(4)}`,
  `chi2_antiparallel_exp,${s7.exp.chi2.toFixed(3)},${N} points`,
  `chi2_antiparallel_pureCQD,${s7.chi2AntiStrict.toFixed(2)},w = 0 (strict rule)`].join('\n') + '\n');
fs.writeFileSync(`${__dirname}/results.json`, JSON.stringify(out, null, 1));
console.log('wrote results.json, fit_curves_f1_20260706.csv, fit_curves_f7_20260706.csv, fit_params_f7_20260706.csv');
