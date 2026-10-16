// Double-stage (SG0 -> SG1) CQD model of the source note, Secs. 8-17, with data-driven fitting.
// Same physics as fig12/fig12_scaled.js, but the data (currents, values, per-point sigmas) are passed in.
const fs = require('fs');

const BASE = 13, I_REF = 1.03869, K_PIX = 6.47;            // px, A (= B1 l1 / (beta0 l0)), px
const VA = 600.5, GE = 1.76086e11, BL1 = 0.018931 * 0.070;  // m/s, rad/(s T), T m

// ---------- data ----------
function loadData(file) {
  const rows = fs.readFileSync(file, 'utf8').trim().split(/\r?\n/).slice(1).map(r => r.split(',').map(Number));
  return {
    I: rows.map(r => r[0]),
    par: rows.map(r => r[1]), sPar: rows.map(r => r[2]),
    anti: rows.map(r => r[3]), sAnti: rows.map(r => r[4]),
  };
}

// ---------- quadrature ----------
function gl(n) {
  const x = [], w = [];
  for (let i = 1; i <= n; i++) {
    let z = Math.cos(Math.PI * (i - 0.25) / (n + 0.5)), pp;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2*j-1)*z*p2 - (j-1)*p3) / j; }
      pp = n * (z*p1 - p2) / (z*z - 1);
      const dz = p1 / pp; z -= dz; if (Math.abs(dz) < 1e-15) break;
    }
    x.push(z); w.push(2 / ((1 - z*z) * pp * pp));
  }
  return [x, w];
}
const [GX, GW] = gl(48), [HX, HW] = gl(24);
const integ = (f, a, b, X = GX, W = GW) => { let s = 0; const h = (b-a)/2, m = (b+a)/2;
  for (let i = 0; i < X.length; i++) s += W[i] * f(m + h*X[i]); return s*h; };
const integSplit = (f, pts) => { let s = 0; for (let i = 0; i+1 < pts.length; i++) s += integ(f, pts[i], pts[i+1]); return s; };

// ---------- model ----------
const T = (c, q, D) => { const t = Math.tanh(D); return (c + q*t) / (1 + q*c*t); };           // Eq. (145)
function ztil(cin, D1, rho) {                                                                  // Eqs. (204), (211)
  const C = Math.atanh(Math.max(-1+1e-15, Math.min(1-1e-15, cin)));
  return integ(l => (D1 + rho - l) * Math.tanh(C + l), 0, D1, HX, HW) / (rho*D1 + D1*D1/2);
}
const Zpar = (D0, D1, rho) => 0.5 * integ(c => (1+c) * ztil(T(c, 1, D0), D1, rho), -1, 1);       // (217)
const Zret = (D0, D1, rho) => 0.5 * integ(c => (1+c) * ztil(-T(c, 1, D0), D1, rho), -1, 1);      // (251)
function Zgeom(D0, D1, rho) {                                                                     // (218)-(221)
  const k = Math.tanh(D0/2);
  return 0.5 * integSplit(c => { const Tp = T(c,1,D0), Tm = T(c,-1,D0);
    const Lp = 1 + Math.min(c, -Tp), Lm = Math.max(0, -Tm - c);
    return Lp*ztil(-Tp, D1, rho) + (Lm > 0 ? Lm*ztil(-Tm, D1, rho) : 0); }, [-1, -k, k, 1]);
}
function solveDIR(D1) {                                                                           // (249)
  const g = DIR => K_PIX*(DIR + 0.5)*Zpar(0, D1, DIR*D1) - BASE;
  let a = 0.01, b = 200; for (let i = 0; i < 80; i++) { const m = (a+b)/2; (g(a)*g(m) <= 0) ? b = m : a = m; }
  return (a+b)/2;
}
const ki = D1 => D1 * VA / (GE * BL1);

// Geometry for a given D_c1: returns DIR, rho, Z0
function geometry(D1, dirMode) {
  const DIR = dirMode === 'baseline' ? solveDIR(D1) : dirMode;
  return { DIR, rho: DIR * D1, Z0: Zpar(0, D1, DIR * D1) };
}
// Responses at currents I (array) for SG0 divisor f; w(D0) is the retained-label weight function
function responses(D1, f, dirMode, I) {
  const g = geometry(D1, dirMode);
  const pts = I.map(I0 => { const D0 = D1 * I0 / (I_REF * f);
    return { I0, D0, zp: Zpar(D0, D1, g.rho), zg: Zgeom(D0, D1, g.rho), zr: Zret(D0, D1, g.rho) }; });
  return { ...g, pts };
}
const dPar = (g, q) => BASE * (q.zp / g.Z0 - 1);                                                 // (241)
const dAnti = (g, q, w) => BASE * (((1 - w) * q.zg + w * q.zr) / g.Z0 - 1);                      // (253), (241)
const W = {
  none: () => 0,
  exp: (D0, p) => 1 - Math.exp(-p * D0),
  lin: (D0, p) => Math.min(1, p * D0),
};

// ---------- small optimizers ----------
const ssq = r => r.reduce((s, v) => s + v*v, 0);
function golden(fn, a, b, n = 70) {
  const r = (Math.sqrt(5) - 1) / 2; let c = b - r*(b-a), d = a + r*(b-a), fc = fn(c), fd = fn(d);
  for (let i = 0; i < n; i++) { if (fc < fd) { b = d; d = c; fd = fc; c = b - r*(b-a); fc = fn(c); }
                                else { a = c; c = d; fc = fd; d = a + r*(b-a); fd = fn(d); } }
  return fc < fd ? [c, fc] : [d, fd];
}
// minimize over log(D1) in [lo, hi] with golden section after a coarse scan
function minimizeD1(chi, lo = 1e-3, hi = 5) {
  const N = 40, xs = [...Array(N + 1)].map((_, i) => Math.log(lo) + i * (Math.log(hi) - Math.log(lo)) / N);
  let ib = 0, best = Infinity; xs.forEach((x, i) => { const c = chi(Math.exp(x)); if (c < best) { best = c; ib = i; } });
  const a = xs[Math.max(0, ib - 1)], b = xs[Math.min(N, ib + 1)];
  const [x, c] = golden(x => chi(Math.exp(x)), a, b, 50);
  return [Math.exp(x), c];
}
// profile Delta-chi2 = 1 interval of a 1-D chi2(D1) around its minimum
function profileInterval(chi, D1, c0) {
  const side = up => { let a = D1, b = up ? D1 * 4 : D1 / 4;
    if (chi(b) - c0 < 1) return NaN;
    for (let i = 0; i < 60; i++) { const m = Math.sqrt(a * b); (chi(m) - c0 < 1) ? a = m : b = m; } return Math.sqrt(a*b); };
  return [side(false), side(true)];
}

module.exports = { loadData, BASE, I_REF, K_PIX, ki, geometry, responses, dPar, dAnti, W, ssq, golden, minimizeD1,
  profileInterval, Zpar, Zgeom, Zret, solveDIR };
