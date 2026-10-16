// Writes table_points.tex (data and fit values at the measured currents) from results.json.
const fs = require('fs');
const r = require('./results.json'), d = r.data;
const s1 = r.strategy2.find(s => s.f === 1 && s.dir === 'baseline');
const s7 = r.strategy2.find(s => s.f === 7 && s.dir === 'baseline');
const n = v => { const t = (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3); return t.startsWith('-') ? '$-$' + t.slice(1) : t; };
const pm = (v, s) => `$${(Math.abs(v) < 5e-4 ? 0 : v).toFixed(3)}\\pm${s.toFixed(2)}$`;
const rows = d.I.map((I, i) => [I.toFixed(3), pm(d.par[i], d.sPar[i]), n(s7.atData[i].par), n(s1.atData[i].par),
  pm(d.anti[i], d.sAnti[i]), n(s7.atData[i].anti), n(s7.atData[i].strict)].join(' & ') + ' \\\\');
fs.writeFileSync(__dirname + '/table_points.tex', rows.join('\n') + '\n');
