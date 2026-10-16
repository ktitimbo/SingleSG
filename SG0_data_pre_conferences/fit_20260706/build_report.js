// Assembles cqd_sg0_fit_20260706.tex: front matter + model derivation (taken from ../fig12/fig12_analysis.tex)
// + results for the 2026-07-06 data. Then compile with pdflatex (twice).
const fs = require('fs');
const src = fs.readFileSync(__dirname + '/../fig12/fig12_analysis.tex', 'utf8').split(/\r?\n/);
const find = s => { const i = src.findIndex(l => l.startsWith(s)); if (i < 0) throw new Error('not found: ' + s); return i; };
// derivation: from "\section{Model derivation}" up to "\section{The fit of Figure 12"
const deriv = src.slice(find('\\section{Model derivation}'), find('\\section{The fit of Figure 12'));
// leading-order picture: from its subsection to the next subsection
const lo0 = find('\\subsection{Leading-order picture'), lo1 = find('\\subsection{The antiparallel curve}');
const leading = src.slice(lo0, lo1).map(l => l.startsWith('\\subsection{Leading-order picture')
  ? '\\subsection{Leading-order picture of the parallel curve}\n\\label{sec:leading}' : l)
  .map(l => l.replace('This predicts about $1.5$~px\\,A$^{-1}$, against the measured $1.32$~px\\,A$^{-1}$',
                      'With the $f=1$ value $\\alpha_0=0.057$~A$^{-1}$ this predicts about $1.5$~px\\,A$^{-1}$, against about $1.3$~px\\,A$^{-1}$ in the data'));
const out = [fs.readFileSync(__dirname + '/partA_front.tex', 'utf8'), ...deriv, ...leading,
  fs.readFileSync(__dirname + '/partC_results.tex', 'utf8')
    .replace('\\input{table_points.tex}\n', fs.readFileSync(__dirname + '/table_points.tex', 'utf8'))].join('\n');
fs.writeFileSync(__dirname + '/cqd_sg0_fit_20260706.tex', out);
console.log('wrote cqd_sg0_fit_20260706.tex');
