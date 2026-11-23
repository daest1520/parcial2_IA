// Utilidades para un informe con formato IEEE (Letter, Times New Roman, 2 columnas)
const fs = require("fs");
const d = require("docx");
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
        BorderStyle, ShadingType, VerticalAlign, SectionType, Footer, PageNumber, HeightRule } = d;

const FONT = "Times New Roman";
const COL_W = 5040;            // ancho de columna en DXA (3.5 in)
const FULL_W = 10800;          // ancho de texto a una columna (7.5 in)

// Texto con marcado ligero: **negrita**, ¦cursiva¦, ^sup^, ~sub~
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|¦[^¦]+¦|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("¦")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), subScript: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opt = {}) => new Paragraph({
  children: runs(text, { size: opt.size || 20 }),
  alignment: opt.align || AlignmentType.JUSTIFIED,
  indent: opt.noIndent ? undefined : { firstLine: 202 },
  spacing: { after: opt.after ?? 0, before: opt.before ?? 0, line: 228 },
  keepNext: opt.keepNext || false,
});

const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
let h1n = 0, h2n = 0;
const H1 = (text) => { h2n = 0; const n = ROMAN[h1n++];
  return new Paragraph({ children: [new TextRun({ text: `${n}. ${text}`, font: FONT, size: 20, smallCaps: true })],
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true }); };
const H2 = (text) => { const n = String.fromCharCode(65 + h2n++);
  return new Paragraph({ children: [new TextRun({ text: `${n}. ${text}`, font: FONT, size: 20, italics: true })],
    spacing: { before: 120, after: 60 }, keepNext: true }); };

const EQ = (text, num) => new Paragraph({
  children: [...runs(text, { size: 20, italics: false }), ...(num ? [new TextRun({ text: `\t(${num})`, font: FONT, size: 20 })] : [])],
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  tabStops: num ? [{ type: d.TabStopType.RIGHT, position: COL_W }] : undefined,
});

const BUL = (text) => new Paragraph({
  children: runs(text, { size: 20 }), numbering: { reference: "bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 20, line: 228 },
});

let figN = 0, tabN = 0;
function FIG(path, widthIn, caption, full = false) {
  const buf = fs.readFileSync(path);
  // dimensiones del PNG/JPG para mantener la proporción
  let w, h;
  if (buf[0] === 0x89) { w = buf.readUInt32BE(16); h = buf.readUInt32BE(20); }
  else { let i = 2; while (i < buf.length) { if (buf[i] !== 0xFF) { i++; continue; } const mk = buf[i + 1];
    if (mk >= 0xC0 && mk <= 0xC3) { h = buf.readUInt16BE(i + 5); w = buf.readUInt16BE(i + 7); break; }
    i += 2 + buf.readUInt16BE(i + 2); } }
  const pxW = widthIn * 96, pxH = pxW * h / w;
  figN++;
  const els = [
    new Paragraph({ children: [new ImageRun({ type: path.endsWith(".png") ? "png" : "jpg", data: buf,
      transformation: { width: pxW, height: pxH } })], alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: `Fig. ${figN}. `, font: FONT, size: 16 }), ...runs(caption, { size: 16 })],
      alignment: full ? AlignmentType.CENTER : AlignmentType.JUSTIFIED, spacing: { after: 120 } }),
  ];
  return full ? [{ full: els }] : els;
}

const B = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TAB(caption, headers, rows, widths, opt = {}) {
  tabN++;
  const total = widths.reduce((a, b) => a + b, 0);
  const sz = opt.size || 15;
  const mk = (txt, i, head, last) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: head ? B : NB, bottom: head || last ? B : NB, left: NB, right: NB },
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    verticalAlign: VerticalAlign.CENTER,
    shading: opt.shade && opt.shade(rows.indexOf(txt), i) ? { fill: "EDEDED", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ children: runs(String(txt), { size: sz, bold: head }),
      alignment: i === 0 && !opt.centerFirst ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  const trs = [new TableRow({ tableHeader: true, children: headers.map((h, i) => mk(h, i, true, false)) })];
  rows.forEach((r, k) => trs.push(new TableRow({ cantSplit: true,
    children: r.map((c, i) => mk(c, i, false, k === rows.length - 1)) })));
  return [
    new Paragraph({ children: [new TextRun({ text: `${process.env.REPORT_EN ? "TABLE" : "TABLA"} ${ROMAN[tabN - 1] || tabN}`, font: FONT, size: 16 })],
      alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: caption.toUpperCase(), font: FONT, size: 16, smallCaps: false })],
      alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: trs, alignment: AlignmentType.CENTER }),
    new Paragraph({ children: [], spacing: { after: 100 } }),
  ];
}

module.exports = { d, FONT, COL_W, FULL_W, runs, P, H1, H2, EQ, BUL, FIG, TAB, figRef: () => figN, tabRef: () => tabN };
