/* Camera treatments from GLaDOS-webui, using real frames and the existing eye coupling. */
(function(root) {
  "use strict";
  const clamp01=x=>Math.max(0,Math.min(1,x));
  const hex=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
  const BAYER = (() => {
    let m = [[0, 2], [3, 1]];
    while (m.length < 8) {
      const n = m.length, o = [];
      for (let i = 0; i < 2 * n; i++) { o.push([]); for (let j = 0; j < 2 * n; j++) o[i].push(4 * m[i % n][j % n] + [[0, 2], [3, 1]][(i / n) | 0][(j / n) | 0]); }
      m = o;
    }
    return m.map((r) => r.map((v) => (v + 0.5) / 64));
  })();
  const GB = [[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]];
  const RAMP = " .,:;-=+*x%#@";

  class ThemedVision extends root.GladosVision.Vision {
    constructor(canvas) { super(canvas); this.palette = "thermal"; }
    setTheme(t) {
      this.palette = t.optic; this.mode = t.optic === "whitehot" ? "thermal" : t.optic;
      this.ink = hex(t.opticInk); this.bg = hex(t.opticBg); this.bgHex = t.opticBg; this.tintKey = "";
    }
    makeLut(stops) {
      if (this.palette === "whitehot") stops = [[0, this.bg], [0.6, [150, 152, 156]], [1, [255, 255, 255]]];
      return super.makeLut(stops);
    }
    /* sample a camera-space array at viewport-relative coordinates */
    at(arr, fx, fy) {
      const { SW, SH } = this, P = this.viewport;
      return arr[Math.min(SH - 1, ((P.y + fy * P.height) * SH) | 0) * SW + Math.min(SW - 1, ((P.x + fx * P.width) * SW) | 0)];
    }
    draw_ascii(W, H) {
      const g = this.g, cols = Math.max(48, Math.round(W / 9)), cw = W / cols, fs = cw / 0.6, rows = Math.ceil(H / fs), Rn = RAMP.length - 1;
      g.font = `500 ${fs.toFixed(1)}px "IBM Plex Mono", monospace`; g.textBaseline = "top";
      g.fillStyle = `rgb(${this.ink})`; g.shadowColor = `rgba(${this.ink},.5)`; g.shadowBlur = fs * 0.35;
      const hot = [];
      for (let y = 0; y < rows; y++) {
        let line = "";
        for (let x = 0; x < cols; x++) {
          const fx = (x + 0.5) / cols, fy = (y + 0.5) / rows, ch = RAMP[Math.round(clamp01(this.at(this.lum, fx, fy)) * Rn)];
          if (this.at(this.motion, fx, fy) > 0.25 && ch !== " ") { hot.push(x, y, ch); line += " "; } else line += ch;
        }
        g.fillText(line, 0, y * fs);
      }
      // whatever moves catches her eye, in her centre colour
      g.fillStyle = `rgb(${this.tint})`; g.shadowColor = `rgba(${this.tint},.9)`; g.shadowBlur = fs * 0.8;
      for (let i = 0; i < hot.length; i += 3) g.fillText(hot[i + 2], hot[i] * cw, hot[i + 1] * fs);
      g.shadowBlur = 0;
    }
    draw_dither(W, H) {
      const { SW, SH, lum, motion } = this, px = this.img.data;
      for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
        const i = y * SW + x, j = i * 4, v = clamp01(lum[i]) * 3;
        let k = Math.min(3, (v | 0) + (v % 1 > BAYER[y & 7][x & 7] ? 1 : 0));
        if (motion[i] > 0.25 && k > 0) k = 3;
        px[j] = GB[k][0]; px[j + 1] = GB[k][1]; px[j + 2] = GB[k][2]; px[j + 3] = 255;
      }
      this.blit(W, H, false, false);
    }
  }

  root.GladosVision.ThemedVision=ThemedVision;
})(window);
