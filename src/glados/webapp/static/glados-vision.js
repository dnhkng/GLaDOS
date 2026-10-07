// Adapted from the user-provided glados-optic-feed.html: Original (thermal) mode.
// GLaDOS vision: draws a camera (or any <img>/<video>/<canvas>) the way GLaDOS sees it.
// No dependencies. Browser only (window.GladosVision).
//
//   const v = new GladosVision.Vision(canvas);
//   v.setSource(videoEl);           // webcam, clip, or a canvas
//   v.setMode('thermal');           // the design's Original treatment
//   each frame: v.setEye(animatorOutput); const f = v.frame(dt);
//   v.setSubject({x, y, kind}); // viewer coordinates shared with the eye
//
// The feed is coupled to her eye: lids shutter the frame (blinks included), the blade opening
// sets the vignette, gaze pans the view, and the centre colour tints whatever moves.
(function (root) {
  const MODES = ['thermal'];
  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const mk = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  const approach = (cur, tgt, rate, dt) => cur + (tgt - cur) * (1 - Math.exp(-rate * dt));

  class Vision {
    constructor(canvas, opts = {}) {
      this.cv = canvas;
      this.g = canvas.getContext('2d');
      this.ink = hexToRgb(opts.ink || '#E8ECEF');
      this.bg = hexToRgb(opts.bg || '#0B0E12');
      this.bgHex = opts.bg || '#0B0E12';
      this.mode = 'thermal';
      this.mirror = true;      // selfie view; gaze output is always in the viewer's perspective
      this.reducedMotion = false;
      this.couple = true;      // lids, blade opening and gaze shape the feed
      this.SW = 256; this.SH = 144;
      this.s = mk(this.SW, this.SH);
      this.sg = this.s.getContext('2d', { willReadFrequently: true });
      const N = this.SW * this.SH;
      this.raw = new Float32Array(N); this.prev = new Float32Array(N);
      this.lum = new Float32Array(N); this.motion = new Float32Array(N); this.heat = new Float32Array(N);
      this.out = mk(this.SW, this.SH); this.og = this.out.getContext('2d');
      this.img = this.og.createImageData(this.SW, this.SH);
      this.lo = 0.05; this.hi = 0.95;
      this.focus = { x: 0.5, y: 0.45, e: 0, tx: 0.5, ty: 0.45 };
      this.tint = [242, 194, 101]; this.tintKey = '';
      this.lut = null;
      this.eye = null;
      this.t = 0; this.scan = 0; this.glitch = 0; this.flash = 0;
      this.hasFrame = false;
      this.motionEnergy = 0;
      this.motionTargets = [];
      this.subject = null;
      this.pan = {x:0.5,y:0.5};
      this.viewport = {x:0,y:0,width:1,height:1};
    }
    setSource(src) { this.src = src; this.hasFrame = false; this.motionEnergy = 0; this.motionTargets=[]; this.prev.fill(0); this.heat.fill(0); this.motion.fill(0); }
    setMode(m) { if (MODES.includes(m)) this.mode = m; }
    // rig Animator output; drives tint, lids, vignette, gaze pan and glitch
    setEye(p) { this.eye = p; if (p && p.color) this.tint = hexToRgb(p.color); }
    // Viewer coordinates, from the same attention target as the eye.
    setSubject(subject) { this.subject = subject; }

    resize() {
      const r = this.cv.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(160, Math.round(r.width * dpr)), h = Math.max(90, Math.round(r.height * dpr));
      if (this.cv.width !== w || this.cv.height !== h) { this.cv.width = w; this.cv.height = h; }
      this.dpr = dpr;
    }

    // ---- analysis: luminance with auto levels, motion, heat, focus of attention ----
    analyse(dt) {
      const src = this.src;
      const sw = src && (src.videoWidth || src.naturalWidth || src.width), sh = src && (src.videoHeight || src.naturalHeight || src.height);
      if (!sw || !sh || (src.readyState !== undefined && src.readyState < 2)) { this.motionTargets=[]; return false; }
      // Sample the full image, so panning can reach faces at any edge. Motion
      // analysis remains in camera space and does not mistake our own pan for motion.
      const sampleHeight = Math.max(1, Math.min(512, Math.round(this.SW * sh / sw)));
      if (sampleHeight !== this.SH) {
        this.SH = sampleHeight;this.s.height=sampleHeight;this.out.height=sampleHeight;
        const N=this.SW*this.SH;
        for (const key of ['raw','prev','lum','motion','heat']) this[key]=new Float32Array(N);
        this.img=this.og.createImageData(this.SW,this.SH);this.hasFrame=false;
      }
      const { SW, SH, sg } = this;
      sg.save();
      if (this.mirror) { sg.translate(SW, 0); sg.scale(-1, 1); }
      sg.drawImage(src, 0, 0, SW, SH);
      sg.restore();
      const d = sg.getImageData(0, 0, SW, SH).data;
      const N = SW * SH, raw = this.raw, prev = this.prev, mo = this.motion, heat = this.heat;
      const hist = new Uint32Array(64);
      for (let i = 0, j = 0; i < N; i++, j += 4) {
        const l = (0.2126 * d[j] + 0.7152 * d[j + 1] + 0.0722 * d[j + 2]) / 255;
        raw[i] = l; hist[(l * 63.99) | 0]++;
      }
      // 2nd and 98th percentiles, smoothed, so dim webcams still read
      let acc = 0, lo = 0, hi = 1;
      for (let b = 0; b < 64; b++) { acc += hist[b]; if (acc > N * 0.02) { lo = b / 64; break; } }
      acc = 0;
      for (let b = 63; b >= 0; b--) { acc += hist[b]; if (acc > N * 0.02) { hi = (b + 1) / 64; break; } }
      if (hi - lo < 0.15) hi = Math.min(1, lo + 0.15);
      this.lo = approach(this.lo, lo, 3, dt); this.hi = approach(this.hi, hi, 3, dt);
      const inv = 1 / (this.hi - this.lo);
      // Motion heat loses 95% of its intensity in about 0.6 seconds.
      const decay = Math.exp(-dt * 4), hdecay = Math.exp(-dt * Math.log(20) / 0.6);
      const first = !this.hasFrame;
      let motionTotal = 0;
      for (let y = 0, i = 0; y < SH; y++) {
        for (let x = 0; x < SW; x++, i++) {
          this.lum[i] = clamp01((raw[i] - this.lo) * inv);
          let m = first ? 0 : Math.abs(raw[i] - prev[i]);
          m = m > 0.05 ? Math.min(1, (m - 0.05) * 6) : 0;
          mo[i] = Math.max(mo[i] * decay, m);
          motionTotal += mo[i];
          heat[i] = Math.min(1, heat[i] * hdecay + m * dt * 6);
          prev[i] = raw[i];
        }
      }
      this.hasFrame = true;
      // Match the reference HUD's 0–1 motion scale, independently of face presence.
      this.motionEnergy = clamp01(motionTotal / N * 40);
      this.motionTargets = motionTotal/N>=0.0002?motionRegions(mo,SW,SH,this.mirror):[];
      const F = this.focus;
      const subject = this.subject;
      F.e = approach(F.e, subject ? 1 : 0, subject ? 12 : 8, dt);
      if (subject) {
        // Full-source viewer coordinates; image and reticle share the same crop.
        const viewX = this.mirror ? subject.x : -subject.x;
        F.tx = clamp01(0.5 + viewX * 0.5);
        F.ty = clamp01(0.5 + subject.y * 0.5);
        F.x = F.tx;F.y = F.ty;
      }
      return true;
    }

    frame(dt) {
      dt = Math.min(dt, 0.1);
      this.t += this.reducedMotion ? 0 : dt;
      this.resize();
      const ok = this.analyse(dt);
      const { g, cv } = this, W = cv.width, H = cv.height;
      const e = this.eye || {};
      const awake = e.awake == null ? 1 : clamp01(e.awake);
      g.save();
      g.fillStyle = this.bgHex; g.fillRect(0, 0, W, H);
      if (ok) {
        // Retain 80% of the image in the limiting dimension, without stretching.
        const ratio=(this.SW/this.SH)/(W/H);
        const width=0.8/Math.max(1,ratio),height=0.8*Math.min(1,ratio);
        const scanning = !this.subject;
        const tx=scanning ? 0.5+(1-width)*0.5*Math.sin(this.t*0.31) :
          0.5+(e.lookX||0)*0.5+Math.sin(this.t*0.23)*0.008;
        const ty=scanning ? 0.5+(1-height)*0.5*Math.sin(this.t*0.21+0.7) :
          0.5+(e.lookY||0)*0.5+Math.sin(this.t*0.17)*0.006;
        const bound=(v,size)=>Math.max(size/2,Math.min(1-size/2,v));
        this.pan.x=approach(this.pan.x,bound(tx,width),scanning?0.8:5,dt);
        this.pan.y=approach(this.pan.y,bound(ty,height),scanning?0.8:5,dt);
        this.viewport={x:bound(this.pan.x,width)-width/2,y:bound(this.pan.y,height)-height/2,width,height};
        g.save();
        g.globalAlpha = 0.15 + 0.85 * awake;
        this['draw_' + this.mode](W, H);
        g.globalAlpha = 1;
        this.drawReticle(W, H, awake);
        g.restore();
        if (!this.reducedMotion) this.drawGlitch(W, H, e);
      }
      this.drawScan(W, H, e, this.reducedMotion ? 0 : dt);
      if (this.couple) this.drawLids(W, H, e);
      this.drawFrame(W, H);
      g.restore();
      const F = this.focus;
      const gx = (this.mirror ? F.x : 1 - F.x) * 2 - 1;
      return { ok, gx, gy: F.y * 2 - 1, energy: F.e, motion: ok ? this.motionEnergy : 0 };
    }

    // ---- treatments ----
    draw_thermal(W, H) {
      const lut = this.tintLut();
      const { SW, SH, lum, heat } = this, px = this.img.data;
      for (let i = 0, N = SW * SH; i < N; i++) {
        const v = clamp01(lum[i] * 0.7 + heat[i] * 0.6);
        const k = ((v * 255) | 0) * 3, j = i * 4;
        px[j] = lut[k]; px[j + 1] = lut[k + 1]; px[j + 2] = lut[k + 2]; px[j + 3] = 255;
      }
      this.blit(W, H, true, false);
      // isotherm lines on top keep it line-art
      const g = this.g, cols = 96, cs = W / cols, rows = Math.ceil(H / cs);
      const val = (x, y) => { const V=this.viewport; const i = Math.min(SH - 1, ((V.y+(y+0.5)/rows*V.height)*SH) | 0)*SW + Math.min(SW - 1, ((V.x+(x+0.5)/cols*V.width)*SW) | 0); return ((lum[i] * 0.7 + heat[i] * 0.6) * 7) | 0; };
      g.strokeStyle = `rgba(${this.ink},.22)`; g.lineWidth = Math.max(1, W / 1000);
      g.beginPath();
      for (let y = 0; y < rows - 1; y++) for (let x = 0; x < cols - 1; x++) {
        const a = val(x, y);
        if (a !== val(x + 1, y)) { g.moveTo((x + 1) * cs, y * cs); g.lineTo((x + 1) * cs, (y + 1) * cs); }
        if (a !== val(x, y + 1)) { g.moveTo(x * cs, (y + 1) * cs); g.lineTo((x + 1) * cs, (y + 1) * cs); }
      }
      g.stroke();
    }
    makeLut(stops) {
      stops = stops || [[0, this.bg], [0.3, [22, 38, 66]], [0.55, [70, 84, 150]], [0.8, this.tint], [1, [255, 255, 255]]];
      const lut = new Uint8ClampedArray(256 * 3);
      for (let i = 0; i < 256; i++) {
        const v = i / 255;
        let s = 0; while (s < stops.length - 2 && v > stops[s + 1][0]) s++;
        const [a, ca] = stops[s], [b, cb] = stops[s + 1], f = clamp01((v - a) / (b - a));
        for (let c = 0; c < 3; c++) lut[i * 3 + c] = ca[c] + (cb[c] - ca[c]) * f;
      }
      return lut;
    }

    tintLut() {
      const key = this.tint.map((c) => c >> 3).join();
      if (key !== this.tintKey) { this.tintKey = key; this.lut = this.makeLut(); }
      return this.lut;
    }
    // put this.img on the canvas, scaled; optional glow pass
    blit(W, H, smooth, glow) {
      const g = this.g;
      this.og.putImageData(this.img, 0, 0);
      g.imageSmoothingEnabled = smooth;
      if (glow && 'filter' in g) {
        const ga = g.globalAlpha;
        g.filter = `blur(${(W / 220).toFixed(1)}px)`; g.globalAlpha = ga * 0.45;
        g.globalCompositeOperation = 'lighter';
        const V=this.viewport;
      g.drawImage(this.out,V.x*this.SW,V.y*this.SH,V.width*this.SW,V.height*this.SH,0,0,W,H);
        g.filter = 'none'; g.globalAlpha = ga; g.globalCompositeOperation = 'source-over';
      }
      const V=this.viewport;
      g.drawImage(this.out,V.x*this.SW,V.y*this.SH,V.width*this.SW,V.height*this.SH,0,0,W,H);
      g.imageSmoothingEnabled = true;
    }

    // ---- overlays ----
    drawReticle(W, H, awake) {
      const F = this.focus, a = F.e * awake;
      if (a < 0.03) return;
      const V=this.viewport;
      const g = this.g, x = (F.x-V.x)/V.width*W, y = (F.y-V.y)/V.height*H, r = H * 0.09, u = H / 400;
      g.save();
      g.translate(x, y); g.rotate(this.t * 0.4);
      g.strokeStyle = `rgba(${this.tint},${(0.85 * a).toFixed(3)})`; g.lineWidth = 1.5 * u;
      for (let k = 0; k < 4; k++) { g.beginPath(); g.arc(0, 0, r, k * Math.PI / 2 + 0.2, (k + 1) * Math.PI / 2 - 0.2); g.stroke(); }
      g.beginPath(); g.arc(0, 0, r * 0.35, 0, Math.PI * 2); g.stroke();
      g.rotate(-this.t * 0.4);
      g.fillStyle = `rgba(${this.tint},${(0.9 * a).toFixed(3)})`;
      g.font = `500 ${(11 * u).toFixed(1)}px "IBM Plex Mono", ui-monospace, monospace`;
      g.textBaseline = 'middle';
      g.fillText(this.subject?.kind==='motion'?'MOTION':'SUBJECT', r + 8 * u, -r * 0.6);
      g.restore();
    }
    drawGlitch(W, H, e) {
      // angry and lingering anger tear the frame; any mood gets a rare tear
      const angry = e.color ? Math.max(0, (hexToRgb(e.color)[0] - hexToRgb(e.color)[2]) / 255 - 0.35) : 0;
      if (Math.random() < 0.004) this.flash = 1;
      this.flash = Math.max(0, this.flash - 0.08);
      const amt = Math.min(1, angry * 1.6 + this.flash);
      if (amt < 0.05 || Math.random() > 0.25 + amt * 0.6) return;
      const g = this.g, n = 1 + ((Math.random() * 4 * amt) | 0);
      for (let k = 0; k < n; k++) {
        const y = Math.random() * H, h = H * (0.005 + Math.random() * 0.04), dx = (Math.random() - 0.5) * W * 0.06 * amt;
        g.drawImage(this.cv, 0, y, W, h, dx, y, W, h);
      }
    }
    drawScan(W, H, e, dt) {
      const g = this.g, proc = e.bladeRot ? clamp01((e.bladeRot - 0.4) / 0.6) : 0;
      this.scan = (this.scan + (0.08 + proc * 0.5) * dt) % 1.2;
      const y = this.scan * H, band = H * 0.12;
      const gr = g.createLinearGradient(0, y - band, 0, y);
      const c = proc > 0.2 ? this.tint : this.ink;
      gr.addColorStop(0, `rgba(${c},0)`); gr.addColorStop(1, `rgba(${c},${0.05 + proc * 0.1})`);
      g.fillStyle = gr; g.fillRect(0, y - band, W, band);
      // fine scanlines
      g.fillStyle = `rgba(${this.bg},.14)`;
      const step = Math.max(2, Math.round(H / 240));
      for (let yy = 0; yy < H; yy += step * 2) g.fillRect(0, yy, W, step);
    }
    drawLids(W, H, e) {
      const g = this.g;
      // Aperture changes shade the edges gently without obscuring the scene.
      const a = e.a == null ? 22 : e.a;
      const r0 = H * (0.65 + clamp01(a / 44) * 0.1), r1 = r0 + H * 0.65;
      const vg = g.createRadialGradient(W / 2, H / 2, r0, W / 2, H / 2, r1);
      vg.addColorStop(0, `rgba(${this.bg},0)`); vg.addColorStop(1, `rgba(${this.bg},.45)`);
      g.fillStyle = vg; g.fillRect(0, 0, W, H);
      // Expression lids only skim the top/bottom. The explicit blink signal
      // closes the full feed, independent of how narrow the eye already is.
      const shutter = Math.max(clamp01(e.blink || 0), 1 - clamp01(e.awake == null ? 1 : e.awake));
      const narrowTop = Math.min(0.08, clamp01(((e.lidTop == null ? -90 : e.lidTop) + 90) / 180) * 0.12) * H;
      const narrowBot = Math.min(0.08, clamp01((90 - (e.lidBot == null ? 90 : e.lidBot)) / 180) * 0.12) * H;
      const top = narrowTop * (1 - shutter) + H * 0.5 * shutter;
      const bot = narrowBot * (1 - shutter) + H * 0.5 * shutter;
      const slant = angle => Math.max(-H * 0.015, Math.min(H * 0.015,
        Math.tan((angle * Math.PI) / 180) * W * 0.04)) * (1 - shutter);
      const st = slant(e.lidTopAng || 0), sb = slant(e.lidBotAng || 0);
      g.fillStyle = this.bgHex;
      g.strokeStyle = `rgba(${this.ink},.55)`; g.lineWidth = Math.max(1, H / 360);
      // lid edges; where they would cross, they meet instead
      let tl = top + st, tr = top - st, bl = H - bot + sb, br = H - bot - sb;
      if (bl < tl) tl = bl = (tl + bl) / 2;
      if (br < tr) tr = br = (tr + br) / 2;
      if (top > 1) {
        g.beginPath(); g.moveTo(0, 0); g.lineTo(W, 0); g.lineTo(W, tr); g.lineTo(0, tl); g.closePath(); g.fill();
        g.beginPath(); g.moveTo(0, tl); g.lineTo(W, tr); g.stroke();
      }
      if (bot > 1) {
        g.beginPath(); g.moveTo(0, H); g.lineTo(W, H); g.lineTo(W, br); g.lineTo(0, bl); g.closePath(); g.fill();
        g.beginPath(); g.moveTo(0, bl); g.lineTo(W, br); g.stroke();
      }
    }
    drawFrame(W, H) {
      const g = this.g, m = H * 0.04, L = H * 0.07;
      g.strokeStyle = `rgba(${this.ink},.6)`; g.lineWidth = Math.max(1, H / 300);
      g.beginPath();
      for (const [x, y, sx, sy] of [[m, m, 1, 1], [W - m, m, -1, 1], [m, H - m, 1, -1], [W - m, H - m, -1, -1]]) {
        g.moveTo(x, y + sy * L); g.lineTo(x, y); g.lineTo(x + sx * L, y);
      }
      g.stroke();
    }
  }

  // Group the existing full-frame motion map into a few localized attention
  // targets. Cropping/panning never feeds back into this analysis.
  function motionRegions(motion,width,height,mirror=true) {
    const cols=16,rows=Math.max(1,Math.ceil(height/width*cols)),n=cols*rows;
    const count=new Uint32Array(n),mass=new Float32Array(n),xs=new Float32Array(n),ys=new Float32Array(n);
    let changed=0;
    for(let y=0,i=0;y<height;y++)for(let x=0;x<width;x++,i++) {
      const weight=motion[i];
      if(weight<0.15)continue;
      const cell=Math.min(rows-1,Math.floor(y/height*rows))*cols+Math.min(cols-1,Math.floor(x/width*cols));
      count[cell]++;mass[cell]+=weight;xs[cell]+=(x+0.5)/width*weight;ys[cell]+=(y+0.5)/height*weight;changed++;
    }
    // Exposure changes and camera shake cover the image; don't chase them.
    if(changed>width*height*0.45)return [];
    const threshold=Math.max(3,width*height/n*0.08),seen=new Uint8Array(n),regions=[];
    for(let start=0;start<n;start++) {
      if(seen[start] || count[start]<threshold)continue;
      const pending=[start];seen[start]=1;
      let pixels=0,energy=0,x=0,y=0,tiles=0;
      while(pending.length) {
        const cell=pending.pop(),cx=cell%cols,cy=Math.floor(cell/cols);
        pixels+=count[cell];energy+=mass[cell];x+=xs[cell];y+=ys[cell];tiles++;
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++) {
          const nx=cx+dx,ny=cy+dy,index=ny*cols+nx;
          if(nx<0 || nx>=cols || ny<0 || ny>=rows || seen[index] || count[index]<threshold)continue;
          seen[index]=1;pending.push(index);
        }
      }
      if(pixels<width*height*0.0015 || tiles>n*0.45 || !energy)continue;
      regions.push({kind:'motion',x:(x/energy*2-1)*(mirror?1:-1),y:y/energy*2-1,
        energy:energy/(width*height),closeness:0.3});
    }
    return regions.sort((a,b)=>b.energy-a.energy).slice(0,3);
  }

  root.GladosVision = { Vision, MODES, motionRegions };
})(typeof window !== 'undefined' ? window : globalThis);
