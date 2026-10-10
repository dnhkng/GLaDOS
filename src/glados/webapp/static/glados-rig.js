// Adapted from the user-provided GLaDOS Concept Stills.html.
// GLaDOS aperture rig: pure parameters -> SVG, plus a programmatic animator.
// Runs in the browser (window.GladosRig) and in Node (globalThis.GladosRig) with no dependencies.
// Everything a mood does is a set of numbers, so any two states blend by interpolation,
// and idle motion, pointer gaze and voice are layers added on top of the blended look.
(function (root) {
  const INK = '#E8ECEF';
  const BG = '#0E1116';
  const f = (n) => Math.round(n * 100) / 100;

  const NEUTRAL = {
    lookX: 0, lookY: 0,   // -1..1 gaze; inner layers move more than outer (parallax)
    ox: 0, oy: 0,         // offset of the inner stack against the housing (emotion, not gaze)
    skew: 0,              // how unevenly the rings separate along the offset (0 = concentric)
    a: 22,                // blade opening radius (pupil size)
    bladeRot: 0.2,        // blade rotation, radians
    ringRot: 0.3,         // outer housing rotation, radians
    ringGap: 0.22,        // gap between housing segments, radians
    scale: 1,             // housing size
    zoom: 1,              // whole-eye camera nearness zoom
    lidTop: -90, lidTopAng: 0, // lid edges in eye space; ±90 is fully open
    lidBot: 90, lidBotAng: 0,
    color: '#F2C265',     // centre colour
    glow: 1,
    ex: 0, dy: 0, tilt: 0, // whole-eye drift and roll
    awake: 1,             // 0 when offline: no pointer gaze, eye flicks or blinks, barely any idle motion
    lidFollow: 1,         // 1: lid asymmetry swings to the gaze side; 0: the look's slant is fixed
    speak: 0,             // 0..1 voice amplitude (a layer, not a look)
  };

  const LOOKS = {
    neutral:      {},
    quizzical:    { lidFollow: 0, tilt: 12, ox: -10, oy: -12, skew: 0.8, lookX: -0.3, lookY: -0.3, a: 24, lidTop: -58, lidTopAng: 18, ringRot: 0.1, color: '#9FE3F0' },
    'angry glare':{ scale: 0.92, oy: 8, skew: 0.3, a: 12, lidTop: -26, lidTopAng: 12, lidBot: 32, lidBotAng: -6, ringGap: 0.1, ringRot: 0.785, color: '#FF5A3C', glow: 1.6, dy: 6 },
    suspicious:   { lookX: -0.9, ox: -6, skew: 0.5, a: 16, lidTop: -30, lidBot: 36, color: '#F2A93B' },
    smug:         { tilt: -8, dy: -6, ox: -4, oy: -4, a: 18, lidTop: -48, lidTopAng: 6, lidBot: 34, lidBotAng: -10, color: '#F6D36B' },
    surprised:    { scale: 1.12, dy: -8, a: 40, ringGap: 0.4, oy: -4, color: '#FFFFFF', glow: 1.3 },
    bored:        { lookX: 0.6, lookY: 0.5, oy: 10, skew: 0.4, a: 20, lidTop: -12, lidTopAng: 4, dy: 8, color: '#8FA0AE', glow: 0.5 },
    disappointed: { oy: 10, skew: 0.3, lookY: 0.3, a: 18, lidTop: -34, lidTopAng: 8, lidBot: 70, ringRot: 0.45, color: '#B39DDB', glow: 0.7, dy: 6, tilt: 4 },
    processing:   { a: 8, bladeRot: 1.1, ringRot: 0.7, ox: -4, oy: -6, skew: 1, lookY: -0.3, color: '#7EB6FF', lidTop: -70, lidBot: 70 },
    offline:      { awake: 0, dy: 16, oy: 14, a: 4, lidTop: 4, lidBot: 6, ringGap: 0.6, color: '#3A424C', glow: 0 },
    sleeping:     { awake: 0, dy: 16, oy: 14, a: 4, lidTop: 4, lidBot: 6, ringGap: 0.6, color: '#796548', glow: 0.15 },
  };

  // ---------- rendering ----------
  function arcPath(cx, cy, r, s, e) {
    return `M${f(cx + r * Math.cos(s))},${f(cy + r * Math.sin(s))} A${f(r)},${f(r)} 0 0 1 ${f(cx + r * Math.cos(e))},${f(cy + r * Math.sin(e))}`;
  }

  // Returns the SVG content (no <svg> wrapper) for viewBox="-200 -200 400 400".
  function renderEye(p0, opts = {}) {
    const q = { ...NEUTRAL, ...p0 };
    const id = opts.id || 'ap' + Math.random().toString(36).slice(2, 7);
    const v = Math.max(0, Math.min(1, q.speak));
    const p = { ...q, a: q.aFinal ? q.a : q.a + v * Math.max(4, q.a * 0.35), glow: q.glow * (1 + 0.5 * v) };
    const S = p.scale;
    // Lids open wider on the side she is looking towards. A look's own slant keeps its size but
    // swings to the gaze side (passing through level as it crosses), plus a little extra slant with gaze.
    const gx = Math.max(-1, Math.min(1, p.lookX));
    const own = Math.sign(p.lidTopAng - p.lidBotAng); // +1: this look opens wider on the left
    const w = Math.min(1, Math.abs(gx) * 2.5);
    const lf = Math.max(0, Math.min(1, p.lidFollow));
    const flip0 = own === 0 || gx === 0 ? 1 : (1 - w) + w * (Math.sign(-gx) === own ? 1 : -1);
    const flip = 1 + (flip0 - 1) * lf;
    p.lidTopAng = p.lidTopAng * flip - 9 * gx * lf;
    p.lidBotAng = p.lidBotAng * flip + 9 * gx * lf;
    // The bottom lid never crosses the top one: where they would meet, it tucks just under it.
    {
      const W = 90, tT = Math.tan(p.lidTopAng * Math.PI / 180), tB = Math.tan(p.lidBotAng * Math.PI / 180);
      const yl = Math.max(p.lidBot - tB * W, p.lidTop - tT * W + 2);
      const yr = Math.max(p.lidBot + tB * W, p.lidTop + tT * W + 2);
      p.lidBot = (yl + yr) / 2;
      p.lidBotAng = Math.atan((yr - yl) / (2 * W)) * 180 / Math.PI;
    }
    const G = 1.3; // gaze travel
    const L = (k, e) => [p.lookX * k * G + p.ox * e, p.lookY * k * G + p.oy * e];
    const [hx, hy] = L(3, 0);
    const [mx, my] = L(8, 0.5 + p.skew * 0.3);
    const [bx, by] = L(14, 1 + p.skew * 0.6);
    const [px0, py0] = L(20, 1.3 + p.skew);
    const lim = Math.max(0, Math.min(p.a, 62) * 0.5), ddx = px0 - bx, ddy = py0 - by, dl = Math.hypot(ddx, ddy) || 1;
    const kk = Math.min(1, lim / dl), cx = bx + ddx * kk, cy = by + ddy * kk;

    const Ro = 108 * S + v * 6;
    let arcs = '';
    for (let i = 0; i < 4; i++) {
      const s = i * Math.PI / 2 + p.ringGap / 2 + p.ringRot, e = (i + 1) * Math.PI / 2 - p.ringGap / 2 + p.ringRot;
      arcs += `<path d="${arcPath(hx, hy, Ro, s, e)}"/>`;
    }
    let echo = '';
    if (v > 0.01) {
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 2; j++) {
          const R = Ro + (j ? 30 : 16), s = i * Math.PI / 2 + 0.55 + p.ringRot + j * 0.2, e = s + 0.5 - j * 0.2;
          echo += `<path d="${arcPath(hx, hy, R, s, e)}" stroke="${p.color}" opacity="${f((j ? .35 : .7) * v)}" stroke-width="1.6"/>`;
        }
      }
    }
    const Rm = 84 * S, R = 68 * S, a = Math.max(1, Math.min(p.a, R - 6));
    let blades = '';
    for (let i = 0; i < 7; i++) {
      const t = (i / 7) * Math.PI * 2 + p.bladeRot;
      const ai = p.bladeA ? Math.max(1, Math.min(p.bladeA[i], R - 4)) : a; // each blade settles on its own
      const px = ai * Math.cos(t), py = ai * Math.sin(t);
      const len = Math.sqrt(R * R - ai * ai);
      blades += `<line x1="${f(bx + px)}" y1="${f(by + py)}" x2="${f(bx + px - Math.sin(t) * len)}" y2="${f(by + py + Math.cos(t) * len)}" opacity=".8"/>`;
    }
    const lidRect = (y, ang, top) => `<rect x="-150" y="${f(top ? y - 200 : y)}" width="300" height="200" transform="rotate(${f(ang)} 0 ${f(y)})" fill="${BG}" stroke="none"/>`;
    const lidEdge = (y, ang) => `<line x1="-100" y1="${f(y)}" x2="100" y2="${f(y)}" transform="rotate(${f(ang)} 0 ${f(y)})"/>`;

    const listening = Math.max(0, Math.min(1, p.listening || 0));
    const listeningHalo = listening > 0.01 ?
      `<circle class="listening-halo" cx="${f(hx)}" cy="${f(hy)}" r="${f(Ro + 8)}" stroke="#9FE3F0" stroke-width="2.8" opacity="${f(listening * 0.65)}"/>` : '';
    let hud = '';
    if (p.hud) {
      const br = (x, y, sx, sy) => `<path d="M${x},${y + sy * 14} L${x},${y} L${x + sx * 14},${y}" stroke-width="1.4" opacity=".55"/>`;
      const txt = p.hud.map((l, i) => `<text x="-180" y="${152 + i * 13}">${l}</text>`).join('');
      hud = `${br(-186, -186, 1, 1)}${br(186, -186, -1, 1)}${br(-186, 186, 1, -1)}${br(186, 186, -1, -1)}
<g fill="${INK}" stroke="none" font-family="IBM Plex Mono, ui-monospace, Menlo, monospace" font-size="9.5" letter-spacing=".6" opacity=".6">${txt}</g>`;
    }

    return `<defs>
<radialGradient id="${id}g" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="scale(190)">
<stop offset="0" stop-color="#1A2029"/><stop offset="1" stop-color="${BG}"/></radialGradient>
<filter id="${id}b" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="7"/></filter>
<clipPath id="${id}c"><circle cx="${f(mx)}" cy="${f(my)}" r="${f(Rm + 4)}"/></clipPath>
</defs>
<rect x="-200" y="-200" width="400" height="400" fill="url(#${id}g)"/>
<g fill="none" stroke="${INK}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<g transform="translate(${f(p.ex)} ${f(p.dy)}) rotate(${f(p.tilt)}) scale(${f(p.zoom)})">
${echo}${arcs}${listeningHalo}
<g clip-path="url(#${id}c)">
<circle cx="${f(mx)}" cy="${f(my)}" r="${f(Rm)}" opacity=".5"/>
<circle cx="${f(bx)}" cy="${f(by)}" r="${f(R)}"/>
${blades}
<circle cx="${f(bx)}" cy="${f(by)}" r="${f(a)}" fill="${BG}"/>
<circle cx="${f(cx)}" cy="${f(cy)}" r="${f(Math.min(a * 1.1, 36))}" fill="${p.color}" stroke="none" opacity="${f(Math.min(1, .45 * p.glow))}" filter="url(#${id}b)"/>
<circle cx="${f(cx)}" cy="${f(cy)}" r="${f(Math.max(2, a * 0.42))}" fill="${p.color}" stroke="none" opacity="${f(Math.min(1, .25 + .75 * p.glow))}"/>
${lidRect(p.lidBot, p.lidBotAng, false)}${lidEdge(p.lidBot, p.lidBotAng)}
${lidRect(p.lidTop, p.lidTopAng, true)}${lidEdge(p.lidTop, p.lidTopAng)}
</g>
</g>
${hud}
</g>`;
  }

  function renderSVG(p, opts) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-200 -200 400 400" width="400" height="400">${renderEye(p, opts)}</svg>`;
  }

  // ---------- animation ----------
  const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const rgbToHex = (c) => '#' + c.map((x) => Math.round(Math.max(0, Math.min(255, x))).toString(16).padStart(2, '0')).join('');
  const NUM_KEYS = Object.keys(NEUTRAL).filter((k) => typeof NEUTRAL[k] === 'number' && k !== 'speak');
  const approach = (cur, tgt, rate, dt) => cur + (tgt - cur) * (1 - Math.exp(-rate * dt));

  // Deterministic-ish smooth noise from a few sines (no dependency needed).
  const wob = (t, s) => Math.sin(t * 0.73 + s) * 0.6 + Math.sin(t * 1.37 + s * 2.1) * 0.3 + Math.sin(t * 2.91 + s * 3.7) * 0.1;

  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const SARCASTIC = new Set(['smug', 'disappointed', 'bored', 'suspicious']);
  // how strongly a mood lingers after she leaves it (0 = clean switch)
  const LINGER = { 'angry glare': 1, disappointed: 0.6, surprised: 0.45, suspicious: 0.4, smug: 0.3, quizzical: 0.2, processing: 0.2 };
  // moods in which she sometimes ignores you on purpose
  const ALOOF = { bored: 1, smug: 0.35, disappointed: 0.5 };

  class Animator {
    constructor(opts = {}) {
      this.blendRate = opts.blendRate ?? 5;     // higher = faster mood changes
      this.idle = opts.idle ?? 1;               // 0..1 amount of idle motion
      this.speechGaze = opts.speechGaze ?? 1;   // 0..1 glances that follow the phrasing
      this.faceScan = opts.faceScan ?? 1;       // 0..1 eye-to-eye-to-mouth scanning
      this.mechanical = opts.mechanical ?? 1;   // 0..1 blade overshoot and housing ratchet
      this.state = { ...NEUTRAL };
      this.rgb = hexToRgb(NEUTRAL.color);
      this.target = { ...NEUTRAL };
      this.mood = 'neutral';
      this.t = 0;
      this.gaze = { x: 0, y: 0, tx: 0, ty: 0, active: 0, follow: false, faceWidth: 0.15, faceHeight: 0.25 };
      this.sacc = { x: 0, y: 0, tx: 0, ty: 0, next: 1, face: true, fixation: 'center' };
      this.blink = { v: 0, next: 2.5, phase: 0, close: 0.07, hold: 0, open: 0.12, held: 0, slow: false };
      this.amp = 0; this.ampIn = 0;
      this.talk = { x: 0, y: 0, tx: 0, ty: 0, nod: 0, prev: 0, onsets: 0, quiet: 9, every: 1 };
      // Presence and approximate distance from the camera's face detector.
      this.pres = { present: true, gone: 0, arrive: 0, arriveT: 9, sulk: 0, close: 0.3, closeS: 0.3, att: 1, attS: 1 };
      // mechanics: each blade is a slightly different spring; the housing ratchets in steps
      this.blades = Array.from({ length: 7 }, (_, i) => ({ x: NEUTRAL.a, v: 0, k: 150 + ((i * 3) % 7) * 22 }));
      this.housing = { x: NEUTRAL.ringRot, v: 0, tgt: NEUTRAL.ringRot, tick: 0, nudge: 0, next: 5 };
      // afterglow of the previous mood: its colour and tension fade out over a few seconds
      this.after = { w: 0, rgb: [0, 0, 0], a: 0, flick: 0, next: 0.5 };
      // aloof: looking at "something more interesting" instead of you
      this.aloof = { on: false, w: 0, x: 0, y: 0, t: 0, next: 5 };
      // listening: someone is talking to her (voice activity from the mic)
      this.listen = { on: false, w: 0, since: 0 };
    }
    setLook(name) {
      const look = typeof name === 'string' ? LOOKS[name] : name;
      if (!look) return;
      const prev = this.mood;
      if (typeof name === 'string') this.mood = name;
      const L = LINGER[prev] || 0;
      if (L > 0 && prev !== this.mood) {
        const A = this.after;
        A.w = L; A.rgb = hexToRgb(this.target.color); A.a = this.target.a; A.flick = 0; A.next = 0.4 + Math.random();
      }
      this.aloof.on = false; this.aloof.next = 3 + Math.random() * 4;
      this.target = { ...NEUTRAL, ...look };
    }
    get sarcastic() { return SARCASTIC.has(this.mood); }
    // x, y in -1..1 relative to the eye; pass null to release the gaze
    lookAt(x, y, follow = false, faceSize = null, kind = 'face') {
      if (x == null) { this.gaze.active = 0; this.gaze.follow = false; return; }
      if (follow && (!this.gaze.active || !this.gaze.follow || this.gaze.kind!==kind)) {
        // Establish eye contact before scanning; don't carry an idle glance into a new face.
        Object.assign(this.sacc, { x: 0, y: 0, tx: 0, ty: 0, next: 0.5, fixation: 'center' });
      }
      this.gaze.tx = Math.max(-1, Math.min(1, x)); this.gaze.ty = Math.max(-1, Math.min(1, y)); this.gaze.active = 1;
      this.gaze.follow = follow;
      this.gaze.kind = kind;
      if (faceSize && [faceSize.width, faceSize.height].every(v => Number.isFinite(v) && v > 0 && v <= 1)) {
        this.gaze.faceWidth = faceSize.width; this.gaze.faceHeight = faceSize.height;
      }
    }
    // Is someone there, and how close (0 far .. 1 very close)?
    setPresence(present, closeness) {
      const p = this.pres;
      if (present && !p.present && p.gone > 0.8) { p.arrive = 1; p.arriveT = 0; }
      if (present) p.gone = 0;
      p.present = !!present;
      if (closeness != null) p.close = clamp01(closeness);
    }
    // Is the viewer paying attention (facing her)? 1 yes, 0 looking away.
    setAttention(a) { this.pres.att = clamp01(a); }
    // Someone is talking to her (voice activity detected on the mic). She stops ignoring you and listens.
    setListening(on) {
      const L = this.listen;
      if (on && !L.on) { L.since = 0; if (this.aloof.on) { this.aloof.on = false; this.aloof.next = 8 + Math.random() * 6; } }
      L.on = !!on;
    }
    // raw voice amplitude 0..1, call every frame while audio plays (or 0 when silent)
    setVoice(a) { this.ampIn = clamp01(a); }
    // A short blink shared with the optic feed when Vision starts an inference.
    inferenceBlink() {
      if (this.blink.phase !== 0) return;
      Object.assign(this.blink, { phase: 1, close: 0.08, hold: 0.03, open: 0.14, held: 0, slow: false,
        next: 2.5 + Math.random() * 4.5 });
    }
    // A slow, deliberate blink: close, hold, open. Good right before a cutting remark.
    slowBlink() {
      const b = this.blink;
      Object.assign(b, { phase: 1, close: 0.32, hold: 0.28, open: 0.5, held: 0, slow: true });
    }

    step(dt) {
      dt = Math.min(dt, 0.1);
      this.t += dt;
      const s = this.state;
      for (const k of NUM_KEYS) s[k] = approach(s[k], this.target[k], this.blendRate, dt);
      const trgb = hexToRgb(this.target.color);
      this.rgb = this.rgb.map((c, i) => approach(c, trgb[i], this.blendRate * 0.8, dt));
      this.amp = approach(this.amp, this.ampIn, this.ampIn > this.amp ? 30 : 9, dt);

      // speech gaze: syllable onsets move the gaze; strong onsets nod; silence recentres
      const k = this.talk;
      if (this.ampIn < 0.15) k.prev = 0;
      const on = k.prev === 0 && this.ampIn > 0.35;
      if (on) k.prev = 1;
      k.quiet = this.amp < 0.08 ? k.quiet + dt : 0;
      if (on) {
        k.onsets++;
        if (this.ampIn > 0.75) k.nod = 1;
        if (k.onsets >= k.every) {
          k.onsets = 0; k.every = 1 + Math.floor(Math.random() * 3);
          const side = k.tx > 0.05 ? -1 : k.tx < -0.05 ? 1 : (Math.random() < 0.5 ? -1 : 1);
          k.tx = side * (0.12 + Math.random() * 0.25);
          k.ty = (Math.random() - 0.6) * 0.3;
        }
      }
      if (k.quiet > 0.5) { k.tx = 0; k.ty = 0; }
      k.nod = Math.max(0, k.nod - dt * 4);
      k.x = approach(k.x, k.tx, 9, dt);
      k.y = approach(k.y, k.ty + k.nod * 0.18, 12, dt);

      // presence
      const P = this.pres;
      if (!P.present) P.gone += dt;
      P.arriveT += dt;
      P.arrive = Math.max(0, P.arrive - dt / 1.4);
      P.sulk = approach(P.sulk, !P.present && P.gone > 5 ? 1 : 0, P.present ? 3 : 0.8, dt);
      P.closeS = approach(P.closeS, P.close, 4, dt);
      P.attS = approach(P.attS, P.att, 6, dt);

      // gaze target (pointer / face)
      const g = this.gaze;
      const tracking = g.active && P.present;
      const following = tracking && g.follow;
      const gazeRate = tracking ? (g.follow ? 20 : 14) : 4;
      g.x = approach(g.x, tracking ? g.tx : 0, gazeRate, dt);
      g.y = approach(g.y, tracking ? g.ty : 0, gazeRate, dt);

      // saccades: face scanning when looking at someone, small random flicks otherwise
      const m = this.sacc;
      m.next -= dt;
      m.face = P.present && (tracking || Math.hypot(this.target.lookX, this.target.lookY) < 0.25);
      if (m.next <= 0) {
        if (following && g.kind==='motion') {
          m.fixation='center';m.tx=0;m.ty=0;m.next=0.2;
        } else if (following) {
          // Mostly alternate between eyes, with briefer mouth glances and a return to center.
          const r = Math.random();
          const alternate = m.fixation === 'left' ? 'right' : m.fixation === 'right' ? 'left' :
            (Math.random() < 0.5 ? 'left' : 'right');
          m.fixation = r < 0.72 ? alternate : r < 0.9 ? 'mouth' : 'center';
          const [fx, fy] = m.fixation === 'left' ? [-0.32, -0.14] : m.fixation === 'right' ? [0.32, -0.14] :
            m.fixation === 'mouth' ? [0, 0.3] : [0, 0];
          m.tx = fx + (Math.random() - 0.5) * 0.025;
          m.ty = fy + (Math.random() - 0.5) * 0.025;
          m.next = m.fixation === 'mouth' ? 0.18 + Math.random() * 0.22 : 0.55 + Math.random() * 0.75;
        } else if (m.face) {
          const r = Math.random();
          const [fx, fy] = r < 0.36 ? [-0.32, -0.14] : r < 0.72 ? [0.32, -0.14] : r < 0.92 ? [0, 0.3] : [0, 0.02];
          m.tx = fx + (Math.random() - 0.5) * 0.05; m.ty = fy + (Math.random() - 0.5) * 0.05;
          m.next = 0.3 + Math.random() * 0.8;
        } else {
          m.tx = (Math.random() - 0.5) * 0.3; m.ty = (Math.random() - 0.5) * 0.2; m.next = 0.7 + Math.random() * 2.2;
        }
      }
      m.x = approach(m.x, m.tx, following ? 38 : 25, dt);
      m.y = approach(m.y, m.ty, following ? 38 : 25, dt);

      // blinks: quick ones at random; in sarcastic moods some are slow and deliberate
      const b = this.blink;
      b.next -= dt;
      if (b.next <= 0 && b.phase === 0) {
        b.next = 2.5 + Math.random() * 4.5;
        if (this.sarcastic && Math.random() < 0.35) this.slowBlink();
        else Object.assign(b, { phase: 1, close: 0.07, hold: 0, open: 0.12, held: 0, slow: false });
      }
      if (b.phase === 1) { b.v += dt / b.close; if (b.v >= 1) { b.v = 1; b.phase = 2; b.held = 0; } }
      else if (b.phase === 2) { b.held += dt; if (b.held >= b.hold) b.phase = 3; }
      else if (b.phase === 3) { b.v -= dt / b.open; if (b.v <= 0) { b.v = 0; b.phase = 0; } }

      // mechanics: blades chase the opening with staggered, slightly underdamped springs
      const aT = this.openingTarget();
      const M = this.mechanical;
      const sub = 4, h = dt / sub;
      for (const bl of this.blades) {
        const kk = bl.k, c = 2 * 0.38 * Math.sqrt(kk);
        for (let i = 0; i < sub; i++) { bl.v += (kk * (aT - bl.x) - c * bl.v) * h; bl.x += bl.v * h; }
        if (!M) { bl.x = aT; bl.v = 0; }
      }
      // housing: a ratchet that advances in small clicks toward where it should be, overshooting slightly
      const H = this.housing;
      H.next -= dt;
      if (H.next <= 0) { H.nudge += (Math.random() < 0.5 ? -1 : 1) * 0.07 * this.idle; H.nudge *= 0.7; H.next = 4 + Math.random() * 5; }
      const want = s.ringRot + H.nudge + this.amp * 0.04;
      H.tick -= dt;
      if (H.tick <= 0) {
        H.tick = 0.085;
        const d = want - H.tgt;
        if (Math.abs(d) > 0.004) H.tgt += Math.max(-0.09, Math.min(0.09, d));
      }
      const hk = 520, hc = 2 * 0.3 * Math.sqrt(hk);
      for (let i = 0; i < sub; i++) { H.v += (hk * (H.tgt - H.x) - hc * H.v) * h; H.x += H.v * h; }
      if (!M) { H.x = want; H.v = 0; }

      // afterglow decays; while it lasts, the old colour flickers through now and then
      const Af = this.after;
      Af.w = Math.max(0, Af.w - dt / 4.5);
      Af.next -= dt;
      if (Af.w > 0.08 && Af.next <= 0) { Af.flick = 1; Af.next = 0.5 + Math.random() * 1.4; }
      Af.flick = Math.max(0, Af.flick - dt / 0.14);

      // listening
      const Ls = this.listen;
      Ls.since += dt;
      Ls.w = approach(Ls.w, Ls.on ? 1 : 0, Ls.on ? 12 : 3, dt);

      // aloof: in some moods she glances off at something else, then slowly comes back
      const Al = this.aloof, aloofness = following ? 0 : (ALOOF[this.mood] || 0) * (P.present ? 1 : 0);
      Al.next -= dt;
      if (Al.on) {
        Al.t -= dt;
        // leaning in close, talking to her or her speaking ends it early
        if (following || Al.t <= 0 || P.closeS > 0.75 || Ls.on || this.amp > 0.2) { Al.on = false; Al.next = 5 + Math.random() * 7; }
      } else if (aloofness > 0 && Al.next <= 0 && !Ls.on && this.amp < 0.05) {
        if (Math.random() < aloofness) {
          Al.on = true; Al.t = 2 + Math.random() * 2.5;
          const side = this.gaze.x > 0 ? -1 : 1; // away from you
          Al.x = side * (0.7 + Math.random() * 0.25); Al.y = -0.25 - Math.random() * 0.3;
        }
        Al.next = 4 + Math.random() * 6;
      }
      Al.w = approach(Al.w, Al.on ? 1 : 0, Al.on ? 3.5 : 0.9, dt); // quick to look away, slow to return
      return this.params();
    }

    // blade opening before mechanics, including idle breathing and the voice
    openingTarget() {
      const s = this.state, P = this.pres;
      const A = clamp01(s.awake), I = this.idle * (0.15 + 0.85 * A);
      const lean = clamp01((P.closeS - 0.6) / 0.4) * A;
      const stare = this.stare();
      let a = s.a + wob(this.t, 3) * 1.2 * I;
      a *= (1 - 0.35 * lean) * (1 - 0.2 * stare);
      a += P.arrive * 12 * A;
      a += this.listen.w * A * (6 + Math.sin(this.listen.since * 4) * 0.8);
      return a + this.amp * Math.max(4, a * 0.35);
    }
    // pointed stare: the viewer looks away while she is (or was just) talking
    stare() {
      const talking = clamp01(1.5 - this.talk.quiet);
      return clamp01((1 - this.pres.attS) * talking * 1.6) * clamp01(this.state.awake) * (this.pres.present ? 1 : 0);
    }

    params() {
      const s = this.state, t = this.t, v = this.amp, P = this.pres;
      const A = clamp01(s.awake);
      const I = this.idle * (0.15 + 0.85 * A);
      const stare = this.stare();
      const lean = clamp01((P.closeS - 0.6) / 0.4) * A;
      const sulk = P.sulk * A;
      const following = this.gaze.active && this.gaze.follow && P.present;
      const SG = this.speechGaze * A * (1 - stare) * (following ? 0.15 : 1);
      const gz = { x: this.gaze.x * A, y: this.gaze.y * A };
      const gw = A * Math.min(1, this.gaze.active ? 1 : Math.hypot(gz.x, gz.y) * 3);
      const aloof = following ? 0 : this.aloof.w * A * (1 - stare);
      const listen = this.listen.w * A;
      const SW = (this.sacc.face ? A * this.faceScan : I * A * (1 - gw * 0.5)) *
        (1 - stare) * (1 - aloof) * (1 - 0.4 * listen);
      // Approximate facial features within the detected box. Offsets move with
      // the tracked face and shrink while catching up to a quick head movement.
      const catchingUp = following ? clamp01(1 - Math.hypot(this.gaze.tx - this.gaze.x, this.gaze.ty - this.gaze.y) * 3) : 1;
      const scanX = following ? Math.max(-0.12, Math.min(0.12, this.sacc.x * this.gaze.faceWidth * 1.4)) : this.sacc.x;
      const scanY = following ? Math.max(-0.14, Math.min(0.14, this.sacc.y * this.gaze.faceHeight * 1.4)) : this.sacc.y;

      // searching for someone who left, then sulking
      const searching = !P.present && P.gone < 5 ? A * clamp01(P.gone * 2) : 0;
      const sx = Math.sin(P.gone * 1.7) * 0.8 * searching;
      // arrival look-over: glance down their body, then up to the face
      const at = P.arriveT;
      const over = at < 1.3 ? (at < 0.45 ? at / 0.45 : Math.max(0, 1 - (at - 0.45) / 0.5)) * A : 0;

      const moodGaze = following ? 0 : (1 - gw * 0.8);
      let lookX = s.lookX * moodGaze + gz.x + scanX * SW * catchingUp + this.talk.x * SG + sx;
      let lookY = s.lookY * moodGaze + gz.y + scanY * SW * catchingUp + this.talk.y * SG - 0.1 * searching + over * (following ? 0 : 0.7);
      lookX = lookX * (1 - sulk) + 0.55 * sulk;
      lookY = lookY * (1 - sulk) + 0.45 * sulk;
      lookX = lookX * (1 - stare * 0.7) + gz.x * stare * 0.7;
      lookY = lookY * (1 - stare * 0.7) + gz.y * stare * 0.7;
      lookX = lookX * (1 - aloof) + this.aloof.x * aloof;
      lookY = lookY * (1 - aloof) + this.aloof.y * aloof;

      // lids: blink, then narrowing from leaning in, staring and sulking
      let top = s.lidTop, bot = s.lidBot;
      const mid = (top + bot) / 2;
      const narrow = Math.max(lean * 0.55, stare * 0.4);
      if (bot - top > 24) { top += (mid - 12 - top) * narrow; bot += (mid + 12 - bot) * narrow; }
      top = top * (1 - sulk) + Math.max(top, -16) * sulk;
      const open = Math.max(0, bot - top);
      const bl = open > 12 && A > 0.5 ? this.blink.v : 0;
      const mid2 = (top + bot) / 2;
      top += (mid2 - 1 - top) * bl; bot += (mid2 + 1 - bot) * bl;

      const aT = this.openingTarget();
      const bladeA = this.mechanical ? this.blades.map((b) => b.x) : null;
      // centre colour: current mood, pulled toward the lingering one (with flickers), and red when staring
      const Af = this.after, aw = Math.min(1, Af.w * (0.45 + 0.55 * Af.flick)) * A;
      let rgb = this.rgb.map((c, i) => c + (Af.rgb[i] - c) * aw);
      if (stare > 0.01) rgb = rgb.map((c, i) => c + ([255, 90, 60][i] - c) * 0.3 * stare);
      // A distinct attention cue while the microphone detects speech. Preserve
      // the mood beneath it and fade back naturally when the person stops.
      rgb = rgb.map((c,i)=>c+([159,227,240][i]-c)*listen*0.65);
      const colour = rgbToHex(rgb);
      return {
        ...s,
        color: colour,
        lookX, lookY,
        ex: gz.x * 16,
        ox: s.ox + wob(t * 0.6, 1) * 2 * I,
        oy: s.oy + wob(t * 0.6, 2) * 2 * I + sulk * 6,
        a: aT, aFinal: true, bladeA: bladeA && bladeA.map((x) => x + (Af.a - x) * 0.3 * Af.w * A + 3 * listen),
        scale: s.scale * (1 - 0.05 * lean) * (1 + 0.06 * P.arrive * A),
        zoom: s.zoom * (1 + 0.6 * lean),
        bladeRot: s.bladeRot + wob(t * 0.4, 4) * 0.04 * I + v * 0.12,
        ringRot: this.housing.x,
        glow: s.glow * (1 - 0.4 * sulk) * (1 + 0.3 * stare),
        dy: s.dy + gz.y * 10 + Math.sin(t * 0.9) * 3 * I + sulk * 8,
        tilt: s.tilt - gz.x * gz.y * 6 + wob(t * 0.35, 6) * 1.2 * I - lean * 3 + listen * 4 - aloof * this.aloof.x * 3,
        lidTop: top, lidBot: bot, blink: bl,
        speak: v, listening: listen,
      };
    }
  }

  // Synthetic speech envelope for demos: syllables, word gaps, sentence pauses.
  class DemoVoice {
    constructor() { this.t = 0; this.events = []; this.until = 0; }
    say(seconds) {
      const ev = []; let t = this.t + 0.05; const end = t + seconds;
      while (t < end) {
        const words = 1 + Math.floor(Math.random() * 4);
        for (let w = 0; w < words && t < end; w++) {
          const syl = 1 + Math.floor(Math.random() * 3);
          for (let k = 0; k < syl; k++) {
            const d = 0.08 + Math.random() * 0.14, peak = 0.45 + Math.random() * 0.55;
            ev.push([t, d, peak]); t += d * 0.9;
          }
          t += 0.06 + Math.random() * 0.16;
        }
        t += 0.25 + Math.random() * 0.3;
      }
      this.events = ev; this.until = end;
      return end - this.t;
    }
    step(dt) {
      this.t += dt;
      let a = 0;
      for (const [s, d, pk] of this.events) {
        const x = (this.t - s) / d;
        if (x > 0 && x < 1) a = Math.max(a, pk * Math.sin(Math.PI * x));
      }
      return a;
    }
    get speaking() { return this.t < this.until; }
  }

  root.GladosRig = { NEUTRAL, LOOKS, renderEye, renderSVG, Animator, DemoVoice };
})(typeof window !== 'undefined' ? window : globalThis);
