/* Animated tab icon: GLaDOS's eye, driven by the same presentation as the avatar. */
(function (root) {
  "use strict";

  const SIZE = 64, FPS = 12;
  // Colour and motion per activity; hues follow the console's portal orange / blue.
  const MODES = {
    Idle:         { rgb: [255, 160, 32], pulse: 0.25, speed: 0.6, spin: 0.15 },
    Listening:    { rgb: [64, 160, 255], pulse: 0.45, speed: 2.5, spin: 0.4 },
    Speaking:     { rgb: [255, 160, 32], pulse: 0.7,  speed: 7,   spin: 0.6 },
    Processing:   { rgb: [255, 200, 80], pulse: 0.2,  speed: 1.5, spin: 3 },
    Sleeping:     { rgb: [120, 90, 40],  pulse: 0.3,  speed: 0.3, spin: 0 },
    Quiet:        { rgb: [120, 90, 40],  pulse: 0.3,  speed: 0.3, spin: 0 },
    Connecting:   { rgb: [140, 140, 140], pulse: 0.5, speed: 2,   spin: 1 },
    Disconnected: { rgb: [220, 50, 40],  pulse: 0.8,  speed: 1.5, spin: 0 },
  };

  function mode(state) {
    const avatar = root.GladosAvatar;
    const activity = avatar ? avatar.presentation(state || {}).activity : "Idle";
    return MODES[activity.replace(/^Demo · /, "")] || MODES.Idle;
  }

  function draw(ctx, m, t) {
    const c = SIZE / 2, [r, g, b] = m.rgb;
    const glow = 1 - m.pulse + m.pulse * (0.5 + 0.5 * Math.sin(t * m.speed));
    const rgba = a => `rgba(${r},${g},${b},${a})`;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "#14161a";
    ctx.beginPath(); ctx.arc(c, c, c - 1, 0, Math.PI * 2); ctx.fill();
    // Aperture ring: six segments turning around the eye.
    ctx.strokeStyle = rgba(0.35 + 0.35 * glow);
    ctx.lineWidth = 5;
    const turn = t * m.spin;
    for (let i = 0; i < 6; i++) {
      const start = turn + i * Math.PI / 3;
      ctx.beginPath(); ctx.arc(c, c, c - 6, start, start + Math.PI / 4); ctx.stroke();
    }
    const iris = ctx.createRadialGradient(c, c, 0, c, c, 18);
    iris.addColorStop(0, rgba(1));
    iris.addColorStop(0.35, rgba(0.6 + 0.4 * glow));
    iris.addColorStop(1, rgba(0));
    ctx.fillStyle = iris;
    ctx.beginPath(); ctx.arc(c, c, 8 + 10 * glow, 0, Math.PI * 2); ctx.fill();
  }

  function mount(doc) {
    const canvas = doc.createElement("canvas");
    canvas.width = canvas.height = SIZE;
    const ctx = canvas.getContext("2d");
    if (!ctx || !canvas.toDataURL) return { update() {}, destroy() {} };
    let link = doc.querySelector('link[rel~="icon"]');
    if (!link) {
      link = doc.createElement("link");
      link.rel = "icon";
      doc.head.appendChild(link);
    }
    link.type = "image/png";
    const still = root.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    let state = {};
    const frame = () => {
      draw(ctx, mode(state), still ? 0 : performance.now() / 1000);
      link.href = canvas.toDataURL("image/png");
    };
    frame();
    const timer = still ? null : setInterval(frame, 1000 / FPS);
    return {
      update(next) { state = next; if (still) frame(); },
      destroy() { if (timer) clearInterval(timer); },
    };
  }

  root.GladosFavicon = { mode, mount };
})(typeof window !== "undefined" ? window : globalThis);
