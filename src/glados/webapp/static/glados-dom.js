/* Shared DOM helper for live views that re-render on every telemetry update. */
(function (root) {
  "use strict";

  const written = new WeakMap();

  // Browsers normalise innerHTML (entities, quotes), so remember both the source string and the DOM
  // it produced: unchanged content is left alone even if other code edited the element in between.
  // When content does change, keep scroll positions and open <details> so readers are not thrown back.
  function setHTML(host, html) {
    if (!host) return false;
    const last = written.get(host);
    if (last && last.html === html && last.dom === host.innerHTML) return false;
    const scrollers = [host, ...Array.from(host.querySelectorAll?.("[id]") || [])]
      .filter(el => el.scrollTop || el.scrollLeft)
      .map(el => [el === host ? null : el.id, el.scrollTop, el.scrollLeft]);
    const open = Array.from(host.querySelectorAll?.("details") || []).map(d => d.open);
    host.innerHTML = html;
    Array.from(host.querySelectorAll?.("details") || []).forEach((d, i) => { if (i < open.length) d.open = open[i]; });
    for (const [id, top, left] of scrollers) {
      const el = id === null ? host : host.querySelector?.("#" + (root.CSS?.escape ? root.CSS.escape(id) : id));
      if (el) { el.scrollTop = top; el.scrollLeft = left; }
    }
    written.set(host, { html, dom: host.innerHTML });
    return true;
  }

  root.GladosDom = { setHTML };
})(typeof window !== "undefined" ? window : globalThis);
