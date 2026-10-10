/* Portal artwork over the existing console. No second engine connection or audio stream. */
(function (root) {
  'use strict';
  const S2 = 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
  const ICONS = {
    mic: `<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4M9 21h6" ${S2}/>`,
    voice: `<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12" ${S2}/>`,
    camera: `<rect x="19.8" y="2.5" width="2.7" height="8" rx="1"/><path d="M20 6h-5.6l-1.8 3.2" ${S2}/><g transform="rotate(-14 11 14.2)"><path fill-rule="evenodd" d="M3.3 14.2a7.7 5.3 0 1 0 15.4 0a7.7 5.3 0 1 0-15.4 0zM5.5 14.4a3 3 0 1 0 6 0a3 3 0 1 0-6 0z"/><circle cx="8.5" cy="14.4" r="1.35"/></g>`,
    quiet: `<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>`,
  };
  const icons = {microphone:'mic', voice:'voice', camera:'camera', quiet:'quiet'};
  const labels = {microphone:'Microphone', voice:'Voice output', camera:'Camera', quiet:'Quiet mode'};
  function mount(doc) {
    const node = id => doc.getElementById(id);
    const boot = node('portal-loading');
    const reduced = root.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const openedAt = Date.now();
    node('app').inert = true;
    let snapshotReady = false, dismissed = false, finishTimer = null;
    const buttons = [...doc.querySelectorAll('[data-control]')];
    const pair = node('portal-indicator');
    const html = doc.documentElement;
    function dismiss() {
      if (dismissed) return;
      dismissed = true;
      root.clearTimeout(finishTimer);
      root.clearTimeout(timeout);
      boot.hidden = true;
      node('app').inert = false;
      // Do not leave focus on a hidden loading-screen control.
      if (boot.contains(doc.activeElement)) node('surface-switch').focus();
    }
    const timeout = root.setTimeout(() => {
      node('boot-message').textContent = 'The engine is taking its time. You may continue; reconnection runs in the background.';
      dismiss();
    }, 6500);
    node('boot-skip').onclick = dismiss;
    if (new URLSearchParams(root.location?.search || '').get('boot') === '0') dismiss();
    function milestone(id, ready) {
      const label = node(id);
      label.textContent = ready ? 'OK' : 'WAIT';
      label.classList.toggle('ready', ready);
    }
    milestone('boot-interface', true);
    milestone('boot-optic', true);
    buttons.forEach(button => {
      button.innerHTML = '<svg class="pict" viewBox="0 0 24 24" aria-hidden="true">' + ICONS[icons[button.dataset.control]] + '</svg>';
      button.onclick = () => {
        const target = node('control-' + button.dataset.control);
        if (!target.disabled) target.click();
      };
    });
    function activity(state) {
      const view = root.GladosAvatar.presentation(state);
      const active = view.activity.replace(/^Demo · /, '').toLowerCase();
      html.dataset.activity = active;
      pair.textContent = view.activity;
    }
    function update(state) {
      activity(state);
      const linked = state.connection === 'live';
      milestone('boot-snapshot', snapshotReady);
      milestone('boot-telemetry', linked);
      node('boot-progress').style.width = ((2 + Number(snapshotReady) + Number(linked)) * 25) + '%';
      if (!dismissed && linked && snapshotReady && finishTimer === null) {
        node('boot-message').textContent = 'All systems nominal. Yours, I cannot vouch for.';
        finishTimer = root.setTimeout(dismiss, reduced ? 0 : Math.max(0, 1300 - (Date.now() - openedAt)));
      } else if (!dismissed && !linked) {
        root.clearTimeout(finishTimer); finishTimer = null;
        node('boot-message').textContent = state.connection === 'disconnected' ?
          'Engine connection unavailable. Retrying in the background.' : 'Please remain calm. Your cooperation has been noted.';
      }
      buttons.forEach(button => {
        const action = button.dataset.control;
        const target = node('control-' + action);
        const on = target.getAttribute('aria-pressed') === 'true';
        button.disabled = target.disabled;
        button.classList.toggle('on', on);
        button.classList.toggle('off', !on);
        button.setAttribute('aria-pressed', String(on));
        button.title = labels[action] + (target.disabled ? ': unavailable' : on ? ': on' : ': off');
        button.setAttribute('aria-label', button.title);
      });
      const vision = state.visionMind;
      const camera = vision?.camera;
      const available = linked && vision && !vision.paused && camera?.enabled;
      html.dataset.camera = available && camera.connected ? 'live' : available ? 'waiting' : 'off';
      node('basic-camera-status').textContent = !linked ? 'Waiting for the engine connection.' : vision?.error ?
        'Camera error: ' + vision.error : !available ? 'Optical sensors offline.' : camera.error ?
        'Camera error: ' + camera.error : 'Waiting for a camera frame…';
    }
    return {update, activity, snapshot() {snapshotReady = true;}};
  }
  root.GladosPortal = {mount};
})(window);
