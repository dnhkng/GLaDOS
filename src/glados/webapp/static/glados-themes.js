/* Local preferences shared by the console and presence surface. */
(function (root) {
  'use strict';
  const choices = {dark:'Dark', white:'Clinical', terminal:'Terminal', potato:'PotatOS'};
  const palettes = {
    dark:{ink:'#E8ECEF',bg:'#0B0E12',grad:'#1A2029',optic:'thermal',opticInk:'#E8ECEF',opticBg:'#0B0E12'},
    white:{ink:'#15171A',bg:'#E4E6E8',grad:'#F7F7F5',optic:'whitehot',opticInk:'#F2F2F0',opticBg:'#121214'},
    terminal:{ink:'#FFB000',bg:'#0B0800',grad:'#1D1400',optic:'ascii',opticInk:'#FFB000',opticBg:'#0B0800'},
    potato:{ink:'#E8D6B0',bg:'#1D130A',grad:'#3A2814',optic:'dither',opticInk:'#9BBC0F',opticBg:'#0F380F'},
  };
  function preference(key) { try {return root.localStorage?.getItem(key);} catch {return null;} }
  const params = new URLSearchParams(root.location?.search || '');
  const requested = params.get('theme') || preference('glados-theme');
  const initial = Object.hasOwn(choices, requested) ? requested : 'dark';
  let surface = ['console','presence'].includes(params.get('surface')) ? params.get('surface') : 'basic';
  root.document.documentElement.dataset.theme = initial;
  root.document.documentElement.dataset.surface = surface;
  function mount(doc, changed) {
    const host = doc.getElementById('theme-choice');
    const link = doc.getElementById('surface-switch');
    const presenceLink = doc.getElementById('presence-switch');
    function arrangeInput() {
      const input = doc.getElementById('conversation-input');
      if (surface === 'presence') doc.getElementById('presence-controls').prepend(input);
      else doc.getElementById('overview-layout').insertBefore(input, doc.getElementById('overview-turn-card'));
    }
    function apply(name) {
      doc.documentElement.dataset.theme = name;
      host.value = name;
      const base = root.location?.protocol === 'file:' ? 'index.html' : '/';
      const destination = surface === 'basic' ? 'console' : 'basic';
      link.textContent = surface === 'basic' ? 'Detailed view ↗' : 'Basic view ↗';
      link.href = base + '?surface=' + destination + '&theme=' + name;
      if (presenceLink) presenceLink.href = base + '?surface=presence&theme=' + name;
    }
    host.innerHTML = Object.entries(choices).map(([value,label]) => `<option value="${value}">${label}</option>`).join('');
    apply(initial);
    arrangeInput();
    link.onclick = event => {
      if (event.button > 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      surface = surface === 'basic' ? 'console' : 'basic';
      doc.documentElement.dataset.surface = surface;
      const url = new URLSearchParams(root.location?.search || '');
      url.set('surface', surface); url.set('theme', host.value);
      root.history?.replaceState(null, '', '?' + url.toString());
      apply(host.value);
      arrangeInput();
      root.go?.('home');
      changed();
    };
    host.onchange = () => {
      if (!Object.hasOwn(choices, host.value)) return;
      apply(host.value);
      try {root.localStorage?.setItem('glados-theme', host.value);} catch { /* Private storage may be disabled. */ }
      changed();
    };
    const settings = doc.getElementById('theme-settings');
    if (settings) {
      settings.innerHTML = host.innerHTML;
      settings.value = initial;
      settings.onchange = () => {host.value = settings.value; host.onchange();};
      const update = host.onchange;
      host.onchange = () => {update(); settings.value = host.value;};
    }
  }
  root.GladosThemes = {mount, current:() => palettes[root.document.documentElement.dataset.theme] || palettes.dark};
})(window);
