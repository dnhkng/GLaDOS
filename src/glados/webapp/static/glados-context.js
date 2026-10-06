/* Aperture Science Neural Buffer: read-only inspection of real inference inputs. */
(function (root) {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g,
    char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const text = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);

  function renderSections(data, search = '', open = null) {
    const query = search.trim().toLowerCase();
    const sections = (data.sections || []).map((section, index) => ({...section, key:section.source + ':' + index}));
    const messages = sections.filter(section => !query || text(section).toLowerCase().includes(query)).map(section => {
      const count = section.messages.length + (section.messages.length === 1 ? ' message' : ' messages');
      const first = section.messages[0]?.index, last = section.messages.at(-1)?.index;
      const range = first ? '#' + first + (last !== first ? '–#' + last : '') : '';
      const expanded = open ? open.has(section.key) : section.source === 'input';
      const rows = section.messages.map(message => {
        const {index, role, content, ...extra} = message;
        return '<article class="context-message"><div class="context-message-head">' +
          (index ? '<span class="chip">#' + escape(index) + '</span>' : '') + '<strong>' + escape(role) + '</strong></div>' +
          '<pre>' + escape(text(content ?? '')) + '</pre>' +
          (Object.keys(extra).length ? '<pre>' + escape(text(extra)) + '</pre>' : '') + '</article>';
      }).join('');
      return '<details class="card context-section" data-context-key="' + escape(section.key) + '"' + (expanded ? ' open' : '') +
        '><summary><span class="chip">' + escape(range) + '</span><span class="context-section-copy"><strong>' +
        escape(section.title) + '</strong><span class="context-purpose">' + escape(section.description || '') +
        '</span></span><span class="console-note">' + escape(count) +
        '</span></summary><div class="context-messages">' + rows + '</div></details>';
    }).join('');
    const tools = data.tools || [], preview = data.kind === 'preview';
    const toolTitle = preview ? 'Available tool catalogue · preview only' : 'Tools offered in this request';
    const toolNote = preview ? 'This catalogue is not a message sent to GLaDOS. Routing selects the offered subset for the next request.' :
      'These definitions were sent in the separate tools field. They have no message number; their placement in the model template depends on the backend.';
    const showTools = tools.length && (!query || text([toolTitle, toolNote, tools]).toLowerCase().includes(query));
    const toolSection = showTools ? '<section class="context-tools"><h2>Tools · separate from message order</h2>' +
      '<details class="card context-section" data-context-key="tools"' + (open?.has('tools') ? ' open' : '') +
      '><summary><span class="context-section-copy"><strong>' + escape(toolTitle) +
      '</strong><span class="context-purpose">' + escape(toolNote) + '</span></span><span class="console-note">' +
      tools.length + ' tools</span></summary><div class="context-messages"><pre>' + escape(text(tools)) +
      '</pre></div></details></section>' : '';
    return messages + toolSection || '<p class="context-empty">No context sections match this search.</p>';
  }

  function mount(document, connected) {
    const $ = id => document.getElementById(id);
    let visible = false, updating = true, pending = false, generation = 0, data = null, rendered = '';
    function render(force = false, openOverride = undefined) {
      if (!data?.available) return;
      const key = JSON.stringify([data.sections, data.tools, $('context-search').value || '']);
      if (!force && key === rendered) return;
      const open = openOverride !== undefined ? openOverride : rendered ?
        new Set(Array.from($('context-sections').querySelectorAll('details[open]')).map(el => el.dataset.contextKey)) : null;
      $('context-sections').innerHTML = renderSections(data, $('context-search').value || '', open);
      rendered = key;
      $('context-overview').innerHTML = '<span class="chip teal">' + escape(data.model) + '</span>' +
        '<span class="chip">' + data.message_count + ' messages</span><span class="chip">' +
        data.text_characters.toLocaleString() + ' text characters</span><span class="chip">' +
        data.tools.length + (data.kind === 'preview' ? ' available tools (preview)' : ' offered tools') + '</span>';
      $('context-export').disabled = false;
    }
    async function refresh() {
      if (!visible || document.hidden || pending) return;
      if (!connected()) {
        $('context-status').textContent = 'Engine disconnected. Displayed context may be stale.';
        return;
      }
      const requestGeneration = generation;
      const mode = $('context-mode').value || 'user', view = $('context-view').value || 'live';
      pending = true; $('context-refresh').disabled = true;
      try {
        const response = await root.fetch('/api/context?mode=' + encodeURIComponent(mode) + '&view=' + encodeURIComponent(view), {cache:'no-store'});
        const value = await response.json();
        if (requestGeneration !== generation || !visible) return;
        if (!response.ok) throw new Error(value.error || 'Context could not be loaded.');
        data = value;
        if (!value.available) {
          $('context-sections').innerHTML = '<p class="context-empty">' + escape(value.reason) + '</p>';
          $('context-overview').innerHTML = ''; $('context-export').disabled = true; rendered = '';
          $('context-status').textContent = value.reason;
          return;
        }
        render();
        $('context-status').textContent = (view === 'request' ? 'Submitted ' : 'Preview updated ') +
          new Date(data.captured_at * 1000).toLocaleTimeString() +
          '. Numbered messages follow the API order; tool definitions are a separate field. Audio/image bytes are omitted from inspection.';
      } catch (error) {
        if (requestGeneration === generation && visible) $('context-status').textContent = error.message + ' Displayed context may be stale.';
      } finally {
        pending = false; $('context-refresh').disabled = false;
        if (requestGeneration !== generation && visible) refresh();
      }
    }
    function changeSource() {
      generation++; data = null; rendered = '';
      $('context-sections').innerHTML = ''; $('context-overview').innerHTML = ''; $('context-export').disabled = true;
      $('context-status').textContent = 'Loading context…'; refresh();
    }
    $('context-mode').onchange = changeSource;
    $('context-view').onchange = changeSource;
    $('context-refresh').onclick = refresh;
    $('context-search').oninput = () => render(true);
    $('context-expand').onclick = () => {
      if (!data?.available) return;
      render(true, new Set([...data.sections.map((section, index) => section.source + ':' + index), 'tools']));
    };
    $('context-collapse').onclick = () => render(true, new Set());
    $('context-live').onclick = () => {
      updating = !updating;
      $('context-live').textContent = 'Live updates: ' + (updating ? 'ON' : 'OFF');
      $('context-live').setAttribute('aria-pressed', String(updating));
      if (updating) refresh();
    };
    $('context-export').onclick = () => {
      if (!data?.available) return;
      const url = root.URL.createObjectURL(new root.Blob([JSON.stringify(data, null, 2)], {type:'application/json'}));
      const link = document.createElement('a'); link.href = url; link.download = 'glados-' + data.mode + '-context.json';
      document.body.appendChild(link); link.click(); link.remove(); root.URL.revokeObjectURL(url);
    };
    const timer = root.setInterval(() => { if (visible && updating) refresh(); }, 2000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden && visible && updating) refresh(); });
    root.addEventListener('pagehide', () => { visible = false; generation++; root.clearInterval(timer); });
    return {refresh, setVisible(value) { visible = value; generation++; if (visible) refresh(); }};
  }
  root.GladosContext = {mount, renderSections};
})(window);
