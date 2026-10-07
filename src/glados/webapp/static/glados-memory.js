/* Aperture Science Test Chamber: saved memory and current-topic recall. */
(function (root) {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g,
    char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  function renderFacts(facts, empty = 'No saved memories match this view.') {
    if (!facts?.length) return '<p class="context-empty">' + escape(empty) + '</p>';
    return facts.map(fact => {
      const date = new Date(fact.created_at * 1000);
      return '<article class="card memory-fact"><div class="context-message-head"><span class="chip">' +
        escape(fact.kind === 'summary' ? 'Summary' : 'Fact') + '</span><span>' + escape(fact.source) +
        '</span><span>' + escape(Number.isFinite(date.getTime()) ? date.toLocaleString() : 'Date unavailable') +
        '</span>' + (fact.excerpt ? '<span class="chip">Excerpt</span>' : '') +
        '</div><p class="memory-content">' + escape(fact.content) + '</p>' +
        (fact.id && fact.revision ? '<div class="memory-actions" data-memory-id="' + escape(fact.id) +
          '"><small>' + escape(fact.id) + '</small> <button data-memory-edit>Edit</button> ' +
          '<button data-memory-delete>Delete</button></div>' : '') + '</article>';
    }).join('');
  }
  function mount(document, connected) {
    const $ = id => document.getElementById(id);
    let visible = false, pending = false, generation = 0, offset = 0, editing = false;
    const records = new Map();
    async function refresh() {
      if (!visible || document.hidden || pending || editing) return;
      if (!connected()) {
        $('memory-status').textContent = 'Engine disconnected. Displayed memory may be stale.';
        return;
      }
      const requestGeneration = generation;
      pending = true; $('memory-refresh').disabled = true;
      try {
        const url = '/api/memory?query=' + encodeURIComponent($('memory-search').value || '') +
          '&kind=' + encodeURIComponent($('memory-kind').value || 'all') + '&offset=' + offset + '&limit=30';
        const response = await root.fetch(url, {cache:'no-store'});
        const value = await response.json();
        if (requestGeneration !== generation || !visible || editing) return;
        if (!response.ok) throw new Error(value.error || 'Memory could not be loaded.');
        if (!value.available) {
          $('memory-status').textContent = value.reason;
          $('memory-facts').innerHTML = renderFacts([], value.reason);
          $('memory-recall').innerHTML = ''; $('memory-topic').textContent = 'Recall unavailable.';
          $('memory-prev').disabled = true; $('memory-next').disabled = true;
          return;
        }
        if (offset > 0 && offset >= value.total) {
          offset = Math.max(0, Math.floor((value.total - 1) / 30) * 30);
          generation++; return;
        }
        const recall = value.recall || {};
        records.clear();
        for (const fact of [...(recall.facts || []), ...value.memories]) records.set(fact.id, fact);
        $('memory-recall').innerHTML = renderFacts(recall.facts, recall.paused ?
          'The Memory Core is paused.' : 'No saved facts match the current topic.');
        $('memory-topic').textContent = recall.enabled ? (recall.query ?
          'Current topic: ' + recall.query : 'Waiting for a conversation topic.') : 'Recall is disabled.';
        $('memory-facts').innerHTML = renderFacts(value.memories);
        $('memory-prev').disabled = offset === 0;
        $('memory-next').disabled = offset + value.memories.length >= value.total;
        $('memory-status').textContent = value.total ?
          'Showing ' + (offset + 1) + '–' + (offset + value.memories.length) + ' of ' + value.total + ' saved memories.' :
          'No saved memories in this view.';
        if (value.limited) $('memory-status').textContent += ' The index includes the newest memories within its size limit.';
        if (value.unavailable) $('memory-status').textContent += ' Some saved memory could not be read.';
      } catch (error) {
        if (requestGeneration === generation && visible) $('memory-status').textContent = error.message;
      } finally {
        pending = false; $('memory-refresh').disabled = false;
        if (requestGeneration !== generation && visible) refresh();
      }
    }
    async function memoryAction(event) {
      const button = event.target.closest('button');
      const host = button?.closest('[data-memory-id]');
      if (!host) return;
      let fact = records.get(host.dataset.memoryId);
      if (!fact) return;
      if (button.hasAttribute('data-memory-edit') || button.hasAttribute('data-memory-delete')) {
        editing = true;
        try {
          const response = await root.fetch('/api/memory?id=' + encodeURIComponent(fact.id), {cache:'no-store'});
          const full = await response.json();
          if (!response.ok) throw new Error(full.error || 'Memory could not be read');
          fact = full; records.set(fact.id, fact);
        } catch (error) {
          editing = false; $('memory-status').textContent = error.message; return;
        }
        const deleting = button.hasAttribute('data-memory-delete');
        host.innerHTML = deleting ? '<p>Delete this saved memory?</p>' :
          '<label>Memory text<textarea class="console-field" data-memory-content></textarea></label>';
        if (!deleting) host.querySelector('textarea').value = fact.content;
        host.insertAdjacentHTML('beforeend', '<button data-memory-save="' + (deleting ? 'delete' : 'edit') +
          '">Confirm ' + (deleting ? 'delete' : 'edit') + '</button> <button data-memory-abort>Cancel</button>');
      } else if (button.hasAttribute('data-memory-abort')) {
        editing = false; refresh();
      } else if (button.hasAttribute('data-memory-save')) {
        button.disabled = true;
        try {
          const response = await root.fetch('/api/memory/edit', {method:'POST',
            headers:{'Content-Type':'application/json'}, body:JSON.stringify({
              id:fact.id, revision:fact.revision, action:button.dataset.memorySave,
              content:host.querySelector('textarea')?.value})});
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || 'Memory change failed');
          editing = false; generation++; refresh();
        } catch (error) {
          $('memory-status').textContent = error.message; button.disabled = false;
        }
      }
    }
    $('memory-facts').addEventListener('click', memoryAction);
    $('memory-recall').addEventListener('click', memoryAction);
    function change() { offset = 0; generation++; refresh(); }
    $('memory-filter').onsubmit = event => { event.preventDefault(); change(); };
    $('memory-kind').onchange = change;
    $('memory-refresh').onclick = refresh;
    $('memory-prev').onclick = () => { offset = Math.max(0, offset - 30); generation++; refresh(); };
    $('memory-next').onclick = () => { offset += 30; generation++; refresh(); };
    const timer = root.setInterval(() => { if (visible) refresh(); }, 5000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden && visible) refresh(); });
    root.addEventListener('pagehide', () => { visible = false; generation++; root.clearInterval(timer); });
    return {refresh, setVisible(value) { visible = value; generation++; if (visible) refresh(); }};
  }
  root.GladosMemory = {mount, renderFacts};
})(window);
