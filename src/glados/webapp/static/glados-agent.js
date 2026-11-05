/* Agent panel: actions waiting for the owner's approval, background jobs, and the audit trail.
   Polls even while hidden so the nav badge and browser notifications stay current. */
(function (root) {
  'use strict';
  const POLL_MS = 4000;
  const escape = value => String(value ?? '').replace(/[&<>"']/g,
    char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const time = seconds => new Date(seconds * 1000).toLocaleTimeString();
  const isJob = slot => /^task_(shell|agent)_/.test(slot.slot_id || '');
  const finished = status => ['done', 'error', 'cancelled'].includes(status);

  function outcome(entry) {
    if (entry.status === 'needs_owner_code') return '<span class="chip amber">Waiting for approval</span>';
    if (entry.ok === false) return '<span class="chip red">Failed</span>';
    return '<span class="chip teal">OK</span>';
  }

  function argsText(args) {
    const parts = Object.entries(args || {}).map(([key, value]) =>
      key + '=' + (typeof value === 'string' ? value : JSON.stringify(value)));
    return parts.join(' · ').slice(0, 220);
  }

  const proposalLabel = status => ({working: 'Working', ready: 'Ready: approve to apply', failed_checks: 'Rejected by tests',
    no_change: 'No change', error: 'Failed', cancelled: 'Cancelled', applying: 'Applying…', applied: 'Applied',
    rolled_back: 'Reverted', apply_failed: 'Could not apply'})[status] || status;
  const proposalChip = status => ({ready: 'amber', applied: 'teal', working: '', applying: ''})[status] ?? 'red';

  // Green additions, red removals; the diff text itself is escaped first.
  function colour(pre) {
    if (pre.dataset.coloured === pre.textContent.length + '') return;
    pre.innerHTML = pre.textContent.split('\n').map(line => {
      const text = escape(line);
      if (line.startsWith('+') && !line.startsWith('+++')) return '<span class="diff-add">' + text + '</span>';
      if (line.startsWith('-') && !line.startsWith('---')) return '<span class="diff-del">' + text + '</span>';
      return line.startsWith('@@') || line.startsWith('diff ') ? '<span class="diff-meta">' + text + '</span>' : text;
    }).join('\n');
    pre.dataset.coloured = pre.textContent.length + '';
  }

  function mount(document, connected) {
    const $ = id => document.getElementById(id);
    const setHTML = root.GladosDom.setHTML;
    let visible = false, state = null, jobs = [], seenApprovals = null, jobStatus = null, busy = false;
    const diffs = new Map();
    document.addEventListener('toggle', async event => {
      const id = event.target?.dataset?.proposal;
      if (!id || !event.target.open) return;
      try {
        const response = await root.fetch('/api/agent?proposal=' + encodeURIComponent(id), {cache: 'no-store'});
        const result = await response.json();
        diffs.set(id, response.ok ? (result.diff || 'No diff.') : result.error);
      } catch { diffs.set(id, 'The change could not be loaded.'); }
      render();
    }, true);

    function notify(title, body) {
      try {
        if (root.Notification?.permission === 'granted' && root.localStorage?.getItem('glados-agent-notify') === 'on') {
          new root.Notification(title, {body, tag: 'glados-agent-' + title});
        }
      } catch { /* notifications unavailable */ }
    }

    function announce(approvals, slots) {
      const pending = approvals.filter(a => a.status === 'pending');
      if (seenApprovals !== null) {
        for (const a of pending) if (!seenApprovals.has(a.id)) notify('GLaDOS needs your approval', a.summary);
      }
      seenApprovals = new Set(pending.map(a => a.id));
      const statuses = new Map(slots.map(s => [s.slot_id, s.status]));
      if (jobStatus !== null) {
        for (const s of slots) {
          if (finished(s.status) && jobStatus.get(s.slot_id) && !finished(jobStatus.get(s.slot_id))) {
            notify(s.status === 'done' ? 'Background job finished' : 'Background job ' + s.status, s.title);
          }
        }
      }
      jobStatus = statuses;
      const badge = $('agentBadge');
      if (badge) { badge.textContent = String(pending.length); badge.hidden = !pending.length; }
    }

    function render() {
      if (!visible) return;
      const approvals = state?.approvals || [];
      const pending = approvals.filter(a => a.status === 'pending');
      setHTML($('agent-approvals'), pending.length ? pending.map(a =>
        '<div class="card agent-approval"><p><strong>' + escape(a.summary) + '</strong></p>' +
        '<p class="console-note">Requested ' + escape(time(a.created)) + '. One approval allows one run of this exact ' +
        'action for 3 minutes.</p><div class="console-actions">' +
        '<button type="button" class="btn-secondary" data-approve="' + escape(a.id) + '">Approve</button>' +
        '<button type="button" class="btn-secondary" data-deny="' + escape(a.id) + '">Deny</button></div></div>'
      ).join('') : '<p class="console-note">No action is waiting for approval.</p>');
      const running = jobs.filter(s => !finished(s.status));
      setHTML($('agent-jobs'), jobs.length ? jobs.slice(0, 20).map(s =>
        '<div class="kv"><span>' + escape(s.title) + '<br><span class="console-note">' + escape(s.status) + ' · ' +
        escape((s.summary || '').slice(0, 160)) + '</span></span>' +
        (finished(s.status) ? '' : '<button type="button" class="btn-secondary" data-cancel="' + escape(s.slot_id) +
          '">Cancel</button>') + '</div>'
      ).join('') : '<p class="console-note">No background jobs.</p>');
      $('agent-jobs-count').textContent = String(running.length);
      const recent = state?.recent || [];
      setHTML($('agent-recent'), recent.length ? '<table class="agent-log"><thead><tr><th>Time</th><th>By</th>' +
        '<th>Tool</th><th>Details</th><th>Result</th></tr></thead><tbody>' + recent.map(e =>
        '<tr><td>' + escape(time(e.time)) + '</td><td>' + escape(e.source === 'worker' ? 'Worker' : 'GLaDOS') +
        '</td><td>' + escape(e.tool) + '</td><td>' + escape(argsText(e.args)) +
        (e.error ? '<br><span class="console-note">' + escape(e.error) + '</span>' : '') + '</td><td>' + outcome(e) +
        '</td></tr>').join('') + '</tbody></table>' : '<p class="console-note">No agent actions yet.</p>');
      const proposals = state?.proposals || [];
      setHTML($('agent-proposals'), proposals.length ? proposals.map(p =>
        '<details class="card agent-proposal" data-proposal="' + escape(p.id) + '"><summary><strong>' +
        escape(p.request) + '</strong> <span class="chip ' + proposalChip(p.status) + '">' +
        escape(proposalLabel(p.status)) + '</span></summary>' +
        (p.files ? '<p class="console-note">' + escape(p.files) + '</p>' : '') +
        '<p class="agent-summary">' + escape(p.summary || '') + '</p>' +
        '<pre class="agent-diff" data-diff-for="' + escape(p.id) + '">' + escape(diffs.get(p.id) ?? 'Loading the change…') +
        '</pre></details>'
      ).join('') : '<p class="console-note">No proposals. Ask GLaDOS to improve her code or change a model.</p>');
      for (const pre of document.querySelectorAll('[data-diff-for]')) colour(pre);
      const toggle = $('agent-notify');
      const permission = root.Notification?.permission;
      const on = permission === 'granted' && root.localStorage?.getItem('glados-agent-notify') === 'on';
      toggle.textContent = !root.Notification ? 'Notifications unavailable' : on ? 'Notifications: on' : 'Notifications: off';
      toggle.disabled = !root.Notification || permission === 'denied';
    }

    async function poll() {
      if (!connected() || busy) return;
      try {
        const [agent, slots] = await Promise.all([
          root.fetch('/api/agent', {cache: 'no-store'}).then(r => r.json()),
          root.fetch('/api/slots', {cache: 'no-store'}).then(r => r.json()),
        ]);
        if (agent.available === false) return;
        state = agent;
        jobs = (slots.slots || []).filter(isJob).reverse();
        announce(agent.approvals || [], jobs);
        render();
      } catch { /* the engine is reconnecting */ }
    }

    async function post(path, body, done) {
      busy = true;
      $('agent-feedback').textContent = 'Working…';
      try {
        const response = await root.fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'The engine rejected the request.');
        $('agent-feedback').textContent = done;
      } catch (error) {
        $('agent-feedback').textContent = error.message;
      } finally { busy = false; poll(); }
    }

    document.addEventListener('click', event => {
      const target = event.target.closest?.('[data-approve],[data-deny],[data-cancel]');
      if (!target || !$('view-agent')?.contains(target)) return;
      if (target.dataset.approve) post('/api/agent/approve', {id: target.dataset.approve, approve: true},
        'Approved. Ask GLaDOS to go ahead.');
      else if (target.dataset.deny) post('/api/agent/approve', {id: target.dataset.deny, approve: false}, 'Denied.');
      else post('/api/tasks/cancel', {slot_id: target.dataset.cancel}, 'Cancellation requested');
    });
    $('agent-notify').onclick = async () => {
      if (!root.Notification) return;
      const on = root.Notification.permission === 'granted' && root.localStorage?.getItem('glados-agent-notify') === 'on';
      if (on) { root.localStorage.setItem('glados-agent-notify', 'off'); return render(); }
      const permission = root.Notification.permission === 'granted' ? 'granted' : await root.Notification.requestPermission();
      if (permission === 'granted') root.localStorage.setItem('glados-agent-notify', 'on');
      render();
    };
    root.setInterval(poll, POLL_MS);
    poll();
    return {setVisible(value) { visible = value; if (value) { render(); poll(); } }};
  }

  root.GladosAgent = {mount};
})(window);
