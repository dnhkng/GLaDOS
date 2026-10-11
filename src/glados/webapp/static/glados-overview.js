/* Read-only overview over the console's existing state and observability stream. */
(function (root) {
  'use strict';
  const age = (seconds) => seconds < 60 ? Math.floor(seconds)+'s' : seconds < 3600 ? Math.floor(seconds/60)+'m' : Math.floor(seconds/3600)+'h';
  function mount(doc, helpers) {
    const {escape:esc, coreOwner, cores, identity, activity} = helpers;
    const node = id => doc.getElementById(id);
    let latest = null, state = {}, input = 'No input received yet.', routingKey = '';
    function timing() {
      const host = node('turn-waterfall'), summary = node('turn-latency');
      if (!latest) {host.textContent='Timing begins with the next user input.'; return;}
      const points = latest.points;
      const end = points.at(-1)?.t ?? latest.t;
      const duration = Math.max(1,end-latest.t);
      summary.textContent = latest.finished ? (points.length?((end-latest.t)/1000).toFixed(2)+' s · ':'')+latest.finished : state.connection==='disconnected'?'Telemetry disconnected':'Turn in progress';
      host.innerHTML = points.map(p => {
        const offset = Math.max(0,p.t-latest.t);
        return '<div class="timing-label">'+esc(p.label)+'</div><div class="timing-track"><span class="timing-point '+esc(p.kind)+'" style="left:'+Math.min(99,offset/duration*100)+'%"></span></div><div class="timing-value">'+Math.round(offset)+' ms</div>';
      }).join('') || '<p class="console-note">Input accepted; waiting for a recorded response event.</p>';
    }
    function event(ev) {
      const meta = ev.meta || {};
      if (ev.kind === 'user_input') {
        input = ev.src === 'audio' ? 'Voice input · '+(state.controls?.user_transcripts?'transcript pending':'no transcript kept') : ev.msg;
        latest = {t:ev.t,generation:meta.generation,points:[],finished:meta.generation == null?'timing unavailable':null};
        node('latest-input').textContent=input;
      } else if (ev.src === 'asr' && ev.kind === 'transcript' && latest && meta.generation === latest.generation) {
        input=ev.msg; node('latest-input').textContent=input;
      }
      if (!latest || latest.finished || meta.autonomy || meta.generation == null || meta.generation !== latest.generation || ev.t < latest.t) {timing(); return;}
      const labels = {
        'llm.routed':['Routing completed','route'],
        'llm.admitted':['Reply inference admitted','llm'],
        'llm.first_token':['First response token','llm'],
        'tts.synthesize':['First clause to TTS','tts'],
        'tts.ready':['First audio ready','tts'],
        'tts.play':['Playback requested','audio'],
      };
      const key=ev.src+'.'+ev.kind, label=labels[key];
      if (label && (meta.lane == null || meta.lane === 'priority') && !latest.points.some(p=>p.key===key)) {
        latest.points.push({key,label:label[0],kind:label[1],t:ev.t});
        latest.points.sort((a,b)=>a.t-b.t);
        if (key === 'tts.play') latest.finished=meta.muted?'text delivered (voice muted)':'playback requested';
        if (key === 'llm.routed' && ['ignore','quiet'].includes(meta.action)) latest.finished='no reply selected';
      }
      timing();
    }
    function renderRouting() {
      const decision = state.routing?.latest;
      const host=node('overview-route');
      const key=JSON.stringify(decision ?? null);
      if (key===routingKey) return; // Keep disclosures open across telemetry refreshes.
      routingKey=key;
      if (!decision) {host.textContent='No routing decision yet.'; return;}
      const title=o=>{
        if (!o) return 'Selected';
        try {const value=JSON.parse(o.description); if (value.tool || value.server) return value.tool || value.server;} catch (_) {}
        const name=String(o.id && !/^[a-f0-9]{32}$/i.test(o.id)?o.id:helpers.describe(o) || 'Selected').replace(/^(area_|command_)/,'').replace(/[_-]/g,' ');
        return name.length>40?name.slice(0,37)+'…':name.charAt(0).toUpperCase()+name.slice(1);
      };
      const stages=decision.stages || [{name:'Decision',scores:decision.scores,accepted:decision.accepted}];
      host.innerHTML=stages.map(s => {
        const scores=[...(s.scores || [])].sort((a,b)=>b.probability-a.probability);
        const winner=scores[0];
        const chosen=scores.find(o=>o.id===s.option_id) || winner;
        const choices=scores.map(o=>'<p><b>'+esc(o.label || title(o))+' · '+(o.probability*100).toFixed(1)+'%</b> '+esc(helpers.describe(o) || '')+'</p>').join('');
        return '<div class="decision-step"><div class="slice">'+esc(s.name || 'Decision')+'</div><strong>'+esc(s.fallback_selected || !s.accepted?'Fallback':title(chosen))+'</strong><p class="console-note">'+(winner?(winner.probability*100).toFixed(1)+'% preference · lead '+((s.margin || 0)*100).toFixed(1)+'%':'')+'</p>'+(choices?'<details class="route-details"><summary>All choices</summary>'+choices+'</details>':'')+'</div>';
      }).join('')+'<div class="decision-step decision-action"><div class="slice">'+(decision.dry_run?'Preview only':'Action')+'</div><strong>'+esc(decision.tool || decision.action)+'</strong>'+(decision.reason?'<details class="route-details"><summary>Reason</summary><p>'+esc(decision.reason)+'</p></details>':'')+'</div>';
    }
    function renderCapacity() {
      const inference=state.inference;
      if (!inference) {node('overview-slots').textContent='Capacity unavailable.'; node('overview-hold').textContent=''; node('overview-queue').textContent=''; return;}
      node('overview-slots').innerHTML=Array.from({length:inference.capacity},(_,i)=>{
        const request=inference.active.find(r=>r.slot===i);
        return '<div class="capacity-tile '+(request?'busy '+(request.lane==='priority'?'interactive':'background'):'')+'"><div class="capacity-number">'+String(i+1).padStart(2,'0')+'</div><strong>'+esc(request?coreOwner(request.owner):'Idle')+'</strong><p class="console-note">'+(request?esc(request.lane)+' · '+Math.max(0,Date.now()/1000-request.started_at).toFixed(1)+' s':i<inference.reserved_interactive?'Interactive reservation':'Shared capacity')+'</p></div>';
      }).join('');
      const status=inference.interaction_hold?'User interaction active · new background inference paused. Running requests may finish.':inference.reserved_interactive+' reserved for interaction · '+inference.active.length+' / '+inference.capacity+' active';
      node('overview-hold').textContent=(state.connection==='disconnected'?'Disconnected · last known state: ':'')+status;
      node('overview-queue').textContent=inference.waiting.length?'Waiting: '+inference.waiting.map(r=>coreOwner(r.owner)+' ('+(r.wait_reason==='user_response'?'user response':'capacity')+')').join(', '):'No requests waiting.';
    }
    function renderCores() {
      const reports=[...(state.slots || [])].sort((a,b)=>b.updated_at-a.updated_at);
      const attention=state.controls?.autonomy_enabled===false?' · important update':' · awaiting Autonomy';
      node('overview-cores').innerHTML=cores().map(core=>{
        const status=activity(core);
        const slot=reports.find(s=>s.owner_id===(core.id || core.agent_id));
        const important=slot?.update_priority==='important' && !slot.handled;
        const error=status.label==='Error' || slot?.status==='error';
        const updated=slot?.updated_at?'<small>'+age(Math.max(0,Date.now()/1000-slot.updated_at))+' ago'+(error?' · report failed':'')+(important?attention:'')+'</small>':'';
        return '<tr class="'+(error?'core-error':important?'core-important':'')+'"><td>'+esc(identity(core).title)+'</td><td><span class="status-pill '+esc(status.className || '')+'">'+esc(status.label)+'</span></td><td><p>'+esc(slot?.summary || core.summary || status.detail || 'No update yet.')+'</p>'+updated+'</td></tr>';
      }).join('') || '<tr><td colspan="3">No core state available.</td></tr>';
      const pending=s=>s.update_priority==='important' && !s.handled;
      const slots=[...(state.slots || [])].filter(s=>s.summary).sort((a,b)=>Number(pending(b))-Number(pending(a)) || b.updated_at-a.updated_at).slice(0,3);
      node('presence-thoughts').innerHTML=slots.map(s=>'<div class="thought '+(pending(s)?'important':'')+'"><span>'+esc(s.title || coreOwner(s.owner_id))+'</span><p>'+esc(s.summary)+'</p><small>'+esc(s.status)+(s.queue_position!=null?' · queue '+s.queue_position:'')+' · '+age(Math.max(0,Date.now()/1000-s.updated_at))+' ago'+(pending(s)?attention:'')+'</small></div>').join('') || '<p class="console-note">No core reports yet.</p>';
    }
    function update(next) {
      state=next;
      const release=state.inference?.last_interaction_release;
      if (latest && !latest.finished && release && release.generation===latest.generation && release.reason==='no_response') latest.finished='no response recorded';
      refreshActivity(state);
      const view=root.GladosAvatar.presentation(state);
      const vision=state.visionMind;
      node('vision-caption').textContent=vision?.error?'Vision error: '+vision.error:vision?.scene || (view.cameraOn?'Waiting for an observation.':'Camera off.');
      node('vision-caption-meta').textContent='Vision Core'+(vision?.inference_ms!=null?' · '+Math.round(vision.inference_ms)+' ms':'')+(vision?.captured_at?' · '+age(Math.max(0,Date.now()/1000-vision.captured_at))+' ago':'')+(vision?.paused?' · paused':'')+(state.connection==='disconnected'?' · last known observation':'');
      const autonomy=state.autonomyCore, last=autonomy?.last_decision;
      node('overview-autonomy').textContent=autonomy?(autonomy.status+' · '+autonomy.pending_updates+' pending updates'+(last?' · '+last.outcome+': '+last.reason:'')):'Autonomy Core unavailable.';
      node('overview-handoff').textContent=last?.instruction || 'No pending handoff to Central Core.';
      renderRouting(); renderCapacity(); renderCores(); timing();
    }
    function refreshActivity(next) {
      const view=root.GladosAvatar.presentation(next);
      node('avatar-card').dataset.activity=view.activity.toLowerCase();
      if (node('presence-activity').textContent!==view.activity) node('presence-activity').textContent=view.activity;
      if (node('presence-detail').textContent!==view.detail) node('presence-detail').textContent=view.detail;
    }
    return {update,event,activity:refreshActivity};
  }
  root.GladosOverview={mount};
})(window);
