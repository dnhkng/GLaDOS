// Exercise the shipped browser script with real server JSON, without browser dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class CanvasContext {
  constructor(){this.calls=[];this.intensity=40;}
  createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};}
  getImageData(x,y,w,h){
    const image=this.createImageData(w,h);image.data.fill(this.intensity);
    if(this.sample)for(let py=0;py<h;py++)for(let px=0;px<w;px++){
      const value=this.sample(px,py,w,h),i=(py*w+px)*4;
      image.data[i]=value;image.data[i+1]=value;image.data[i+2]=value;
    }
    return image;
  }
  createLinearGradient(){return {addColorStop(){}};}
  createRadialGradient(){return {addColorStop(){}};}
}
for(const method of ['save','restore','drawImage','translate','rotate','scale','fillRect','clearRect',
  'putImageData','beginPath','arc','moveTo','lineTo','closePath','fill','stroke','fillText']) {
  CanvasContext.prototype[method]=function(...args){this.calls.push([method,...args]);if(this.calls.length>10000)this.calls=[];};
}

class Element {
  constructor() {
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.textContent = "";
    this.listeners = {};
    this.attributes = {};
    this.checked = true;
  }
  getContext(){return this.context ||= new CanvasContext();}
  set innerHTML(html) { this.htmlWrites=(this.htmlWrites || 0)+1; this.html = html; this.children = []; }
  get innerHTML() { return this.html || ""; }
  get lastChild() { return this.children.at(-1); }
  appendChild(child) { child.parent = this; this.children.push(child); }
  prepend(child) { child.parent = this; this.children.unshift(child); }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
  remove() { this.parent.removeChild(this); }
  querySelectorAll() { return []; }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  removeEventListener(type) { delete this.listeners[type]; }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; if(name==='src') this.src=''; }
  getBoundingClientRect() { return {left:0, top:0, width:400, height:400}; }
}

(async () => {
  const themeScript=fs.readFileSync(path.join(__dirname,'../src/glados/webapp/static/glados-themes.js'),'utf8');
  for (const [search,stored,theme,surface] of [
    ['', 'potato', 'potato', 'console'],
    ['?surface=presence&theme=white', 'terminal', 'white', 'presence'],
    ['?surface=invalid&theme=invalid', null, 'dark', 'console'],
  ]) {
    const bootDoc={documentElement:new Element()},bootNodes=new Map();
    bootDoc.getElementById=id=>{if(!bootNodes.has(id))bootNodes.set(id,new Element());return bootNodes.get(id);};
    const boot=vm.createContext({URLSearchParams,document:bootDoc,location:{search},
      localStorage:{getItem(){if(stored===null)throw Error('Storage unavailable');return stored;},setItem(){throw Error('Storage unavailable');}}});
    vm.runInContext('window=globalThis',boot);vm.runInContext(themeScript,boot);
    boot.GladosThemes.mount(bootDoc,()=>{});
    assert.equal(bootDoc.documentElement.dataset.theme,theme);
    assert.equal(bootDoc.documentElement.dataset.surface,surface);
    assert.match(bootDoc.getElementById('surface-switch').href,new RegExp('surface='+(surface==='presence'?'console':'presence')));
    bootDoc.getElementById('theme-choice').value='terminal';bootDoc.getElementById('theme-choice').onchange();
    assert.equal(bootDoc.documentElement.dataset.theme,'terminal','Disabled storage never blocks theme switching');
  }
  const snapshot = JSON.parse(fs.readFileSync(0, "utf8"));
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, new Element());
    return nodes.get(id);
  };
  let stream;
  const posts=[];
  const contextGets=[];
  const memoryGets=[];
  let memoryPayload={available:true,total:31,indexed_facts:31,offset:0,limit:30,memories:[
    {id:'test',kind:'fact',content:'User prefers brief project updates.',source:'user',created_at:1}],
    recall:{enabled:true,query:'Project updates',facts:[{id:'test',content:'User prefers brief project updates.',source:'user',created_at:1}]}};
  let contextPayload={available:true,kind:'preview',mode:'user',captured_at:1,model:'E4B',message_count:2,
    text_characters:40,tools:[{type:'function',function:{name:'get_time',parameters:{}}}],sections:[
      {source:'emotion',title:'Emotion Core',messages:[{index:1,role:'system',content:'P=-0.6; use an irritated tone.'}]},
      {source:'vision',title:'Vision Core',messages:[{index:2,role:'system',content:'A jacket with a zipper.'}]}]};
  const devicePayload={audio:{available:true,input:[{id:1,name:'USB Microphone',host_api:'ALSA'}],output:[{id:2,name:'Speakers',host_api:'ALSA'}],selected_input:null,selected_output:null},
    camera:{available:true,devices:[{id:'/dev/video0',name:'Webcam'},{id:'/dev/video2',name:'Second webcam'}],selected:'/dev/video0'}};
  let failPost=false;
  class EventSource {
    constructor() { this.listeners = {}; stream = this; }
    addEventListener(name, listener) { this.listeners[name] = listener; }
    send(name, payload) { this.listeners[name]({ data: JSON.stringify(payload) }); }
  }
  const frames = new Map(), rootEvents = {}, docEvents = {}, mediaEvents = {};
  const listen = (events,type,callback) => {const previous=events[type];events[type]=previous ? (...args)=>{previous(...args);callback(...args);} : callback;};
  let nextFrame = 0, clockNow=1000;
  const media = { matches: false, addEventListener(type, callback) { mediaEvents[type] = callback; },
    removeEventListener(type) { delete mediaEvents[type]; } };
  const document = { getElementById: node, querySelectorAll: () => [], createElement: () => new Element(),
    documentElement: new Element(),
    hidden: false, addEventListener(type, callback) { listen(docEvents,type,callback); },
    removeEventListener(type) { delete docEvents[type]; } };
  const context = vm.createContext({
    document,
    URLSearchParams,
    matchMedia: () => media,
    performance: {now: () => clockNow},
    requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
    cancelAnimationFrame(id) { frames.delete(id); },
    addEventListener(type, callback) { listen(rootEvents,type,callback); },
    removeEventListener(type) { delete rootEvents[type]; },
    fetch: async (url,options) => {
      if(url.startsWith('/api/context')) {contextGets.push(url);return {ok:true,json:async()=>contextPayload};}
      if(url.startsWith('/api/memory')) {memoryGets.push(url);return {ok:true,json:async()=>memoryPayload};}
      if(url==='/api/devices' && options?.method!=='POST') return {ok:true,json:async()=>devicePayload};
      if(options?.method==='POST'){
        const body=JSON.parse(options.body);posts.push({url,body});
        if(failPost) return {ok:false,json:async()=>({error:'Device unavailable'})};
        if(url==='/api/devices'){
          if(body.kind==='camera')devicePayload.camera.selected=body.device;
          else devicePayload.audio[body.kind==='microphone'?'selected_input':'selected_output']=body.device;
          return {ok:true,json:async()=>devicePayload};
        }
        if(url==='/api/vision/settings'){
          snapshot.vision_mind.interval_min_s=body.interval_min_s;
          snapshot.vision_mind.interval_max_s=body.interval_max_s;
          snapshot.vision_mind.target_hz=2/(body.interval_min_s+body.interval_max_s);
          return {ok:true,json:async()=>snapshot.vision_mind};
        }
        if(url==='/api/search/settings'){
          snapshot.search_settings=body;
          return {ok:true,json:async()=>body};
        }
        if(url==='/api/control'){
          if(body.action==='quiet')snapshot.controls.quiet_mode=body.enabled;
          else if(body.action==='autonomy')snapshot.controls.autonomy_enabled=body.enabled;
          else snapshot.controls.microphone_muted=!body.enabled;
          return {ok:true,json:async()=>snapshot};
        }
        if(url==='/api/instructions') return {ok:true,json:async()=>({instructions:body.instructions})};
        if(url==='/api/decisions' && body.action==='activate'){
          const settings=vm.runInContext('S.decisions',context);
          return {ok:true,json:async()=>({...settings,active_list:body.id,revision:settings.revision+1})};
        }
        return {ok:true,json:async()=>({accepted:true})};
      }
      return {ok:true,json:async()=>snapshot};
    },
    EventSource, setInterval() {}, clearInterval() {},
  });
  vm.runInContext("globalThis.window = globalThis", context);
  for (const asset of ["glados-themes.js", "glados-rig.js", "glados-vision.js", "glados-vision-themes.js", "glados-avatar.js", "glados-overview.js", "glados-routing.js", "glados-context.js", "glados-memory.js", "glados-devices.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/glados/webapp/static", asset), "utf8"), context);
  }
  const optics=[],Vision=context.GladosVision.Vision,ThemedVision=context.GladosVision.ThemedVision;
  context.GladosVision.ThemedVision=function(canvas){const instance=new ThemedVision(canvas);optics.push(instance);return instance;};
  const animators = [], Animator = context.GladosRig.Animator;
  context.GladosRig.Animator = function () { const instance = new Animator(); animators.push(instance); return instance; };
  const activity=()=>node('glados-eye').attributes['aria-label'].match(/aperture: ([^,]+)/)[1].replace(/^./,c=>c.toUpperCase());
  const expression=()=>node('glados-eye').attributes['aria-label'].split(', ')[1];
  const html = fs.readFileSync(path.join(__dirname, "../src/glados/webapp/static/index.html"), "utf8");
  assert.doesNotMatch(html, /avatar-follow|avatar-pause|avatar-controls/, 'Gaze and animation controls are automatic');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  await new Promise(resolve => setImmediate(resolve));
  assert.match(html,/data-view="cores"/);
  assert.deepEqual([...html.matchAll(/class="nav-item(?: active)?" data-view="([^"]+)"/g)].map(m=>m[1]),
    ['home','cores','memory','context','tools','tasks','settings','wire','vitals'],'Menu order and destinations stay unchanged');
  assert.doesNotMatch(html,/data-view="(?:minds|slots)"|id="view-slots"/);
  const coresSection=html.match(/<section class="view" id="view-cores">([\s\S]*?)<\/section>/)[1];
  assert.match(coresSection,/<h1>Cores<\/h1>/);
  assert.match(coresSection,/id="inference-slots"/);
  assert.match(coresSection,/id="core-processing"/);
  assert.equal((html.match(/id="inference-summary"/g)||[]).length,1);
  assert.match(html,/data-view="context"/);
  assert.match(html,/<h1>Test Chamber<\/h1>/);
  assert.equal(contextGets.length,0,'Context is only fetched while its tab is visible');
  stream.onopen();
  assert.equal(vm.runInContext("coreActivity({id:'probe',kind:'background',running:true,execution_status:'queued'}).label",context),'Queued');
  assert.equal(vm.runInContext("coreActivity({id:'probe',kind:'background',running:true,execution_status:'running'}).label",context),'Working');
  assert.equal(vm.runInContext("coreActivity({id:'probe',kind:'background',running:true,interval_s:null}).detail",context),'On demand');
  assert.match(vm.runInContext("coreActivity({id:'probe',kind:'background',running:true,interval_s:5}).detail",context),/after completion/);
  vm.runInContext("go('context')",context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(contextGets.at(-1),'/api/context?mode=user&view=live');
  assert.match(node('context-sections').innerHTML,/Emotion Core|irritated tone/);
  assert.match(node('context-sections').innerHTML,/Vision Core|zipper/);
  assert.match(node('context-sections').innerHTML,/catalogue is not a message sent to GLaDOS/);
  assert.match(node('context-sections').innerHTML,/#1.*Emotion Core/s);
  assert.match(node('context-sections').innerHTML,/#2.*Vision Core/s);
  assert.ok(node('context-sections').innerHTML.indexOf('Emotion Core')<node('context-sections').innerHTML.indexOf('Vision Core'));
  assert.ok(node('context-sections').innerHTML.indexOf('Vision Core')<node('context-sections').innerHTML.indexOf('Tools · separate'));
  node('context-expand').onclick();
  assert.equal((node('context-sections').innerHTML.match(/data-context-key="[^"]+" open/g)||[]).length,3);
  node('context-collapse').onclick();
  assert.doesNotMatch(node('context-sections').innerHTML,/data-context-key="[^"]+" open/);
  assert.match(document.title,/Neural Buffer/);
  node('context-search').value='zipper';node('context-search').oninput();
  assert.doesNotMatch(node('context-sections').innerHTML,/Emotion Core/);
  assert.match(node('context-sections').innerHTML,/Vision Core/);
  const hostile={...contextPayload,sections:[{source:'input',title:'<script>title</script>',
    messages:[{index:1,role:'user',content:'<img src=x onerror="bad()">',tool_calls:[{function:{arguments:'<script>bad()</script>'}}]}]}]};
  assert.doesNotMatch(context.GladosContext.renderSections(hostile),/<script>|<img/);
  node('context-search').value='';node('context-search').oninput();
  contextPayload={available:false,reason:'No request has been submitted in this mode since startup.'};
  node('context-view').value='request';node('context-view').onchange();
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(contextGets.at(-1),/view=request/);
  assert.match(node('context-sections').innerHTML,/No request has been submitted/);
  assert.equal(node('context-export').disabled,true);
  node('context-mode').value='autonomy';node('context-mode').onchange();
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(contextGets.at(-1),/mode=autonomy/);
  node('context-live').onclick();
  assert.equal(node('context-live').textContent,'Live updates: OFF');
  assert.equal(memoryGets.length,0,'Memory is fetched only when its view is visible');
  vm.runInContext("go('memory')",context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(document.title,/Test Chamber/);
  assert.match(node('memory-facts').innerHTML,/brief project updates/);
  assert.match(node('memory-recall').innerHTML,/brief project updates/);
  assert.match(node('memory-topic').textContent,/Project updates/);
  assert.equal(node('memory-next').disabled,false);
  node('memory-next').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(memoryGets.at(-1),/offset=30/);
  node('memory-search').value='different topic';
  node('memory-filter').onsubmit({preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(memoryGets.at(-1),/query=different%20topic.*offset=0/);
  assert.match(node('memory-topic').textContent,/Project updates/,'Browsing must not change recall');
  assert.doesNotMatch(context.GladosMemory.renderFacts([{content:'<script>bad()</script>',source:'<img src=x>',created_at:1}]),/<script>|<img/);
  memoryPayload={available:true,total:0,indexed_facts:0,memories:[],recall:{enabled:true,paused:true,facts:[]}};
  node('memory-refresh').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(node('memory-recall').innerHTML,/paused/);
  assert.match(node('memory-status').textContent,/No saved memories/);
  assert.equal(node('memory-next').disabled,true);
  vm.runInContext("go('settings')",context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(node('device-microphone').innerHTML,/USB Microphone/);
  assert.equal(node('device-microphone').value,'default');
  node('device-microphone').value='1';await node('device-microphone').onchange();
  assert.deepEqual(posts.at(-1),{url:'/api/devices',body:{kind:'microphone',device:1}});
  assert.equal(node('device-microphone').value,'1');
  node('device-speaker').value='2';await node('device-speaker').onchange();
  assert.deepEqual(posts.at(-1),{url:'/api/devices',body:{kind:'speaker',device:2}});
  node('device-camera').value='/dev/video2';await node('device-camera').onchange();
  assert.deepEqual(posts.at(-1),{url:'/api/devices',body:{kind:'camera',device:'/dev/video2'}});
  failPost=true;node('device-camera').value='/dev/video0';await node('device-camera').onchange();
  assert.equal(node('device-camera').value,'/dev/video2','Failed selection returns to the active device');
  assert.equal(node('device-feedback').textContent,'Device unavailable');failPost=false;
  assert.doesNotMatch(context.GladosDevices.options([{id:1,name:'<script>bad()</script>'}],1,true),/<script>/);
  vm.runInContext("go('home')",context);
  const savedInference=vm.runInContext('S.inference',context);
  context.coreTestInference={capacity:2,reserved_interactive:1,active:[{owner:'Emotion',slot:1,lane:'priority',model:'E4B'}],waiting:[{owner:'Vision',lane:'autonomy',model:'E4B'}]};
  vm.runInContext('S.inference=coreTestInference',context);
  assert.equal(context.coreIdentity({id:'emotion'}).title,'Emotion Core');
  assert.equal(context.coreIdentity({id:'compaction'}).title,'Memory Core');
  assert.equal(context.coreIdentity({id:'search'}).title,'Search Core');
  assert.equal(context.coreOwner('Search'),'Search Core');
  assert.match(context.coreActivity({id:'search',running:true,research:{status:'researching',rounds:2,max_rounds:3}}).detail,/Research round 2 of 3/);
  assert.equal(context.coreActivity({id:'search',paused:true,research:{status:'researching'}}).label,'Suspended');
  assert.equal(context.coreActivity({id:'search',running:true,research:{status:'done'}}).label,'Standby');
  assert.equal(context.coreActivity({id:'emotion',running:true}).label,'Working');
  assert.equal(context.coreActivity({id:'vision',running:true}).label,'Queued');
  assert.equal(context.coreActivity({id:'compaction',kind:'background',running:true}).label,'Standby');
  assert.match(context.coreActivity({id:'emotion',paused:true}).detail,/Finishing current run/);
  context.savedCoreInference=savedInference;
  vm.runInContext('S.inference=savedCoreInference',context);
  vm.runInContext("go('slots')",context);
  assert.match(document.title,/^Cores .*Aperture Science$/);
  vm.runInContext("go('home')",context);
  assert.ok(stream, "Client should establish the live stream, not fall back to demo data");
  stream.onopen();
  assert.equal(node("connection-status").textContent, "LIVE TELEMETRY");
  assert.equal(node("mc-inflight").textContent, 1);
  assert.equal(node("mcInflight").textContent, "in-flight 1");
  assert.equal(activity(), "Listening");
  assert.equal(expression(), "neutral");
  assert.equal(frames.size, 1);
  assert.match(node("glados-eye").innerHTML, /<clipPath/);
  assert.equal(node("priotrack").children.length, 1, "An active request with an empty queue remains visible");
  assert.equal(node("autotrack").children.length, 2);
  assert.match(node("autoQ").textContent, /background · 2 in-flight · 2 queued/);
  assert.match(node("v-audio").innerHTML, /hot/);
  assert.match(node("v-audio").innerHTML, /-20.0 dB/);
  assert.match(node("v-audio").innerHTML, /67%/);
  assert.equal(node("g-pv").textContent, "Off", "A disabled emotion agent is explicitly labeled");
  snapshot.operator={instructions:'Be concise.',default_instructions:'Be helpful.'};
  stream.send('snapshot',snapshot);
  assert.equal(node('instructions-preview').textContent,'Be concise.');

  const updated = JSON.parse(JSON.stringify(snapshot));
  updated.lanes.priority = { queue: 2, inflight: 0 };
  updated.lanes.autonomy = { queue: 0, inflight: 0, workers: 0 };
  updated.lanes.enabled = false;
  updated.audio = { rms: 0, vad_active: false };
  updated.interaction = { seconds_since_user: null, seconds_since_assistant: null };
  stream.send("state", updated);
  assert.equal(node("mc-inflight").textContent, 0);
  assert.equal(node("priotrack").children.length, 0, "Queued work must not be called in-flight");
  assert.equal(node("autotrack").children.length, 0);
  assert.match(node("workers").innerHTML, /Background processing is unavailable/);
  assert.match(node("v-audio").innerHTML, /idle/);
  assert.match(node("v-audio").innerHTML, /-120.0 dB/);
  assert.match(node("v-audio").innerHTML, /<span>0%<\/span>/);
  assert.equal(node("mc-lastuser").textContent, "never");
  // Continuous vision/emotion work must not make GLaDOS look busy with the user.
  updated.lanes.autonomy.inflight = 2;
  stream.send("state", updated);
  assert.equal(node("mc-inflight").textContent, 0);
  assert.equal(node("mcInflight").textContent, "in-flight 0");
  assert.equal(activity(), "Idle");
  assert.equal(expression(), "neutral");
  assert.doesNotMatch(html,/avatar-activity|avatar-expression|avatar-detail|Oh\. It/);
  assert.equal(node("autotrack").children.length, 2, "Background activity remains visible in its lane");
  // VAD reaches the eye between dashboard updates, without repainting the console.
  vm.runInContext('globalThis.savedAudioPaint=paint;globalThis.audioPaintCalls=0;paint=()=>audioPaintCalls++',context);
  const audioAt=Date.now()/1000;
  stream.send('audio',{rms:0.1,vad_active:true,updated_at:audioAt});
  assert.equal(activity(),'Listening');
  assert.equal(animators[0].listen.on,true);
  assert.equal(context.audioPaintCalls,0,'Fast audio telemetry only updates the avatar');
  stream.send('state',{audio:{rms:0,vad_active:false,updated_at:audioAt-0.1}});
  assert.equal(animators[0].listen.on,true,'A stale dashboard snapshot cannot erase live VAD');
  stream.send('audio',{rms:0,vad_active:false,updated_at:audioAt+0.1});
  assert.equal(animators[0].listen.on,false,'The eye releases attention after speech ends');
  vm.runInContext('paint=savedAudioPaint',context);

  updated.lanes.autonomy.inflight = 0;
  stream.send("state", updated);

  updated.agents = [{ agent_id: "a1", title: "Updated agent", running: true, tick_count: 8 }];
  updated.agent_minds = [...snapshot.agent_minds, {...updated.agents[0],id:'a1',kind:'background'}];
  updated.slots = [{ slot_id: "s1", title: "Fresh task", status: "done", summary: "New result" }, {slot_id:'a1',title:'Private mind state',status:'idle'}];
  updated.vision = "New scene";
  stream.send("snapshot", updated);
  assert.match(node("minds").children.map(n=>n.innerHTML).join(''), /Updated agent/);
  assert.doesNotMatch(node("minds").children.map(n=>n.innerHTML).join(''), /Forecast Mind/);
  assert.match(node('components').innerHTML,/Forecast Mind/);
  assert.match(node("slots").children[0].innerHTML, /New result/);
  assert.equal(node('slots').children.length,1,'Agent state is not a user task');
  assert.match(node("v-vision").innerHTML, /New scene/);
  updated.agents = []; updated.agent_minds=[]; updated.minds = []; updated.slots = [];
  stream.send("snapshot", updated);
  assert.match(node("slots").innerHTML, /No assignments/);
  assert.match(node("minds").innerHTML, /No cores connected/);

  const homeBeforeDebug = node('home-wire').children.length;
  stream.send('obs', {source:'vision',kind:'observation',level:'debug',message:'Debug camera caption'});
  assert.equal(node('home-wire').children.length,homeBeforeDebug,'Debug does not crowd the home feed');
  vm.runInContext("wireFilter='debg';renderFullWire()",context);
  assert.match(node('full-wire').children[0].innerHTML,/Debug camera caption/);
  stream.send('obs', {source:'audio',kind:'gap',level:'warning',message:'Microphone gap'});
  vm.runInContext("wireFilter='warn';renderFullWire()",context);
  assert.match(node('full-wire').children[0].innerHTML,/Microphone gap/);
  vm.runInContext("wireFilter='all'",context);

  for (let i = 0; i < 120; i++) {
    stream.send("obs", { timestamp: i, source: "test", kind: "tick", level: "info", message: `event-${i}` });
  }
  vm.runInContext("renderFullWire()", context);
  assert.match(node("full-wire").children[0].innerHTML, /event-119/);
  assert.match(node("full-wire").lastChild.innerHTML, /event-10/);
  stream.onerror();
  assert.equal(node("connection-status").textContent, "DISCONNECTED · RETRYING");
  stream.onopen();
  assert.equal(node("connection-status").textContent, "LIVE TELEMETRY");
  // Actual stream state controls the avatar, including stale speaking data on disconnect.
  updated.audio.vad_active = false;
  updated.lanes.priority.inflight = 1;
  stream.send("state", updated);
  assert.equal(expression(), "processing");
  updated.speaking = true;
  updated.emotion = {pleasure: -0.7, arousal: 0.6, dominance: 0.5};
  stream.send("state", updated);
  assert.equal(activity(), "Speaking");
  assert.equal(expression(), "angry glare");
  stream.send("performance", {revision: 5, active: true, emotion: "smug"});
  assert.equal(expression(), "smug", "Spoken directions override PAD during playback");
  updated.performance = {revision: 4, active: false, emotion: null};
  stream.send("snapshot", updated);
  assert.equal(expression(), "smug", "Older snapshots cannot rewind playback");
  stream.send("performance", {revision: 6, active: false, emotion: null});
  assert.equal(expression(), "processing");
  stream.send("performance", {revision: 7, active: true, emotion: "disappointed"});
  assert.equal(expression(), "disappointed");
  updated.performance = null;
  function frame(now) {
    clockNow=now;
    const [id, callback] = frames.entries().next().value;
    frames.delete(id); callback(now);
  }
  frame(1000);
  const pose = node("glados-eye").innerHTML;
  frame(1100);
  assert.notEqual(node("glados-eye").innerHTML, pose);
  assert.doesNotMatch(node("glados-eye").innerHTML, /NaN|Infinity/);
  stream.onerror();
  assert.equal(expression(), "offline");
  assert.equal(activity(), "Disconnected");
  stream.onopen();
  assert.equal(activity(), "Speaking");
  stream.send("performance", {revision: 1, active: true, emotion: "surprised"});
  assert.equal(expression(), "surprised", "New engine may restart its revision counter");
  stream.send("performance", {revision: 2, active: false, emotion: null});

  // The vision inspector refreshes observations without rebuilding its controls.
  const vision={scene:'A desk and cup.',changes:'A cup appeared.',recent_events:'A cup was placed on the desk.',window_capture_times:[1,3,8,10],next_delay_s:4.5,motion_activity:0.1,revision:1,captured_at:Date.now()/1000,
    paused:false,camera:{connected:true,enabled:true},target_hz:1,completed_hz:1,inference_ms:780,
    selected_age_ms:120,candidates:5,sharpness:90,scoring_ms:0.5};
  updated.agent_minds=[{id:'vision',agent_id:'vision',title:'Vision',role:'Scene and visual changes',
    kind:'background',model:'gemma-4-E4B',running:true,paused:false,interval_s:1}];
  updated.vision_mind=vision;
  stream.send('snapshot',updated);
  await vm.runInContext("openMind('vision')",context);
  assert.equal(node('vision-scene').textContent,'A desk and cup.');
  assert.equal(node('vision-events').textContent,'A cup was placed on the desk.');
  assert.match(node('vision-metrics').innerHTML,/4 \/ 4 images.*9.0 s span/);
  assert.match(node('vision-settings-status').textContent,/Motion activity: 10%/);
  assert.equal(node('vision-changes').textContent,'A cup appeared.');
  assert.match(node('vision-preview').src,/^\/api\/vision\/live\?started=/);
  const liveSource=node('vision-preview').src;
  assert.match(node('vision-metrics').innerHTML,/780 ms/);
  const toggle=node('mind-toggle').onclick;
  vision.scene='A desk, cup and book.';vision.changes='A book appeared.';vision.revision=2;
  stream.send('state',{vision_mind:vision});
  assert.equal(node('vision-scene').textContent,'A desk, cup and book.');
  assert.equal(node('vision-preview').src,liveSource,'Caption updates must not restart the live stream');
  assert.match(node('vision-metrics').innerHTML,/YuNet · CPU/);
  assert.equal(node('mind-toggle').onclick,toggle,'Live observations must preserve button handlers');
  vision.last_question={question:'Does my jacket have a zipper?',answer:'Yes, a zipper is visible.',evidence:'A metal pull is visible.'};
  stream.send('state',{vision_mind:vision});
  assert.match(node('vision-question').textContent,/zipper is visible/);
  snapshot.agent_minds=updated.agent_minds;snapshot.vision_mind=vision;
  vision.paused=true;vision.camera.enabled=false;vision.camera.connected=false;
  stream.send('state',{vision_mind:vision});
  assert.equal(node('vision-preview').src,'');
  assert.equal(node('vision-preview').hidden,true);
  await toggle();
  assert.deepEqual(posts.at(-1),{url:'/api/minds/control',body:{agent_id:'vision',action:'resume'}});
  assert.match(node('vision-status').textContent,/camera released/);
  await node('mind-run').onclick();
  assert.deepEqual(posts.at(-1),{url:'/api/minds/control',body:{agent_id:'vision',action:'run'}});
  vm.runInContext("inspectedMind=null",context);

  // Camera gaze and idle sleep use local presence, independently of background inference.
  updated.speaking=false;updated.audio.vad_active=false;updated.lanes.priority.inflight=0;
  updated.lanes.autonomy.inflight=1;updated.interaction.seconds_since_user=null;
  stream.send('performance',{revision:3,active:false,emotion:null});
  vision.paused=false;vision.camera.enabled=true;vision.camera.connected=true;
  const face={available:true,present:true,x:0.6,y:-0.3,closeness:0.4,
    observed_at:Date.now()/1000,absent_seconds:0,sleep_after_s:5};
  vision.camera.face=face;
  stream.send('state',updated);
  assert.equal(node('avatar-optic').hidden,false);
  assert.match(node('avatar-optic-source').src,/^\/api\/vision\/live\?overlay=0&started=/);
  const opticSource=node('avatar-optic-source').src;
  stream.send('state',updated);
  assert.equal(node('avatar-optic-source').src,opticSource,'Tracking updates keep one camera stream');
  node('avatar-optic-source').naturalWidth=640;node('avatar-optic-source').naturalHeight=480;
  frame(2000);
  assert.equal(optics[0].mode,'thermal');
  assert.equal(optics[0].hasFrame,true,'Original treatment processes the shared camera image');
  assert.equal(optics[0].subject.x,optics[0].eye.lookX,'Reticle uses the eye scanning coordinate');
  vision.inference_sequence=0;stream.send('state',updated);
  animators[0].blink.phase=0;
  vision.inference_sequence=1;stream.send('camera',{paused:false,camera:vision.camera,inference_sequence:1,inference_active:1});
  assert.equal(animators[0].blink.phase,1,'Vision inference starts a blink');
  animators[0].blink.phase=0;
  stream.send('camera',{paused:false,camera:vision.camera,inference_sequence:1,inference_active:1});
  assert.equal(animators[0].blink.phase,0,'The same inference does not restart the blink');
  assert.equal(animators[0].gaze.tx,-0.6,'Camera X is reversed to follow the viewer across the screen');
  assert.equal(animators[0].gaze.ty,-0.3);
  assert.equal(animators[0].gaze.follow,true,'Face tracking takes priority over decorative glances');
  vm.runInContext('globalThis.savedPaint=paint;globalThis.cameraPaintCalls=0;paint=()=>cameraPaintCalls++',context);
  const newerFace={...face,x:0.25,y:0.1,observed_at:face.observed_at+0.01};
  stream.send('camera',{paused:false,camera:{...vision.camera,face:newerFace}});
  assert.equal(animators[0].gaze.tx,-0.25,'Camera events move the eye without a slow state update');
  assert.equal(animators[0].gaze.ty,0.1);
  assert.equal(context.cameraPaintCalls,0,'Fast gaze telemetry must not repaint the console');
  vm.runInContext('paint=savedPaint',context);
  stream.send('state',updated);
  assert.equal(animators[0].gaze.tx,-0.25,'An older snapshot must not undo a newer face position');
  face.observed_at=newerFace.observed_at+0.01;stream.send('state',updated);
  assert.equal(animators[0].gaze.tx,-0.6);
  node('avatar-optic-source').naturalWidth=640;node('avatar-optic-source').naturalHeight=480;
  clockNow=5000;frame(clockNow);
  assert.equal(node('optic-subject').hidden,false,'TEST SUBJECT appears when a fresh face is detected');
  assert.equal(node('optic-motion').textContent,'MOTION 0.00','Presence alone does not count as motion');
  rootEvents.pointermove({clientX:1,clientY:1,pointerType:'mouse'});
  assert.equal(animators[0].gaze.tx,-0.6,'Camera gaze takes precedence over the pointer');
  face.present=false;face.absent_seconds=4;
  stream.send('state',updated);
  clockNow=5100;frame(clockNow);
  assert.equal(node('optic-subject').hidden,true,'TEST SUBJECT disappears when the face is absent');
  assert.equal(activity(),'Idle','Brief detector misses do not put the eye to sleep');
  face.absent_seconds=6;
  stream.send('state',updated);
  assert.equal(activity(),'Sleeping');
  assert.equal(expression(),'sleeping');
  assert.equal(animators[0].pres.present,false);
  updated.lanes.priority.inflight=1;stream.send('state',updated);
  assert.equal(activity(),'Processing');
  updated.lanes.priority.inflight=0;updated.audio.vad_active=true;stream.send('state',updated);
  assert.equal(activity(),'Listening');
  updated.audio.vad_active=false;updated.interaction.seconds_since_user=1;stream.send('state',updated);
  assert.equal(activity(),'Idle','Recent user input wakes the idle eye');
  updated.interaction.seconds_since_user=null;vision.paused=true;stream.send('state',updated);
  assert.equal(activity(),'Idle','A paused camera must not imply an empty room');
  vision.paused=false;face.observed_at=Date.now()/1000-10;stream.send('state',updated);
  assert.equal(activity(),'Idle','Stale face signals must not put the eye to sleep');
  face.observed_at=Date.now()/1000;face.present=true;face.absent_seconds=0;stream.send('state',updated);
  assert.equal(animators[0].pres.present,true);
  vision.paused=true;vision.camera.enabled=false;stream.send('state',updated);
  assert.equal(animators[0].gaze.active,0,'Turning the camera off releases its face gaze');
  assert.equal(node('avatar-optic').hidden,true);
  assert.equal(node('avatar-optic-source').src,'');
  rootEvents.pointermove({clientX:1,clientY:1,pointerType:'mouse'});
  assert.equal(animators[0].gaze.active,1,'Camera OFF automatically follows the mouse');
  assert.notEqual(animators[0].gaze.tx,-0.6);
  vision.paused=false;vision.camera.enabled=true;stream.send('state',updated);
  assert.equal(animators[0].gaze.tx,-0.6);
  delete vision.camera.face;stream.send('state',updated);
  assert.equal(animators[0].gaze.active,0,'Camera ON without a face releases the previous gaze');
  rootEvents.pointermove({clientX:1,clientY:1,pointerType:'mouse'});
  assert.equal(animators[0].gaze.active,0,'Camera ON must not follow the pointer when no face is detected');
  vision.camera.face=face;face.observed_at=Date.now()/1000-10;stream.send('state',updated);
  rootEvents.pointermove({clientX:1,clientY:1,pointerType:'mouse'});
  assert.equal(animators[0].gaze.active,0,'Stale camera tracking must not enable pointer gaze');
  face.observed_at=Date.now()/1000;stream.send('state',updated);
  face.present=false;face.absent_seconds=6;stream.send('state',updated);
  assert.equal(activity(),'Sleeping');
  rootEvents.keydown({key:'a'});
  assert.equal(activity(),'Idle','Console interaction wakes the eye');
  delete vision.camera.face;
  updated.lanes.autonomy.inflight=0;
  stream.send('state',updated);

  // Person selection survives detector reorderings, alternates, and recovers after absence.
  const selector=new context.GladosAvatar.GazeSelector();
  const left={x:-0.6,y:0},right={x:0.6,y:0};
  assert.equal(selector.select([right,left],0),left);
  assert.equal(selector.select([left,right],2.9),left);
  assert.equal(selector.select([right,left],3.1),right);
  assert.equal(selector.select([left,right],3.2),right);
  assert.equal(selector.select([left],4),left);
  assert.equal(selector.select([],5),null);
  assert.equal(selector.select([right],6),right);

  // A wave interrupts briefly, then restores the same face despite continuing motion.
  const attention=new context.GladosAvatar.GazeSelector();
  const viewer={kind:'face',x:-0.6,y:0,size:{width:0.15,height:0.25}};
  const hand={kind:'motion',x:0.75,y:0.3,closeness:0.3};
  assert.equal(attention.select([viewer],0),viewer);
  assert.equal(attention.select([viewer],0.1,[hand]),hand);
  const movingHand={...hand,x:0.8};
  assert.equal(attention.select([viewer],0.4,[movingHand]),movingHand);
  assert.equal(attention.select([viewer],1.1,[movingHand]),viewer,'Eye returns to the viewer after a short hand glance');
  assert.equal(attention.select([viewer],2,[movingHand]),viewer,'Continuing movement does not monopolize gaze');
  assert.equal(attention.select([viewer],3.2,[movingHand]),movingHand,'Another wave can catch attention');
  assert.equal(attention.select([],4.2,[movingHand]),movingHand,'Movement can be followed without a visible face');
  assert.equal(attention.select([],4.3,[]),null,'Empty frames release a movement target');
  const faceMotion={...hand,x:viewer.x,y:viewer.y};
  assert.equal(attention.select([viewer],8,[faceMotion]),viewer,'Facial movement stays on the face');

  const groupFace={...face,present:true,absent_seconds:0,faces:[
    {x:0.6,y:-0.2,bbox:[0.7,0.2,0.2,0.3]},
    {x:-0.6,y:-0.2,bbox:[0.1,0.2,0.2,0.3]}]};
  vision.camera.face=groupFace;
  clockNow=10000;stream.send('state',updated);
  assert.equal(animators[0].gaze.tx,-0.6);
  clockNow=13100;frame(clockNow);
  assert.equal(animators[0].gaze.tx,0.6,'Mounted eye switches to the other person');
  animators[0].sacc.next=0.8;
  groupFace.faces.reverse();stream.send('state',updated);
  assert.equal(animators[0].gaze.tx,0.6,'Detector order does not switch the selected person');
  assert.equal(animators[0].sacc.next,0.8,'Repeated group updates preserve face scanning');
  vision.camera.face=face;face.present=true;stream.send('state',updated);

  // The mounted eye and reticle glance at peripheral image motion and return together.
  const faceCloseness=animators[0].pres.close;
  optics[0].sg.sample=(x,y,w,h)=>x>w*0.8 && x<w*0.95 && y>h*0.25 && y<h*0.45?180:40;
  clockNow=13200;frame(clockNow);
  clockNow=13240;frame(clockNow);
  assert.ok(animators[0].gaze.tx>0.6,'A waving hand outside the face crop attracts the eye');
  assert.equal(animators[0].gaze.kind,'motion');
  assert.equal(animators[0].pres.close,faceCloseness,'A hand glance preserves the viewer distance and eye zoom');
  assert.equal(optics[0].subject.kind,'motion');
  assert.equal(optics[0].subject.x,optics[0].eye.lookX,'Motion reticle and eye share coordinates');
  assert.equal(node('optic-subject').textContent,'MOVEMENT');
  clockNow=14300;frame(clockNow);
  assert.equal(animators[0].gaze.tx,-0.6,'Eye contact resumes after the motion glance');
  assert.equal(optics[0].subject.kind,'face');
  assert.equal(node('optic-subject').textContent,'TEST SUBJECT');
  delete optics[0].sg.sample;optics[0].setSource(node('avatar-optic-source'));
  clockNow=15000;frame(clockNow);
  face.present=false;face.absent_seconds=6;stream.send('state',updated);
  const absentState={connection:'live',visionMind:{paused:false,camera:vision.camera},prioInflight:0};
  assert.equal(context.GladosAvatar.presentation(absentState,false).activity,'Sleeping');
  assert.equal(context.GladosAvatar.presentation(absentState,false,true).activity,'Idle','Motion keeps the idle eye open');
  optics[0].sg.sample=(x,y,w,h)=>x>w*0.8 && y>h*0.2 && y<h*0.45?180:40;
  clockNow=15100;frame(clockNow);
  clockNow=15140;frame(clockNow);
  assert.equal(activity(),'Idle','Localized movement opens the idle eye without starting an inference');
  assert.equal(animators[0].gaze.kind,'motion');
  vision.paused=true;stream.send('state',updated);
  assert.equal(animators[0].gaze.active,0,'Suspending the camera releases motion gaze');
  assert.equal(optics[0].motionTargets.length,0,'Stopping the feed clears old motion targets');
  delete optics[0].sg.sample;vision.paused=false;face.present=true;face.absent_seconds=0;
  stream.send('state',updated);

  // Full-source analysis and bounded 80% crop work at edges without false motion from panning.
  const optic=new Vision(new Element());optic.setSource({naturalWidth:640,naturalHeight:480});
  optic.setEye({lookX:0.9,lookY:0.7,color:'#F2C265',lidTop:-90,lidBot:90});
  optic.setSubject({x:0.9,y:0.7});
  for(let i=0;i<100;i++) optic.frame(1/30);
  assert.equal(optic.SH,192,'Full camera aspect is retained before display cropping');
  assert.ok(Math.abs(optic.viewport.height-0.8)<1e-9);
  assert.ok(optic.viewport.x>=0 && optic.viewport.x+optic.viewport.width<=1);
  assert.ok(optic.viewport.y>=0 && optic.viewport.y+optic.viewport.height<=1);
  assert.ok(optic.focus.e>0.99,'Stationary people retain the SUBJECT target');
  assert.equal(Math.max(...optic.heat),0,'Scanning does not create false source motion');
  assert.equal(optic.motionEnergy,0,'Panning and face presence do not inflate the motion readout');
  assert.ok(Math.abs(optic.focus.x-0.95)<1e-9);
  const subjectPan=optic.pan.x;
  optic.setSubject(null);
  for(let i=0;i<180;i++) optic.frame(1/30);
  assert.ok(optic.focus.e<0.03,'Empty-room target fades');
  assert.ok(Math.abs(optic.pan.x-subjectPan)>0.02,'Empty-room view scans across the source');
  optic.sg.intensity=180;const movingObservation=optic.frame(1/30);
  assert.ok(movingObservation.motion>0 && movingObservation.motion<=1,'Motion measures actual frame differences');
  assert.ok(Math.max(...optic.heat)>0,'Actual camera changes generate thermal motion trails');
  assert.equal(optic.motionTargets.length,0,'A uniform exposure change is not a localized target');
  const initialHeat=optic.heat[0];
  for(let i=0;i<18;i++) optic.frame(1/30);
  assert.ok(optic.heat[0]<=initialHeat*0.051,'Thermal trails fade by 95% within 0.6 seconds');

  optic.sg.intensity=40;optic.setSource({naturalWidth:640,naturalHeight:480});
  optic.setSubject({x:-0.9,y:0});optic.setEye({lookX:-0.9,lookY:0});
  for(let i=0;i<30;i++)optic.frame(1/30);
  const cropEdge=optic.viewport.x+optic.viewport.width;
  optic.sg.sample=(x,y,w,h)=>x>w*0.8 && x<w*0.95 && y>h*0.25 && y<h*0.45?180:40;
  optic.frame(1/30);
  assert.ok(optic.motionTargets[0].x>0.6,'Localized movement is found in the full image');
  assert.ok((optic.motionTargets[0].x+1)/2>cropEdge,'Motion outside the current crop still attracts attention');
  for(let i=0;i<30;i++)optic.frame(1/30);
  assert.equal(optic.motionTargets.length,0,'Movement targets expire after the image stops changing');
  optic.setSource(null);assert.equal(optic.motionTargets.length,0);
  const regions=context.GladosVision.motionRegions,map=new Float32Array(256*192);
  for(let y=40;y<70;y++)for(let x=10;x<35;x++)map[y*256+x]=1;
  for(let y=130;y<160;y++)for(let x=215;x<240;x++)map[y*256+x]=1;
  const localized=regions(map,256,192);
  assert.equal(localized.length,2,'Separate moving objects retain distinct targets');
  assert.ok(localized.some(m=>m.x<-0.6) && localized.some(m=>m.x>0.6));
  assert.equal(regions(new Float32Array(256*192).fill(1),256,192).length,0,'Camera shake does not choose a target');
  const edges=new Float32Array(256*192);
  for(let y=0;y<192;y++)if(y%16<4)edges.fill(1,y*256,(y+1)*256);
  assert.equal(regions(edges,256,192).length,0,'Broad moving edges are not treated as a local object');
  const noise=new Float32Array(256*192);noise[0]=1;
  assert.equal(regions(noise,256,192).length,0,'A single noisy pixel cannot attract gaze');
  const mirrored=regions(map,256,192,false);
  assert.ok(Math.abs(localized[0].x+mirrored[0].x)<1e-6,'Motion follows the same viewer coordinate convention');
  const movementRig=new Animator({idle:0,mechanical:0});
  movementRig.lookAt(0.8,0.2,true,null,'motion');
  for(let i=0;i<100;i++)movementRig.step(1/30);
  assert.equal(movementRig.sacc.fixation,'center','A movement target is not scanned as a face');

  // Listening is visible, settles quickly, and keeps microphone audio out of speech animation.
  const listeningRig=new Animator({idle:0,mechanical:0});
  listeningRig.setPresence(true);listeningRig.setListening(true);
  let listeningPose;
  for(let i=0;i<6;i++)listeningPose=listeningRig.step(1/30);
  assert.ok(listeningPose.listening>0.9,'Speech attention appears within 200ms');
  assert.ok(listeningPose.a>context.GladosRig.NEUTRAL.a+4,'Listening opens the aperture');
  assert.notEqual(listeningPose.color,context.GladosRig.NEUTRAL.color);
  assert.match(context.GladosRig.renderEye(listeningPose),/class="listening-halo"/);
  assert.equal(listeningPose.speak,0,'VAD must not animate GLaDOS as if she is speaking');
  listeningRig.setListening(false);
  for(let i=0;i<60;i++)listeningPose=listeningRig.step(1/30);
  assert.ok(listeningPose.listening<0.01);
  assert.doesNotMatch(context.GladosRig.renderEye(listeningPose),/listening-halo/);

  // Settings exposes the actual capability tree and groups editable fixed bindings.
  const treePreview={root:'area',strategy:'hierarchical',nodes:[
    {id:'area',name:'Capability area',options:[{id:'system',description:'Local system',action:'plan',next:'system'}]},
    {id:'system',name:'Local system',options:[{id:'time',description:'CPU load',action:'tool',tool:'run_safe_command',arguments:{task:'cpu_load'}}]}]};
  context.treePreview=treePreview;
  vm.runInContext('renderRoutingStructure(treePreview)',context);
  assert.match(node('routing-structure').innerHTML,/Capability area/);
  assert.match(node('routing-structure').innerHTML,/Fixed: run_safe_command/);
  assert.match(node('routing-structure').innerHTML,/<svg/);
  assert.match(node('routing-structure').innerHTML,/Download Mermaid/);
  const routeTrace={stages:[{list_id:'area',option_id:'system',accepted:true,scores:[{id:'system',probability:.96}]},
    {list_id:'system',option_id:'time',accepted:true,scores:[{id:'time',probability:.99}]}]};
  const overview=context.GladosRouting.graph(treePreview,new Set(['area']),routeTrace);
  assert.equal(overview.nodes.length,2,'Overview hides tool leaves until expanded');
  const expandedTree=context.GladosRouting.graph(treePreview,new Set(['area','system']),routeTrace);
  assert.equal(expandedTree.nodes.length,3);
  assert.equal(expandedTree.edges.filter(edge=>edge.active).length,2,'Both steps of the selected path are highlighted');
  assert.ok(expandedTree.nodes.every(n=>n.y>0&&n.y<expandedTree.height));
  assert.match(context.GladosRouting.svg(treePreview,new Set(['area','system']),routeTrace),/A · 99.0%/);
  const mermaid=context.GladosRouting.mermaid(treePreview,routeTrace);
  assert.match(mermaid,/flowchart LR/);
  assert.match(mermaid,/Fixed: run_safe_command/);
  assert.match(mermaid,/linkStyle 0,1 /);
  const compactTree={root:'area',nodes:[{id:'area',name:'Capability area',options:[
    {id:'area_vision',action:'plan',display_name:'Vision',tool_scope:['vision_look']},
    {id:'back_to_assistant',action:'plan',fallback:true,description:'No capability matches'}]}]};
  const noMatch={stages:[{list_id:'area',option_id:'back_to_assistant',accepted:false,fallback_selected:true}]};
  const compact=context.GladosRouting.graph(compactTree,new Set(['area']),noMatch);
  assert.equal(compact.nodes[1].title,'Vision');
  assert.equal(compact.nodes[2].title,'No matching choice');
  assert.match(compact.nodes[2].detail,/no tool selected/);
  assert.equal(compact.edges.filter(edge=>edge.active).length,1);
  assert.equal(compact.nodes[0].uncertain,false,'Explicit fallback is distinct from low confidence');
  assert.match(context.GladosRouting.mermaid(compactTree,noMatch),/No matching choice/);
  const unsafeTree={root:'area',nodes:[{id:'area',name:'<script>"evil"</script>',options:[{id:'reply',action:'reply',description:'</textarea><script>evil</script>'}]}]};
  assert.doesNotMatch(context.GladosRouting.svg(unsafeTree,new Set(['area'])),/<script>/);
  assert.doesNotMatch(context.GladosRouting.mermaid(unsafeTree),/<script>/);
  assert.match(context.GladosRouting.mermaid(unsafeTree),/#34;/);
  node('routing-expand').onclick();
  assert.match(node('routing-structure').innerHTML,/Assistant fills arguments|Fixed: run_safe_command/);
  node('routing-collapse').onclick();
  snapshot.controls={...snapshot.controls,quiet_available:true,quiet_mode:false,autonomy_available:true,autonomy_enabled:false};
  stream.send('state',snapshot);
  assert.equal(node('settings-autonomy').textContent,'Autonomy: OFF');
  assert.equal(node('settings-autonomy').disabled,false);
  await node('settings-autonomy').onclick();
  assert.deepEqual(posts.at(-1),{url:'/api/control',body:{action:'autonomy',enabled:true}});
  assert.equal(node('settings-autonomy').textContent,'Autonomy: ON');
  assert.equal(node('settings-quiet').textContent,'Put GLaDOS to sleep');
  await node('settings-quiet').onclick();
  assert.deepEqual(posts.at(-1),{url:'/api/control',body:{action:'quiet',enabled:true}});
  assert.equal(node('control-quiet').textContent,'Wake GLaDOS');
  const quietPresentation=context.GladosAvatar.presentation({connection:'live',controls:{quiet_mode:true},audio:{vad:true},speaking:true});
  assert.equal(quietPresentation.look,'sleeping');
  assert.equal(quietPresentation.listening,false);
  assert.equal(quietPresentation.speaking,false);
  await node('control-quiet').onclick();
  assert.equal(snapshot.controls.quiet_mode,false);

  await node('settings-autonomy').onclick();
  assert.equal(node('settings-autonomy').textContent,'Autonomy: OFF');
  context.editableRouting={id:'speech',name:'Routing',strategy:'hierarchical',instructions:'Select intent',
    enabled:true,threshold:.8,margin:.2,fallback:'assist',options:[{id:'time',description:'CPU load',action:'tool',tool:'run_safe_command',arguments:{task:'cpu_load'},enabled:true}]};
  vm.runInContext('S.decisions={revision:1,active_list:"speech",lists:[editableRouting,{...editableRouting,id:"custom",name:"Custom"}],enabled:true};renderDecisionSettings()',context);
  assert.doesNotMatch(node('decision-activation').innerHTML,/decision-enabled|decision-speculative|decision-apply/);
  assert.match(node('decision-activation').innerHTML,/Routing is automatic/);
  node('decision-active').value='custom';
  await node('decision-active').onchange();
  assert.deepEqual(posts.at(-1),{url:'/api/decisions',body:{action:'activate',revision:1,id:'custom'}});
  assert.equal(vm.runInContext('S.decisions.active_list',context),'custom');
  vm.runInContext('S.decisions={revision:1,lists:[editableRouting],enabled:false};editDecision(editableRouting)',context);
  assert.match(node('drawer-body').innerHTML,/Capability, then action/);
  assert.match(node('dl-options').innerHTML,/system choices/);
  assert.match(node('dl-options').innerHTML,/data-field="category"/);
  context.stagePreview={action:'plan',accepted:true,elapsed_ms:100,category:'mcp',server:'lights',tool_scope:['mcp.lights.turn_on'],
    stages:[{name:'Capability area',accepted:true,scores:[{label:'C',description:'MCP services',probability:.98}]},
            {name:'MCP services',accepted:true,scores:[{label:'A',description:'Lights',probability:.97}]}]};
  const stagedHtml=vm.runInContext('decisionResult(stagePreview)',context);
  assert.match(stagedHtml,/Step 1.*Capability area/);
  assert.match(stagedHtml,/Step 2.*MCP services/);
  assert.match(stagedHtml,/mcp.lights.turn_on/);

  // Camera following settles quickly and remains accurate across idle moods and speech.
  const trackingRig = new Animator();
  trackingRig.setLook('bored');trackingRig.setPresence(true);trackingRig.lookAt(0.6,-0.3,true);
  trackingRig.aloof.on=true;trackingRig.aloof.w=1;trackingRig.aloof.x=-0.9;trackingRig.aloof.y=0.8;
  let gazeParams;
  for(let i=0;i<6;i++) gazeParams=trackingRig.step(1/30);
  assert.ok(Math.abs(gazeParams.lookX-0.6)<0.08,'Eye catches up to the face within 200 ms');
  assert.ok(Math.abs(gazeParams.lookY+0.3)<0.08);
  let worstError=0;
  for(let i=0;i<300;i++){
    trackingRig.setVoice(i%12<6?0.8:0);
    gazeParams=trackingRig.step(1/30);
    worstError=Math.max(worstError,Math.abs(gazeParams.lookX-0.6),Math.abs(gazeParams.lookY+0.3));
  }
  assert.ok(worstError<0.18,'Face scanning and speech stay near the tracked face');
  trackingRig.lookAt(null);
  assert.equal(trackingRig.gaze.follow,false,'Releasing face gaze restores normal animation');

  // Scan around the moving face, alternating eyes with shorter mouth fixations.
  let scanSeed=42;
  context.scanRandom=()=>((scanSeed=(1664525*scanSeed+1013904223)>>>0)/2**32);
  vm.runInContext('globalThis.savedScanRandom=Math.random;Math.random=scanRandom',context);
  const scanRig=new Animator({idle:0,speechGaze:0});
  const fixations=new Set(),mouthHolds=[],eyeHolds=[];
  let previousFixation='center',scanError=0;
  for(let i=0;i<1200;i++){
    // New 30 Hz face positions must not restart the scanning clock.
    const x=0.4*Math.sin(i/90),y=-0.2+0.1*Math.sin(i/120);
    scanRig.lookAt(x,y,true,{width:0.25,height:0.4});
    const p=scanRig.step(1/30),fix=scanRig.sacc.fixation;
    fixations.add(fix);
    if(fix!==previousFixation){
      if(fix==='mouth') mouthHolds.push(scanRig.sacc.next);
      else if(fix==='left'||fix==='right') eyeHolds.push(scanRig.sacc.next);
    }
    previousFixation=fix;
    scanError=Math.max(scanError,Math.abs(p.lookX-scanRig.gaze.x),Math.abs(p.lookY-scanRig.gaze.y));
  }
  vm.runInContext('Math.random=savedScanRandom',context);
  for(const fix of ['left','right','mouth']) assert.ok(fixations.has(fix),'Scan includes '+fix);
  assert.ok(Math.max(...mouthHolds)<Math.min(...eyeHolds),'Mouth glances are briefer than eye contact');
  assert.ok(scanError>0.04 && scanError<=0.141,'Scanning is visible and remains within the moving face');

  // Either face dimension reaching 2/3 saturates the zoom, with smooth intermediate sizes.
  const nearnessState={connection:'live',visionMind:{paused:false,camera:{enabled:true,connected:true,
    face:{available:true,present:true,x:0,y:0,observed_at:Date.now()/1000,bbox:[0,0,0.15,0.25]}}}};
  const zoomFor=(width,height)=>{
    nearnessState.visionMind.camera.face.bbox=[0,0,width,height];
    return context.GladosAvatar.presentation(nearnessState).gaze.closeness;
  };
  assert.equal(zoomFor(2/3,0.2),1,'Face width alone reaches full nearness');
  assert.equal(zoomFor(0.2,2/3),1,'Face height alone reaches full nearness');
  assert.equal(zoomFor(0.9,0.9),1,'Larger faces cannot exceed full nearness');
  const zoomRig=new Animator({idle:0,mechanical:0});
  zoomRig.setPresence(true,zoomFor(0.15,0.25));
  for(let i=0;i<120;i++) gazeParams=zoomRig.step(1/30);
  assert.equal(gazeParams.zoom,1);
  zoomRig.setPresence(true,zoomFor(0.2,0.5));
  for(let i=0;i<120;i++) gazeParams=zoomRig.step(1/30);
  assert.ok(gazeParams.zoom>1 && gazeParams.zoom<1.6,'Zoom increases as the face approaches');
  zoomRig.setPresence(true,zoomFor(0.2,2/3));
  const beforeZoom=gazeParams.zoom;
  gazeParams=zoomRig.step(1/30);
  assert.ok(gazeParams.zoom>beforeZoom && gazeParams.zoom<1.6,'Zoom approaches its target smoothly');
  for(let i=0;i<120;i++) gazeParams=zoomRig.step(1/30);
  assert.ok(Math.abs(gazeParams.zoom-1.6)<0.001);
  assert.match(context.GladosRig.renderEye(gazeParams),/scale\(1\.6\)/,'Nearness scales the entire eye');

  // Animation runs automatically, with no background rendering in other views or hidden tabs.
  vm.runInContext("go('vitals')", context);
  assert.equal(frames.size, 0);
  assert.equal(node('avatar-optic-source').src,'','Leaving the front page releases the optic stream');
  vm.runInContext("go('home')", context);
  assert.equal(frames.size, 1);
  assert.match(node('avatar-optic-source').src,/overlay=0/);
  media.matches = true; mediaEvents.change();
  assert.equal(frames.size, 0);
  updated.speaking = false; updated.lanes.priority.inflight = 0;
  updated.emotion = {pleasure: 0.7, arousal: 0.1, dominance: 0.7};
  stream.send("state", updated);
  assert.equal(expression(), "smug");
  assert.match(node("glados-eye").innerHTML, /#F6D36B/);
  assert.equal(frames.size, 0);
  media.matches = false; mediaEvents.change();
  document.hidden = true; docEvents.visibilitychange();
  assert.equal(frames.size, 0);
  assert.equal(node('avatar-optic-source').src,'','Hidden tabs stop loading preview frames');
  document.hidden = false; docEvents.visibilitychange();
  assert.equal(frames.size, 1);
  assert.match(node('avatar-optic-source').src,/overlay=0/);
  snapshot.controls={available:true,microphone_muted:false,voice_muted:false,native_audio:true,user_transcripts:false,model:'E4B',audio_mode:'Direct audio'};
  snapshot.vision_mind={...vision,running:true,paused:false,camera:{enabled:true,connected:true}};
  snapshot.autonomy_core={status:'Waiting for a useful update',pending_updates:2,enabled:true,next_check_s:40,
    last_decision:{outcome:'silent',reason:'No new task result'}};
  stream.send('state',snapshot);
  assert.match(node('autonomy-core-status').textContent,/2 pending updates/);
  assert.match(node('autonomy-core-status').textContent,/idle check in 40s/);
  assert.match(node('autonomy-core-status').textContent,/Last decision: silent — No new task result/);
  stream.send('state',{autonomy_core:{...snapshot.autonomy_core,stage:'central',
    last_decision:{outcome:'prompt',reason:'New GPU alert',instruction:'Report GPU temperature',slot_ids:['health']}}});
  assert.match(node('autonomy-core-handoff').textContent,/Report GPU temperature · Sources: health/);
  assert.equal(node('control-microphone').disabled,false);
  snapshot.search_settings={weather:['dwd.de'],news:['reuters.com'],reddit:[],general:[]};
  stream.send('state',snapshot);
  assert.equal(node('search-sources-weather').value,'dwd.de');
  assert.equal(node('search-settings-save').disabled,false);
  node('search-sources-reddit').value='reddit.com/r/LocalLLaMA\nreddit.com/r/Munich';
  node('search-sources-reddit').oninput();
  stream.send('state',{search_settings:{...snapshot.search_settings,reddit:['reddit.com/r/technology']}});
  assert.match(node('search-sources-reddit').value,/LocalLLaMA/,'Live updates preserve unsaved source edits');
  await node('search-settings-form').listeners.submit({preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(posts.at(-1),{url:'/api/search/settings',body:{weather:['dwd.de'],news:['reuters.com'],
    reddit:['reddit.com/r/LocalLLaMA','reddit.com/r/Munich'],general:[]}});
  assert.match(node('search-settings-feedback').textContent,/sources saved/);
  node('search-sources-weather').value=Array.from({length:9},(_,i)=>'site'+i+'.org').join('\n');
  node('search-sources-weather').oninput();
  const sourceWrites=posts.length;
  await node('search-settings-form').listeners.submit({preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(posts.length,sourceWrites);
  assert.match(node('search-settings-feedback').textContent,/at most eight/);
  stream.send('state',{search_settings:null});
  assert.equal(node('search-settings-save').disabled,true);
  assert.equal(node('control-camera').textContent,'Camera: ON');
  assert.equal(node('control-camera').attributes['aria-pressed'],'true');
  node('vision-interval-min').value='3';node('vision-interval-max').value='6';node('vision-interval-min').oninput();
  stream.send('state',{vision_mind:{...snapshot.vision_mind,interval_min_s:2,interval_max_s:5}});
  assert.equal(node('vision-interval-min').value,'3','Live updates preserve unsaved interval edits');
  await node('vision-settings-form').listeners.submit({preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(posts.at(-1),{url:'/api/vision/settings',body:{interval_min_s:3,interval_max_s:6}});
  assert.match(node('vision-settings-feedback').textContent,/saved: 3–6 seconds/);
  node('vision-interval-min').value='0';node('vision-interval-min').oninput();
  const writesBefore=posts.length;
  await node('vision-settings-form').listeners.submit({preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(posts.length,writesBefore);
  assert.match(node('vision-settings-feedback').textContent,/whole seconds/);
  stream.send('state',{vision_mind:{...snapshot.vision_mind}});
  snapshot.vision_mind.paused=true;
  await node('control-camera').onclick();
  assert.deepEqual(posts.at(-1),{url:'/api/minds/control',body:{agent_id:'vision',action:'pause'}});
  assert.equal(node('control-camera').textContent,'Camera: OFF');
  await node('control-camera').onclick();
  assert.deepEqual(posts.at(-1),{url:'/api/minds/control',body:{agent_id:'vision',action:'resume'}});
  stream.send('state',{vision_mind:null});
  assert.equal(node('control-camera').disabled,true);
  assert.equal(node('control-camera').textContent,'Camera: unavailable');
  assert.equal(node('vision-settings-save').disabled,true);
  assert.equal(html.slice(html.indexOf('<section class="view active" id="view-home">'),html.indexOf('<!-- CORES -->')).includes('id="control-transcripts"'),false);
  assert.equal(html.slice(html.indexOf('<section class="view" id="view-settings">')).includes('id="control-transcripts"'),true);
  await node('control-microphone').onclick();
  assert.deepEqual(posts.at(-1),{url:'/api/control',body:{action:'microphone',enabled:false}});
  assert.equal(node('control-microphone').textContent,'Microphone: off');
  failPost=true;
  await node('control-microphone').onclick();
  assert.equal(node('console-feedback').textContent,'Device unavailable');
  assert.equal(node('control-microphone').textContent,'Microphone: off','Failed writes never change displayed state');
  failPost=false;
  node('edit-instructions').onclick();
  node('operator-instructions').value='Speak briefly.';
  await node('save-instructions').onclick();
  assert.deepEqual(posts.at(-1),{url:'/api/instructions',body:{instructions:'Speak briefly.'}});
  node('message-text').value='Hello GLaDOS';
  node('message-form').listeners.submit({preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(posts.at(-1),{url:'/api/input',body:{text:'Hello GLaDOS'}});
  assert.equal(node('message-text').value,'');
  stream.send('obs',{source:'webapp',kind:'user_input',message:'Hello'});
  stream.send('obs',{source:'tts',kind:'synthesize',message:'Oh.'});
  stream.send('obs',{source:'tts',kind:'synthesize',message:'It is you.'});
  assert.equal(node('latest-reply').textContent,'Oh. It is you.');
  stream.send('obs',{source:'asr',kind:'transcript',message:'Hello GLaDOS',
    meta:{backend:'gemma',record_id:'voice-turn',deferred:true}});
  assert.equal(node('latest-reply').textContent,'Oh. It is you.','Late transcripts preserve the completed reply');
  stream.send('obs',{source:'audio',kind:'user_input',message:'Voice input received'});
  assert.equal(node('latest-reply').textContent,'Waiting for GLaDOS…');
  stream.onerror();
  assert.equal(node('control-microphone').disabled,true);
  assert.equal(node('message-send').disabled,true);

  // The overview uses actual configured capacity and queued requests, including safe HTML rendering.
  stream.onopen();
  stream.send('state',{inference:{capacity:3,reserved_interactive:1,
    active:[{slot:2,owner:'Vision',lane:'autonomy',started_at:Date.now()/1000}],
    waiting:[{owner:'<script>bad</script>',lane:'autonomy',wait_reason:'user_response'}],
    interaction_hold:{generation:42}},vision_mind:{...vision,scene:'A person holding a mug.',inference_ms:421,captured_at:Date.now()/1000}});
  assert.equal((node('overview-slots').innerHTML.match(/capacity-number/g)||[]).length,3);
  assert.match(node('overview-slots').innerHTML,/Vision Core/);
  assert.match(node('overview-queue').textContent,/user response/);
  assert.match(node('vision-caption').textContent,/person holding a mug/);
  assert.match(node('vision-caption-meta').textContent,/421 ms/);

  // Timings correlate the first response milestones; old turns and autonomy cannot finish this turn.
  const obs=(offset,source,kind,meta={},message='event')=>stream.send('obs',{timestamp:100+offset/1000,source,kind,meta,message});
  for (const meta of [{generation:null},{}]) {
    obs(0,'text','user_input',meta,'Uncorrelated input');
    assert.equal(node('turn-latency').textContent,'timing unavailable');
    obs(10,'tts','play',{generation:41});
    assert.equal(node('turn-latency').textContent,'timing unavailable');
    assert.doesNotMatch(node('turn-waterfall').innerHTML,/Playback requested/);
  }
  obs(0,'audio','user_input',{generation:42},'Voice input received');
  assert.equal(node('turn-latency').textContent,'Turn in progress');
  assert.match(node('latest-input').textContent,/Voice input/);
  obs(20,'llm','admitted',{generation:41,lane:'priority'});
  obs(30,'tts','play',{generation:42,autonomy:true});
  assert.doesNotMatch(node('turn-waterfall').innerHTML,/Playback requested/);
  obs(70,'llm','routed',{generation:42,lane:'priority',action:'reply'});
  obs(110,'llm','admitted',{generation:42,lane:'priority'});
  obs(510,'llm','first_token',{generation:42,lane:'priority'});
  obs(610,'tts','synthesize',{generation:42});
  obs(750,'tts','ready',{generation:42});
  obs(780,'tts','play',{generation:42});
  assert.match(node('turn-latency').textContent,/0.78 s.*playback requested/);
  assert.match(node('turn-waterfall').innerHTML,/Routing completed.*70 ms/s);
  obs(1200,'tts','play',{generation:42});
  assert.match(node('turn-latency').textContent,/0.78 s/,'Subsequent clauses do not change first-playback latency');
  obs(1400,'audio','user_input',{generation:43});
  obs(1500,'asr','transcript',{generation:42},'Stale transcript');
  assert.doesNotMatch(node('latest-input').textContent,/Stale transcript/);
  obs(1510,'asr','transcript',{generation:43},'What is the time?');
  assert.equal(node('latest-input').textContent,'What is the time?');
  obs(1800,'tts','play',{generation:43,muted:true});
  assert.match(node('turn-latency').textContent,/text delivered \(voice muted\)/);
  obs(2000,'audio','user_input',{generation:44});
  stream.send('state',{inference:{capacity:3,reserved_interactive:1,active:[],waiting:[],
    last_interaction_release:{generation:44,reason:'no_response'}}});
  assert.equal(node('turn-latency').textContent,'no response recorded');

  // Theme changes reuse the existing avatar and camera; no separate webcam or simulated analyser.
  vm.runInContext("go('home')",context);
  stream.send('state',{vision_mind:{...vision,running:true,paused:false,camera:{enabled:true,connected:true}},
    controls:{available:true,quiet_mode:false,microphone_muted:false}});
  node('avatar-optic-source').naturalWidth=320; node('avatar-optic-source').naturalHeight=180;
  for (const [theme,palette] of [['white','whitehot'],['terminal','ascii'],['potato','dither'],['dark','thermal']]) {
    node('theme-choice').value=theme; node('theme-choice').onchange();
    frame(clockNow+100);
    assert.equal(optics[0].palette,palette);
    assert.equal(document.documentElement.dataset.theme,theme);
    assert.equal(node('theme-settings').value,theme);
    assert.match(node('surface-switch').href,new RegExp('surface=presence&theme='+theme));
    assert.match(node('avatar-optic-source').src,/\/api\/vision\/live\?overlay=0/);
  }
  // Routing disclosures survive identical telemetry, and handled reports lose their alert.
  const overviewProbe=context.GladosOverview.mount(document,{
    escape:vm.runInContext('esc',context),coreOwner:id=>id,
    cores:()=>[{id:'health',title:'Health Core'}],identity:c=>c,
    activity:()=>({label:'Standby',className:'s-idle'}),describe:o=>o.description,
  });
  const routeProbe={latest:{action:'reply',accepted:true,stages:[{name:'Capability',accepted:true,
    option_id:'conversation',margin:.8,scores:[{id:'conversation',label:'A',probability:.9,
      description:'A detailed conversation instruction with <untrusted> content.'}]}]}};
  const slotProbe={owner_id:'health',title:'Health Core',status:'active',updated_at:Date.now()/1000,
    update_priority:'important',handled:false,summary:'GPU temperature elevated.'};
  overviewProbe.update({connection:'live',routing:routeProbe,slots:[slotProbe]});
  assert.match(node('overview-route').innerHTML,/<strong>Conversation<\/strong>/);
  assert.match(node('overview-route').innerHTML,/<details class="route-details">/);
  assert.match(node('overview-route').innerHTML,/&lt;untrusted&gt;/);
  assert.match(node('overview-cores').innerHTML,/core-important.*awaiting Autonomy/s);
  const routeWrites=node('overview-route').htmlWrites;
  overviewProbe.update({connection:'live',routing:JSON.parse(JSON.stringify(routeProbe)),
    slots:[{...slotProbe,handled:true}]});
  assert.equal(node('overview-route').htmlWrites,routeWrites,'Telemetry preserves an open choices disclosure');
  assert.doesNotMatch(node('overview-cores').innerHTML,/core-important|awaiting Autonomy/);
  assert.doesNotMatch(node('presence-thoughts').innerHTML,/class="thought important"/);
  const customRoute={latest:{...routeProbe.latest,stages:[{...routeProbe.latest.stages[0],
    option_id:'0123456789abcdef0123456789abcdef',scores:[{id:'0123456789abcdef0123456789abcdef',
      label:'A',probability:.9,description:'My custom conversational option'}]}]}};
  overviewProbe.update({connection:'live',routing:customRoute,controls:{autonomy_enabled:false},slots:[slotProbe]});
  assert.match(node('overview-route').innerHTML,/<strong>My custom conversational option<\/strong>/);
  assert.match(node('overview-cores').innerHTML,/important update/);
  assert.doesNotMatch(node('overview-cores').innerHTML,/awaiting Autonomy/);
  assert.equal(vm.runInContext("coreActivity({id:'probe',execution_status:'error',running:false}).label",context),'Error');
  rootEvents.pagehide();
  assert.equal(frames.size, 0);
  assert.equal(rootEvents.pointermove, undefined);
  console.log("Webapp client and avatar regression checks passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
