/* Bind the supplied aperture animation to the existing console telemetry. */
(function (root) {
  "use strict";
  const R = root.GladosRig;

  // These thresholds select visual expressions; they do not change engine emotion.
  function expression(emotion) {
    if (!emotion || !emotion.every(Number.isFinite)) return "neutral";
    const [pleasure, arousal, dominance] = emotion;
    if (pleasure < -0.3 && arousal > 0.3) return "angry glare";
    if (pleasure < -0.3 && arousal < -0.3) return "bored";
    if (pleasure < -0.3) return dominance > 0.3 ? "suspicious" : "disappointed";
    if (arousal > 0.6 && dominance < -0.3) return "surprised";
    if (pleasure > 0.3 && dominance > 0.3) return "smug";
    if (arousal > 0.3 || dominance < -0.3) return "quizzical";
    return "neutral";
  }

  function presentation(state, recentInteraction = false, localizedMotion = false) {
    const connection = state.connection || "connecting";
    if (connection === "connecting" || connection === "disconnected") {
      return { look: "offline", activity: connection === "connecting" ? "Connecting" : "Disconnected",
        detail: connection === "connecting" ? "Waiting for telemetry." : "Waiting for the engine to reconnect.",
        speaking: false, listening: false };
    }
    if (state.controls?.quiet_mode) return {look:"sleeping",activity:"Quiet",detail:"Listening only for a wake request.",speaking:false,listening:false,presence:false,cameraOn:false};
    const speaking = state.performance ? state.performance.active === true : !!state.speaking;
    const listening = !!state.audio?.vad && !speaking;
    const inflight = state.prioInflight || 0;
    const vision = state.visionMind, camera = vision?.camera, face = camera?.face;
    const cameraOn = !!vision && !vision.paused && !!camera?.enabled;
    const fresh = cameraOn && !!camera?.connected &&
      face?.available === true && Number.isFinite(face.observed_at) &&
      Math.abs(Date.now() / 1000 - face.observed_at) < (face.source === "e4b" ? 6 : 3);
    const presence = localizedMotion || !fresh || face.present === true || face.absent_seconds < 1.5;
    // The camera faces the viewer: horizontal image coordinates run opposite
    // to the viewer's movement across the screen. Keep preview coordinates raw.
    const toGaze = face => {
      if (!Number.isFinite(face.x) || !Number.isFinite(face.y)) return null;
      const size = face.bbox && [face.bbox[2],face.bbox[3]].every(v => Number.isFinite(v) && v > 0 && v <= 1) ?
        {width:face.bbox[2],height:face.bbox[3]} : null;
      // Either dimension filling two-thirds of the image is fully near.
      const extent = size ? Math.max(size.width,size.height) : face.closeness;
      const closeness = Number.isFinite(extent) ? Math.max(0,Math.min(1,extent/(2/3))) : undefined;
      return { kind:'face', x:-Math.max(-1,Math.min(1,face.x)), y:Math.max(-1,Math.min(1,face.y)), closeness, size };
    };
    const gazes = fresh && face.present === true ?
      (Array.isArray(face.faces) ? face.faces : [face]).map(toGaze).filter(Boolean) : [];
    const gaze = gazes[0] || null;
    const recentlyAddressed = recentInteraction || (state.sinceUser != null && state.sinceUser < 5);
    const sleeping = fresh && face.present === false && face.absent_seconds >= (face.sleep_after_s || 5) &&
      !speaking && !listening && inflight === 0 && !recentlyAddressed && !localizedMotion;
    const cue = state.performance?.emotion;
    const directed = speaking && typeof cue === "string" && Object.hasOwn(R.LOOKS, cue);
    const look = directed ? cue : sleeping ? "sleeping" : !speaking && !listening && inflight > 0 ? "processing" : expression(state.emo);
    const activity = speaking ? "Speaking" : listening ? "Listening" : inflight > 0 ? "Processing" : sleeping ? "Sleeping" : "Idle";
    const detail = speaking ? "Voice output active." : listening ? "Microphone voice activity detected." :
      inflight > 0 ? `${inflight} inference${inflight === 1 ? "" : "s"} in progress.` :
      sleeping ? "No face detected. I'll wake when you return or speak." : "Ready when you are.";
    return { look, activity: connection === "demo" ? `Demo · ${activity}` : activity,
      detail, speaking, listening, presence, gaze, gazes, cameraOn };
  }

  // Keep a face anchor across brief movement glances and detector reorderings.
  class GazeSelector {
    constructor() { this.target=null;this.face=null;this.switchAt=0;this.motionUntil=0;this.nextMotionAt=0; }
    select(gazes,now,motions=[]) {
      const sorted=[...(gazes||[])].sort((a,b)=>a.x-b.x);
      let index=0;
      if (this.face) {
        let distance=Infinity;
        sorted.forEach((g,i)=>{
          const d=Math.hypot(g.x-this.face.x,g.y-this.face.y);
          if(d<distance){distance=d;index=i;}
        });
      } else this.switchAt=now+3;
      if(sorted.length && now>=this.switchAt){index=(index+1)%sorted.length;this.switchAt=now+3;}
      this.face=sorted[index]||null;
      if(!this.face)this.switchAt=0;
      // Head/eye motion is already attended to through face tracking. Hand and
      // other movement outside those boxes can briefly interrupt eye contact.
      const candidates=motions.filter(m=>Number.isFinite(m.x) && Number.isFinite(m.y) &&
        !sorted.some(f=>Math.abs(m.x-f.x)<(f.size?.width||0.15)*1.2 &&
                        Math.abs(m.y-f.y)<(f.size?.height||0.2)*1.2));
      if(this.target?.kind==='motion' && now<this.motionUntil) {
        const nearest=[...candidates].sort((a,b)=>Math.hypot(a.x-this.target.x,a.y-this.target.y)-Math.hypot(b.x-this.target.x,b.y-this.target.y))[0];
        if(nearest && Math.hypot(nearest.x-this.target.x,nearest.y-this.target.y)<0.4)this.target=nearest;
        if(candidates.length || this.face)return this.target;
      }
      if(candidates.length && (!this.face || now>=this.nextMotionAt)) {
        this.target=candidates[0];this.motionUntil=now+0.9;this.nextMotionAt=now+3;
        this.switchAt=Math.max(this.switchAt,now+1.9);
        return this.target;
      }
      this.target=this.face;
      return this.target;
    }
  }

  function mount(doc) {
    const svg = doc.getElementById("glados-eye");
    const card = doc.getElementById("avatar-card");
    const feed = doc.getElementById("avatar-optic");
    const canvas = doc.getElementById("avatar-optic-view");
    const source = doc.getElementById("avatar-optic-source");
    const motionLabel = doc.getElementById("optic-motion"), subjectLabel = doc.getElementById("optic-subject");
    const Optic = root.GladosVision.ThemedVision || root.GladosVision.Vision;
    const optic = new Optic(canvas);
    optic.setMode("thermal");
    const media = root.matchMedia("(prefers-reduced-motion: reduce)");
    const animator = new R.Animator();
    let current = presentation({}), visible = true, destroyed = false;
    let frameId = null, lastFrame = null, lastPointer = 0;
    let lastState = {}, lastInteraction = 0;
    let feedRunning = false, retryFeedAt = 0, lastInference = null;
    const gazeSelector=new GazeSelector();
    function selectGaze() {
      const previous=current,base=presentation(lastState,Date.now()-lastInteraction<5000);
      const motions=base.cameraOn && feedRunning?optic.motionTargets:[];
      current=motions.length?presentation(lastState,Date.now()-lastInteraction<5000,true):base;
      if(current.look!==previous.look)animator.setLook(current.look);
      const next=gazeSelector.select(current.gazes,root.performance.now()/1000,motions);
      if(next && previous.gaze && (next.kind!==previous.gaze.kind || Math.hypot(next.x-previous.gaze.x,next.y-previous.gaze.y)>0.4)) animator.lookAt(null);
      current.gaze=next;
      // Motion supplies direction, not a new distance estimate for the viewer.
      const closeness=next?.kind==='motion'?gazeSelector.face?.closeness:next?.closeness;
      animator.setPresence(!!next || current.presence!==false || current.speaking || current.listening,closeness ?? 0.3);
      svg.setAttribute("aria-label", `GLaDOS aperture: ${current.activity.toLowerCase()}, ${current.look}`);
    }
    animator.setLook(current.look);
    const animate = () => !destroyed && visible && !doc.hidden && !media.matches;
    function stopFeed() {
      source.removeAttribute("src");source.dataset.streaming="";
      optic.setSource(null);feedRunning=false;
      motionLabel.textContent='MOTION 0.00';subjectLabel.hidden=true;
      const context = canvas.getContext("2d");
      context.clearRect(0,0,canvas.width,canvas.height);
    }
    function syncFeed() {
      card.classList.toggle("camera-on", !!current.cameraOn);
      feed.hidden = !current.cameraOn;
      const ready = current.cameraOn && lastState.visionMind?.camera?.connected && visible && !doc.hidden;
      if (!ready) {
        if (feedRunning) stopFeed();
        return;
      }
      if (!feedRunning && Date.now() >= retryFeedAt) {
        feedRunning=true;source.dataset.streaming="true";
        // Reuse the backend camera; face overlays belong to the Vision inspector.
        source.src="/api/vision/live?overlay=0&started="+Date.now();
        optic.setSource(source);
      }
    }
    source.onerror=()=>{stopFeed();retryFeedAt=Date.now()+1000;};
    function draw(params, dt = 1 / 30) {
      const theme = root.GladosThemes?.current();
      let eye = R.renderEye(params, { id: "glados-avatar" });
      if (theme) {
        eye = eye.replaceAll('#E8ECEF', theme.ink).replaceAll('#0E1116', theme.bg).replaceAll('#1A2029', theme.grad);
        if (optic.setTheme && optic.palette !== theme.optic) optic.setTheme(theme);
      }
      svg.innerHTML = eye;
      if (feedRunning) {
        optic.reducedMotion=media.matches;
        optic.setEye(params);
        optic.setSubject(current.gaze ? {x:params.lookX,y:params.lookY,kind:current.gaze.kind} : null);
        const observation=optic.frame(dt);
        motionLabel.textContent='MOTION '+observation.motion.toFixed(2);
        subjectLabel.hidden=!observation.ok || !current.gaze;
        subjectLabel.textContent=current.gaze?.kind==='motion'?'MOVEMENT':'TEST SUBJECT';
      }
    }
    function still() {
      const gaze = current.gaze;
      const closeness=gaze?.kind==='motion'?gazeSelector.face?.closeness:gaze?.closeness;
      const lean = Math.max(0,Math.min(1,((closeness||0)-0.6)/0.4));
      draw({ ...R.NEUTRAL, ...R.LOOKS[current.look], zoom:1+0.6*lean, listening:current.listening ? 1 : 0,
        ...(gaze ? {lookX:gaze.x, lookY:gaze.y} : {}) });
    }
    function frame(now) {
      frameId = null;
      if (!animate()) return;
      if (lastFrame === null || now - lastFrame >= 1000 / 30) {
        const dt = lastFrame === null ? 1 / 30 : Math.min((now - lastFrame) / 1000, 0.1);
        lastFrame = now;
        selectGaze();
        if (current.cameraOn) {
          if (current.gaze) animator.lookAt(current.gaze.x, current.gaze.y, true, current.gaze.size,current.gaze.kind);
          else animator.lookAt(null);
        } else if (now - lastPointer > 6000) animator.lookAt(null);
        // SSE exposes speaking, not the TTS waveform. This is an illustrative
        // speech pulse, never microphone RMS or a second audio playback stream.
        const pulse = current.speaking ? 0.25 + 0.55 * Math.pow(Math.sin(now / 140), 2) : 0;
        animator.setVoice(pulse);
        draw(animator.step(dt),dt);
      }
      frameId = root.requestAnimationFrame(frame);
    }
    function sync() {
      if (frameId !== null) root.cancelAnimationFrame(frameId);
      frameId = null; lastFrame = null;
      syncFeed();
      if (animate()) frameId = root.requestAnimationFrame(frame);
      else if (visible && !doc.hidden) still();
    }
    function update(state) {
      lastState = state;
      const previousCamera=current.cameraOn;
      selectGaze();
      const sequence=state.visionMind?.inference_sequence;
      if(Number.isFinite(sequence)) {
        if(lastInference!==null && sequence>lastInference && current.cameraOn && animate()) animator.inferenceBlink();
        lastInference=Math.max(lastInference ?? 0,sequence);
      }
      if(state.connection==='disconnected') lastInference=null;
      if (current.gaze) animator.lookAt(current.gaze.x, current.gaze.y, true, current.gaze.size,current.gaze.kind);
      else if (current.cameraOn || previousCamera!==current.cameraOn || current.look === "sleeping" || current.look === "offline") {
        animator.lookAt(null);
      }
      animator.setListening(current.listening);
      if (!current.speaking) animator.setVoice(0);
      syncFeed();
      if (!animate() && visible && !doc.hidden) still();
    }
    function pointer(event) {
      if (!animate() || current.look === "offline" || lastState?.controls?.quiet_mode || current.cameraOn || event.pointerType === "touch") return;
      const box = svg.getBoundingClientRect();
      if (!box.width || !box.height) return;
      const x = (event.clientX - box.left - box.width / 2) / (box.width / 2);
      const y = (event.clientY - box.top - box.height * 0.475) / (box.height / 2);
      const distance = Math.hypot(x, y) || 1;
      const amount = Math.tanh(distance * 1.4) / distance;
      animator.lookAt(x * amount, y * amount);
      lastPointer = root.performance.now();
    }
    const release = () => animator.lookAt(null);
    const wake = () => { lastInteraction = Date.now(); update(lastState); };
    const destroy = () => {
      destroyed = true;
      if (frameId !== null) root.cancelAnimationFrame(frameId);
      frameId = null;
      stopFeed();source.onerror=null;
      root.removeEventListener("pointermove", pointer);
      root.removeEventListener("blur", release);
      root.removeEventListener("pagehide", destroy);
      root.removeEventListener("pointerdown", wake);
      root.removeEventListener("keydown", wake);
      doc.removeEventListener("visibilitychange", sync);
      media.removeEventListener("change", sync);
    };
    root.addEventListener("pointermove", pointer, { passive: true });
    root.addEventListener("blur", release);
    root.addEventListener("pagehide", destroy);
    root.addEventListener("pointerdown", wake, { passive: true });
    root.addEventListener("keydown", wake);
    doc.addEventListener("visibilitychange", sync);
    media.addEventListener("change", sync);
    still(); sync();
    return { update, setVisible(value) { visible = value; release(); sync(); }, destroy };
  }

  root.GladosAvatar = { expression, presentation, GazeSelector, mount };
})(typeof window !== "undefined" ? window : globalThis);
