/* Use this browser (e.g. a phone) as GLaDOS's microphone and speaker while the page is open. */
(function (root) {
  'use strict';
  const RATE = 16000, PING_MS = 2000, RETRY_MS = 3000;

  // Average each 1/16 kHz span of input samples; good enough for speech and VAD.
  function downsampler(inputRate) {
    const ratio = inputRate / RATE;
    let tail = new Float32Array(0), position = 0;
    return input => {
      const buffer = new Float32Array(tail.length + input.length);
      buffer.set(tail); buffer.set(input, tail.length);
      const out = new Int16Array(Math.max(0, Math.floor((buffer.length - position) / ratio)));
      let p = position, n = 0;
      while (n < out.length && p + ratio <= buffer.length) {
        const start = Math.floor(p), end = Math.max(start + 1, Math.floor(p + ratio));
        let sum = 0;
        for (let k = start; k < end; k++) sum += buffer[k];
        const value = Math.max(-1, Math.min(1, sum / (end - start)));
        out[n++] = value < 0 ? value * 0x8000 : value * 0x7fff;
        p += ratio;
      }
      const consumed = Math.floor(p);
      tail = buffer.slice(consumed); position = p - consumed;
      return out.subarray(0, n);
    };
  }

  function mount(doc, helpers = {}) {
    const $ = id => doc.getElementById(id);
    const mobile = !!root.matchMedia?.('(pointer: coarse)').matches;
    let wanted = false, session = null, retry = null, status = 'off', detail = '';

    function render() {
      const secure = root.isSecureContext !== false;
      const text = !secure ? 'Needs HTTPS: set webapp.tls: true and open the https:// address.' :
        status === 'live' ? 'This device is GLaDOS’s microphone and speaker. The PC microphone is ignored until you stop or close this page.' :
        status === 'connecting' ? 'Connecting…' :
        status === 'replaced' ? 'Another device took over the microphone.' :
        status === 'error' ? detail : 'GLaDOS uses the PC microphone and speakers.';
      for (const id of ['remote-audio-status', 'remote-audio-banner-text']) if ($(id)) $(id).textContent = text;
      for (const id of ['remote-audio-toggle', 'remote-audio-banner-toggle']) {
        const button = $(id);
        if (!button) continue;
        button.textContent = wanted ? 'Stop using this device' : 'Use this device’s microphone';
        button.disabled = !secure;
        button.setAttribute('aria-pressed', String(wanted));
      }
      const banner = $('remote-audio-banner');
      if (banner) banner.hidden = !(mobile || wanted);
    }

    function setStatus(value, message = '') { status = value; detail = message; render(); }

    async function open() {
      clearTimeout(retry); retry = null;
      setStatus('connecting');
      const media = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true,
        noiseSuppression: true, autoGainControl: true}});
      const Context = root.AudioContext || root.webkitAudioContext;
      const context = new Context();
      await context.resume();
      const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/api/audio/remote');
      ws.binaryType = 'arraybuffer';
      const current = {ws, media, context, source: null, track: null, pending: null, ping: null, lock: null};
      session = current;
      const input = context.createMediaStreamSource(media);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const mute = context.createGain(); mute.gain.value = 0;
      const shrink = downsampler(context.sampleRate);
      processor.onaudioprocess = event => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const pcm = shrink(event.inputBuffer.getChannelData(0));
        if (pcm.length) ws.send(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
      };
      input.connect(processor); processor.connect(mute); mute.connect(context.destination);
      const stopVoice = () => { const s = current.source; current.source = null; if (s) { s.onended = null; try { s.stop(); } catch (e) {} } };
      ws.onopen = () => {
        setStatus('live');
        current.ping = setInterval(() => ws.readyState === WebSocket.OPEN && ws.send('ping'), PING_MS);
        navigator.wakeLock?.request('screen').then(lock => { current.lock = lock; }, () => {});
      };
      ws.onmessage = event => {
        if (typeof event.data === 'string') {
          if (event.data.startsWith('play16:')) {
            const [, track, rate] = event.data.split(':');
            current.pending = {track, rate: Number(rate)};
          } else if (event.data === 'stop') stopVoice();
          else if (event.data === 'replaced') { wanted = false; close(current); setStatus('replaced'); }
          return;
        }
        const pending = current.pending; current.pending = null;
        if (!pending) return;
        const pcm = new Int16Array(event.data), samples = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
        if (!samples.length) { ws.send('played:' + pending.track); return; }
        stopVoice();
        const buffer = context.createBuffer(1, samples.length, pending.rate);
        buffer.copyToChannel(samples, 0);
        const voice = context.createBufferSource();
        voice.buffer = buffer; voice.connect(context.destination);
        voice.onended = () => { if (current.source === voice) { current.source = null; ws.readyState === WebSocket.OPEN && ws.send('played:' + pending.track); } };
        current.source = voice; voice.start();
      };
      ws.onclose = () => {
        if (session !== current) return;
        close(current);
        if (wanted) { setStatus('connecting'); retry = setTimeout(() => start().catch(fail), RETRY_MS); }
      };
    }

    function close(current) {
      if (!current) return;
      if (session === current) session = null;
      clearInterval(current.ping);
      current.lock?.release?.().catch?.(() => {});
      try { current.ws.close(); } catch (e) {}
      current.media.getTracks().forEach(track => track.stop());
      current.context.close().catch(() => {});
    }

    function fail(error) {
      wanted = false; close(session);
      setStatus('error', error?.name === 'NotAllowedError' ? 'Microphone permission was refused.' : (error?.message || 'Remote audio failed.'));
    }

    const start = () => session ? Promise.resolve() : open();
    function toggle() {
      wanted = !wanted;
      if (wanted) start().catch(fail);
      else { clearTimeout(retry); close(session); setStatus('off'); }
    }
    for (const id of ['remote-audio-toggle', 'remote-audio-banner-toggle']) if ($(id)) $(id).onclick = toggle;
    render();
    return {render, get active() { return status === 'live'; }};
  }

  root.GladosRemoteAudio = {downsampler, mount};
})(typeof window !== 'undefined' ? window : globalThis);
