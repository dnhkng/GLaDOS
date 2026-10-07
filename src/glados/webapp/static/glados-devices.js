/* Select the actual host capture/playback backends, without browser-media dependencies. */
(function (root) {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g,
    char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  function options(devices, selected, audio = false) {
    const rows = audio ? [{id:'default', name:'System default'}, ...devices] : [...devices];
    const value = selected == null ? 'default' : String(selected);
    if (!rows.some(row => String(row.id) === value)) rows.push({id:value, name:'Current device · '+value});
    return rows.map(row => '<option value="' + escape(row.id) + '"' +
      (String(row.id) === value ? ' selected' : '') + '>' + escape(row.name) +
      (row.host_api ? ' · '+escape(row.host_api) : '') +
      (!audio && row.id !== 'default' ? ' · '+escape(row.id) : '') + '</option>').join('');
  }
  function mount(document, connected) {
    const $ = id => document.getElementById(id);
    let visible = false, pending = false, state = null, lastConnected = connected();
    function render() {
      const audio = state?.audio, camera = state?.camera;
      for (const [kind, devices, selected, available] of [
        ['camera', camera?.devices || [], camera?.selected, camera?.available],
        ['microphone', audio?.input || [], audio?.selected_input, audio?.available],
        ['speaker', audio?.output || [], audio?.selected_output, audio?.available],
      ]) {
        const select = $('device-'+kind);
        select.innerHTML = available ? options(devices, selected, kind !== 'camera') : '<option>Unavailable</option>';
        select.value = selected == null ? 'default' : String(selected);
        select.disabled = pending || !connected() || !available || (kind === 'camera' && !devices.length);
      }
      $('device-refresh').disabled = pending || !connected();
    }
    async function request(body) {
      if (pending) return;
      if (!connected()) { render(); $('device-feedback').textContent='Connect the engine to select host devices.'; return; }
      pending=true; render();
      $('device-feedback').textContent=body ? 'Switching '+body.kind+'…' : 'Listing connected devices…';
      try {
        const response=await root.fetch('/api/devices', body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {cache:'no-store'});
        const result=await response.json();
        if (!response.ok) throw new Error(result.error || 'Devices could not be updated.');
        state=result;
        $('device-feedback').textContent=body ? 'Selected '+body.kind+' updated.' :
          (result.audio.available ? 'Connected devices refreshed.' : result.audio.reason);
      } catch (error) {
        $('device-feedback').textContent=error.message;
      } finally { pending=false; render(); }
    }
    $('device-refresh').onclick=()=>request();
    for (const kind of ['camera','microphone','speaker']) $('device-'+kind).onchange=()=>{
      const value=$('device-'+kind).value;
      return request({kind,device:kind === 'camera' ? value : value === 'default' ? null : Number(value)});
    };
    return {setVisible(value) {const opening=value && !visible;visible=value;if(opening) request();},refresh:()=>request(),
      updateConnection() {const current=connected();if(current!==lastConnected){lastConnected=current;render();if(current && visible) request();}}};
  }
  root.GladosDevices={mount,options};
})(window);
