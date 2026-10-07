/* Live routing tree: local SVG view and Mermaid export, with no external renderer. */
(function (root) {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const short = (text, n) => text.length > n ? text.slice(0, n - 1) + '…' : text;
  const leafId = (node, option) => JSON.stringify([node, option]);
  function optionName(option) {
    if (option.display_name) return option.display_name;
    if (option.fallback) return 'No matching choice';
    if (option.tool) return 'Fixed: ' + option.tool;
    if (option.tool_scope?.length) return option.tool_scope.join(', ');
    return ({reply:'Reply',ignore:'Ignore background speech',clarify:'Ask for clarification',quiet:'Sleep · listening for wake',wake:'Wake · resume replies'})[option.action] || 'Assistant planning';
  }
  function selection(latest) {
    return new Map((latest?.stages || []).map(stage => [stage.list_id, stage]));
  }
  function graph(structure, expanded, latest) {
    const index = new Map(structure.nodes.map(node => [node.id, node]));
    const selected = selection(latest), nodes = [], edges = [];
    let cursor = 0;
    function visit(id, depth, ancestors) {
      const node = index.get(id);
      if (!node || ancestors.has(id)) return null;
      const stage = selected.get(id);
      const item = {id, title:node.name, detail:node.options.length + ' choices · ' + (expanded.has(id) ? 'collapse −' : 'expand +'),
        depth, branch:true, active:!!stage, uncertain:!!stage && !stage.accepted && !stage.fallback_selected, tooltip:node.name};
      nodes.push(item);
      const children = [];
      if (expanded.has(id)) {
        const path = new Set([...ancestors, id]);
        node.options.forEach((option, i) => {
          const active = !!(stage?.accepted || stage?.fallback_selected) && stage.option_id === option.id;
          let child;
          if (option.next) child = visit(option.next, depth + 1, path);
          else {
            child = {id:leafId(id,option.id), depth:depth+1, title:optionName(option),
              detail:option.fallback ? 'Configured fallback · no tool selected' : option.tool ? JSON.stringify(option.arguments || {}) : option.tool_scope?.length ? 'Assistant fills arguments' : option.description,
              tooltip:option.description + (option.tool ? '\n' + option.tool + ' ' + JSON.stringify(option.arguments || {}) : ''), active, y:cursor++ * 76 + 60};
            nodes.push(child);
          }
          if (!child) return;
          children.push(child);
          const score = active ? stage.scores?.find(s => s.id === option.id)?.probability : null;
          edges.push({from:item, to:child, active, label:String.fromCharCode(65+i) + (score == null ? '' : ' · ' + (score*100).toFixed(1)+'%')});
        });
      }
      item.y = children.length ? (children[0].y + children.at(-1).y)/2 : cursor++ * 76 + 60;
      return item;
    }
    visit(structure.root, 0, new Set());
    return {nodes, edges, width:Math.max(600,...nodes.map(n => n.depth*340+310)), height:Math.max(180,cursor*76+45)};
  }
  function svg(structure, expanded, latest) {
    const tree = graph(structure,expanded,latest);
    const edges = tree.edges.map(edge => {
      const x1=edge.from.depth*340+280, x2=edge.to.depth*340+20, y1=edge.from.y, y2=edge.to.y;
      return '<g class="route-edge'+(edge.active?' selected':'')+'"><path marker-end="url(#routing-arrow'+(edge.active?'-selected':'')+')" d="M '+x1+' '+y1+' C '+(x1+40)+' '+y1+', '+(x2-40)+' '+y2+', '+x2+' '+y2+'"/><text x="'+(x1+6)+'" y="'+(y2-9)+'">'+esc(edge.label)+'</text></g>';
    }).join('');
    const nodes = tree.nodes.map(node => {
      const x=node.depth*340+20, y=node.y-29;
      const interactive=node.branch?' role="button" tabindex="0" aria-expanded="'+expanded.has(node.id)+'" data-route-node="'+esc(node.id)+'"':'';
      return '<g class="route-node '+(node.branch?'branch':'terminal')+(node.active?' selected':'')+(node.uncertain?' uncertain':'')+'" transform="translate('+x+' '+y+')"'+interactive+' aria-label="'+esc(node.title+'; '+node.detail)+'"><title>'+esc(node.tooltip)+'</title><rect width="260" height="58" rx="10"/><text x="14" y="23">'+esc(short(node.title,33))+'</text><text class="route-detail" x="14" y="43">'+esc(short(node.detail,37))+'</text></g>';
    }).join('');
    const markers='<defs>'+['','-selected'].map(suffix=>'<marker id="routing-arrow'+suffix+'" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="'+(suffix?'#2ee6c8':'#32404f')+'"/></marker>').join('')+'</defs>';
    return '<svg xmlns="http://www.w3.org/2000/svg" width="'+tree.width+'" height="'+tree.height+'" viewBox="0 0 '+tree.width+' '+tree.height+'" role="group" aria-label="Routing decision tree. Click a decision to expand or collapse its choices.">'+markers+edges+nodes+'</svg>';
  }
  function mermaid(structure, latest) {
    const selected=selection(latest), lines=['flowchart LR'], ids=new Map(), highlighted=[], visited=new Set();
    let links=0;
    const id=key => {if(!ids.has(key))ids.set(key,'n'+ids.size);return ids.get(key);};
    // Mermaid decimal entities keep registry metadata inside quoted labels.
    const label=text => String(text).replace(/[\r\n]+/g,' ').replace(/[#"&<>|`{}\[\]\\]/g,c=>'#'+c.charCodeAt(0)+';');
    function visit(key) {
      if(visited.has(key))return;
      visited.add(key);
      const node=structure.nodes.find(n=>n.id===key);if(!node)return;
      lines.push('  '+id(key)+'{"'+label(node.name)+'"}');
      node.options.forEach((option,i)=>{
        const destination=option.next || leafId(key,option.id);
        if(!option.next) lines.push('  '+id(destination)+'["'+label(optionName(option)+(option.tool?' '+JSON.stringify(option.arguments||{}):''))+'"]');
        lines.push('  '+id(key)+' -->|'+String.fromCharCode(65+i)+'| '+id(destination));
        const stage=selected.get(key);
        if((stage?.accepted || stage?.fallback_selected) && stage.option_id===option.id)highlighted.push(links);
        links++;
        if(option.next)visit(option.next);
      });
    }
    visit(structure.root);
    if(highlighted.length)lines.push('  linkStyle '+highlighted.join(',')+' stroke:#2ee6c8,stroke-width:3px');
    return lines.join('\n');
  }
  root.GladosRouting={graph,svg,mermaid};
})(globalThis);
