const askBtn = document.getElementById('ask');
const questionEl = document.getElementById('question');
const answerEl = document.getElementById('answer');
const sqlEl = document.getElementById('sql');
const memoryEl = document.getElementById('memory');
const querycraftEl = document.getElementById('querycraft');
const presenterEl = document.getElementById('presenter');
const uploadInput = document.getElementById('upload');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');
const uploadControls = document.getElementById('uploadControls');
const modeToggle = document.getElementById('modeToggle');
let currentDocId = null;
let currentMode = 'text2sql';
const statusEl = document.getElementById('status');
const timeEl = document.getElementById('time');
const recalcBtn = document.getElementById('recalc');
const spinner = document.getElementById('spinner');
const spinnerText = document.getElementById('spinnerText');
const askBtnEl = askBtn;

// Mode toggle logic
modeToggle.addEventListener('change', function() {
  currentMode = modeToggle.value;
  console.log('[DEBUG] Mode changed to:', currentMode); // Debug log
  if(currentMode === 'ocr_qa') {
    uploadControls.classList.remove('hidden');
    console.log('[DEBUG] Upload controls shown'); // Debug log
  } else {
    uploadControls.classList.add('hidden');
    currentDocId = null;
    uploadStatus.textContent = 'No upload';
    uploadStatus.className = 'status-text';
    console.log('[DEBUG] Upload controls hidden'); // Debug log
  }
});

// Initialize UI state
console.log('[DEBUG] Initial mode:', currentMode); // Debug log
if(currentMode === 'ocr_qa') {
  uploadControls.classList.remove('hidden');
} else {
  uploadControls.classList.add('hidden');
}

// Helper to programmatically switch to OCR Q&A mode
function switchToOcrQaMode() {
  if (modeToggle.value !== 'ocr_qa') {
    modeToggle.value = 'ocr_qa';
    currentMode = 'ocr_qa';
    uploadControls.classList.remove('hidden');
    // Fire a change event so any listeners update state consistently
    const ev = new Event('change', { bubbles: true });
    modeToggle.dispatchEvent(ev);
    console.log('[DEBUG] Switched to OCR Q&A mode');
  }
}

// Extract mode change logic into a function so it can be called directly
function onModeChange() {
  currentMode = modeToggle.value;
  console.log('[DEBUG] onModeChange called, mode:', currentMode); // Debug log
  if(currentMode === 'ocr_qa') {
    uploadControls.classList.remove('hidden');
    console.log('[DEBUG] Upload controls shown via onModeChange'); // Debug log
  } else {
    uploadControls.classList.add('hidden');
    currentDocId = null;
    uploadStatus.textContent = 'No upload';
    uploadStatus.className = 'status-text';
    console.log('[DEBUG] Upload controls hidden via onModeChange'); // Debug log
  }
}

// Mode toggle logic
modeToggle.addEventListener('change', onModeChange);

// Upload handler: create a temporary file input on demand to avoid issues with a permanently hidden input.
if (uploadBtn) {
  uploadBtn.addEventListener('click', () => {
    // If not in OCR Q&A mode, switch and highlight, then open file picker after a delay
    if (modeToggle.value !== 'ocr_qa') {
      switchToOcrQaMode();
      // Visually highlight the dropdown to show the switch
      modeToggle.style.boxShadow = '0 0 0 3px rgba(15, 98, 254, 0.2)';
      modeToggle.style.transition = 'box-shadow 0.2s';
      setTimeout(() => {
        modeToggle.style.boxShadow = '';
        // Now open the file picker
        const tmp = document.createElement('input');
        tmp.type = 'file';
        tmp.accept = 'image/*';
        tmp.style.position = 'absolute';
        tmp.style.left = '-9999px';
        document.body.appendChild(tmp);

        tmp.addEventListener('change', async () => {
          const files = tmp.files;
          if (!files || files.length === 0) {
            uploadStatus.textContent = 'No file selected';
            document.body.removeChild(tmp);
            return;
          }
          switchToOcrQaMode();
          uploadStatus.textContent = 'Uploading...';
          uploadStatus.className = 'status-text text-warning';
          const fd = new FormData();
          fd.append('file', files[0]);
          try {
            const r = await fetch('/upload-image', { method: 'POST', body: fd });
            if (!r.ok) {
              const j = await r.json().catch(()=>({}));
              uploadStatus.textContent = 'Upload failed: ' + (j.detail || r.statusText);
              uploadStatus.className = 'status-text text-error';
              document.body.removeChild(tmp);
              return;
            }
            const j = await r.json();
            currentDocId = j.doc_id;
            uploadStatus.textContent = `Uploaded (doc_id=${currentDocId})`;
            uploadStatus.className = 'status-text text-success';
          } catch (e) {
            uploadStatus.textContent = 'Upload error: ' + String(e);
            uploadStatus.className = 'status-text text-error';
          }
          document.body.removeChild(tmp);
        });
        tmp.click();
      }, 350); // 350ms to allow UI to update and user to see the switch
      return;
    }
    // Already in OCR Q&A mode: open file picker immediately
    const tmp = document.createElement('input');
    tmp.type = 'file';
    tmp.accept = 'image/*';
    tmp.style.position = 'absolute';
    tmp.style.left = '-9999px';
    document.body.appendChild(tmp);

    tmp.addEventListener('change', async () => {
      const files = tmp.files;
      if (!files || files.length === 0) {
        uploadStatus.textContent = 'No file selected';
        uploadStatus.className = 'status-text';
        document.body.removeChild(tmp);
        return;
      }
      switchToOcrQaMode();
      uploadStatus.textContent = 'Uploading...';
      uploadStatus.className = 'status-text text-warning';
      const fd = new FormData();
      fd.append('file', files[0]);
      try {
        const r = await fetch('/upload-image', { method: 'POST', body: fd });
        if (!r.ok) {
          const j = await r.json().catch(()=>({}));
          uploadStatus.textContent = 'Upload failed: ' + (j.detail || r.statusText);
          uploadStatus.className = 'status-text text-error';
          document.body.removeChild(tmp);
          return;
        }
        const j = await r.json();
        currentDocId = j.doc_id;
        uploadStatus.textContent = `Uploaded (doc_id=${currentDocId})`;
        uploadStatus.className = 'status-text text-success';
      } catch (e) {
        uploadStatus.textContent = 'Upload error: ' + String(e);
        uploadStatus.className = 'status-text text-error';
      }
      document.body.removeChild(tmp);
    });
    tmp.click();
  });
}

function setRunning(running){
  if(running){
    askBtnEl.disabled = true;
    recalcBtn.disabled = true;
    questionEl.disabled = true;
    spinner.classList.remove('hidden');
    spinnerText.classList.remove('hidden');
    spinnerText.textContent = 'Processing...';
    // hide the small status text while spinner is active
    statusEl.textContent = '';
    console.log('[DEBUG] Loading state: ON, spinner should be visible'); // Debug log
  } else {
    askBtnEl.disabled = false;
    recalcBtn.disabled = false;
    questionEl.disabled = false;
    spinner.classList.add('hidden');
    spinnerText.classList.add('hidden');
    // restore default status
    statusEl.textContent = 'Ready';
    console.log('[DEBUG] Loading state: OFF, spinner should be hidden'); // Debug log
  }
}

function renderPresenterTrace(text){
  if(!text) return '—';
  // Split traces into chunks separated by blank lines and render as collapsible
  const parts = String(text).split(/\n\s*\n/).filter(Boolean);
  const container = document.createElement('div');
  parts.forEach((p, i) => {
    const d = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = p.split('\n')[0].slice(0,80);
    summary.style.cursor = 'pointer';
    summary.style.padding = '0.5rem';
    summary.style.borderRadius = '0.375rem';
    summary.style.marginBottom = '0.5rem';
    summary.style.backgroundColor = 'var(--bg-tertiary)';
    const pre = document.createElement('pre');
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.fontSize = '0.75rem';
    pre.style.padding = '0.75rem';
    pre.style.backgroundColor = 'var(--bg-secondary)';
    pre.style.borderRadius = '0.375rem';
    pre.style.margin = '0';
    pre.textContent = p;
    d.appendChild(summary);
    d.appendChild(pre);
    if(i===0) d.open = true;
    container.appendChild(d);
  });
  return container;
}

// small helper to escape HTML
const esc = s => String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// Detect if text contains a pipe/markdown table and return true
function isMarkdownTable(text){
  if(!text) return false;
  const lines = String(text).split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  if(lines.length < 2) return false;
  // Count lines that contain pipe separators
  const pipeLines = lines.filter(l => l.includes('|'));
  if(pipeLines.length < 2) return false;
  // If the second line is a markdown separator like | --- | --- | or ---|---
  const second = lines[1];
  if(/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(second)) return true;
  // Otherwise, if majority of non-empty lines contain pipes, treat as table
  return (pipeLines.length / lines.length) >= 0.5;
}

// Parse a simple markdown/pipe table into a DOM <table>
function parseMarkdownTable(text){
  const lines = String(text).split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  // find the first contiguous block of pipe lines
  let start = -1, end = -1;
  for(let i=0;i<lines.length;i++){
    if(lines[i].includes('|')){ start = i; break; }
  }
  if(start === -1) return null;
  for(let i=start;i<lines.length;i++){
    if(!lines[i].includes('|')){ end = i; break; }
  }
  if(end === -1) end = lines.length;
  const block = lines.slice(start, end);
  const rows = block.map(r => {
    let s = r;
    if(s.startsWith('|')) s = s.slice(1);
    if(s.endsWith('|')) s = s.slice(0,-1);
    return s.split('|').map(c => c.trim());
  });

  // If second row is separator, treat first row as header
  let header = null;
  if(rows.length >= 2 && rows[1].every(c => /^:?-{3,}:?$/.test(c))) {
    header = rows[0];
    rows.splice(0,2);
  }

  const table = document.createElement('table');
  table.className = 'answer-table';
  const thead = document.createElement('thead');
  const tbody = document.createElement('tbody');

  if(header){
    const tr = document.createElement('tr');
    header.forEach(h => { const th = document.createElement('th'); th.textContent = h || ''; tr.appendChild(th); });
    thead.appendChild(tr);
  } else if(rows.length>0){
    // use first row as header if no separator present
    const tr = document.createElement('tr');
    rows[0].forEach((_,i) => { const th = document.createElement('th'); th.textContent = `col ${i+1}`; tr.appendChild(th); });
    thead.appendChild(tr);
  }

  rows.forEach(r => {
    const tr = document.createElement('tr');
    r.forEach(c => { const td = document.createElement('td'); td.textContent = c; tr.appendChild(td); });
    tbody.appendChild(tr);
  });

  if(thead.childElementCount) table.appendChild(thead);
  table.appendChild(tbody);
  
  // Create enhanced table wrapper with controls
  const wrapper = document.createElement('div');
  wrapper.className = 'table-wrapper';
  
  // Create table controls
  const controls = document.createElement('div');
  controls.className = 'table-controls';
  
  const controlsLeft = document.createElement('div');
  controlsLeft.className = 'table-controls-left';
  
  const tableInfo = document.createElement('div');
  tableInfo.className = 'table-info';
  const rowCount = tbody.children.length;
  const colCount = header ? header.length : (rows.length > 0 ? rows[0].length : 0);
  tableInfo.textContent = `${rowCount} rows × ${colCount} columns`;
  
  controlsLeft.appendChild(tableInfo);
  
  const controlsRight = document.createElement('div');
  controlsRight.className = 'table-controls-right';
  
  const resizeControls = document.createElement('div');
  resizeControls.className = 'table-resize-controls';
  
  // Size control buttons
  const sizes = [
    { name: 'Compact', class: 'compact' },
    { name: 'Comfortable', class: 'comfortable' },
    { name: 'Spacious', class: 'spacious' }
  ];
  
  sizes.forEach((size, index) => {
    const btn = document.createElement('button');
    btn.textContent = size.name;
    btn.className = 'table-resize-btn';
    if (index === 1) btn.classList.add('active'); // Default to comfortable
    
    btn.addEventListener('click', () => {
      // Remove all size classes
      table.classList.remove('compact', 'comfortable', 'spacious');
      // Add selected size class
      table.classList.add(size.class);
      
      // Update active button
      resizeControls.querySelectorAll('.table-resize-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
    
    resizeControls.appendChild(btn);
  });
  
  controlsRight.appendChild(resizeControls);
  
  controls.appendChild(controlsLeft);
  controls.appendChild(controlsRight);
  
  // Create table container
  const container = document.createElement('div');
  container.className = 'table-container';
  container.appendChild(table);
  
  wrapper.appendChild(controls);
  wrapper.appendChild(container);
  
  // Set default size
  table.classList.add('comfortable');
  
  return wrapper;
}

// heuristics: split logs into agent sections
function splitLogs(logs) {
  const sections = {memory: '', querycraft: '', presenter: ''};
  if (!logs) {
    console.log('[DEBUG] splitLogs: No logs provided');
    return sections;
  }
  
  console.log('[DEBUG] splitLogs: Input logs length:', logs.length);
  console.log('[DEBUG] splitLogs: First 500 chars:', logs.substring(0, 500));
  console.log('[DEBUG] splitLogs: Last 500 chars:', logs.substring(Math.max(0, logs.length - 500)));
  
  // Normalize line endings
  const lines = logs.replace(/\r/g,'').split('\n');
  console.log('[DEBUG] splitLogs: Total lines:', lines.length);

  // Look for explicit markers first
  let mode = null;
  let memoryMarkerFound = false;
  let querycraftMarkerFound = false;
  let presenterMarkerFound = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const l = line.trim();
    
    // Match the exact agent section headers from agentic_workflow.py
    if (/=== MEMORY AGENT ===/i.test(l) || /INITIAL MEMORY/i.test(l)) { 
      mode = 'memory'; 
      memoryMarkerFound = true;
      console.log(`[DEBUG] splitLogs: Found MEMORY marker at line ${i}: "${l}"`);
      continue; 
    }
    if (/=== QUERY CRAFT AGENT ===/i.test(l) || /QUERY_CRAFT/i.test(l) || /MICRO-HINT/i.test(l)) { 
      mode = 'querycraft'; 
      querycraftMarkerFound = true;
      console.log(`[DEBUG] splitLogs: Found QUERYCRAFT marker at line ${i}: "${l}"`);
      continue; 
    }
    if (/=== RESULT PRESENTER AGENT ===/i.test(l) || /PRESENTER/i.test(l)) { 
      mode = 'presenter'; 
      presenterMarkerFound = true;
      console.log(`[DEBUG] splitLogs: Found PRESENTER marker at line ${i}: "${l}"`);
      continue; 
    }
    // Additional patterns to catch query craft related logs
    if (/\[FAST\]/i.test(l) || /\[GEN\]/i.test(l) || /\[MICRO-HINT\]/i.test(l)) { 
      mode = 'querycraft'; 
      console.log(`[DEBUG] splitLogs: Found QUERYCRAFT pattern at line ${i}: "${l}"`);
    }
    // Heuristic: SQL lines go to querycraft
    if (/^(SELECT|WITH|INSERT|UPDATE|DELETE)\b/i.test(l)) { 
      mode = 'querycraft'; 
      sections.querycraft += l + '\n'; 
      console.log(`[DEBUG] splitLogs: Found SQL pattern at line ${i}, mode set to querycraft`);
      continue; 
    }
    if (!mode) {
      // Seed with memory until a clearer marker appears
      sections.memory += line + '\n';
    } else {
      sections[mode] += line + '\n';
    }
  }

  console.log('[DEBUG] splitLogs: Markers found - Memory:', memoryMarkerFound, 'QueryCraft:', querycraftMarkerFound, 'Presenter:', presenterMarkerFound);
  console.log('[DEBUG] splitLogs: Section lengths - Memory:', sections.memory.length, 'QueryCraft:', sections.querycraft.length, 'Presenter:', sections.presenter.length);
  console.log('[DEBUG] splitLogs: Memory section preview:', sections.memory.substring(0, 200));
  console.log('[DEBUG] splitLogs: QueryCraft section preview:', sections.querycraft.substring(0, 200));
  console.log('[DEBUG] splitLogs: Presenter section preview:', sections.presenter.substring(0, 200));

  return sections;
}

function findSQL(logs) {
  if (!logs) return '';
  // Primary: explicit markers
  const markerRe = /<<<SQL_REVISED_START>>>([\s\S]*?)<<<SQL_REVISED_END>>>/m;
  let m = logs.match(markerRe);
  if (m) return m[1].trim();
  const markerRe2 = /<<<SQL_START>>>([\s\S]*?)<<<SQL_END>>>/m;
  m = logs.match(markerRe2);
  if (m) return m[1].trim();
  // Fallback: first standalone SQL-ish block up to semicolon
  const fallback = logs.match(/^(?:.*?)(SELECT|WITH)[^;]{0,4000};/im);
  if (fallback) {
    const startIdx = fallback.index;
    if (startIdx != null) {
      const slice = logs.slice(startIdx).match(/([\s\S]*?;)/);
      if (slice) return slice[1].trim();
    }
  }
  return '';
}

async function runQuery(q){
  if (!q) return alert('Please enter a question');
  answerEl.textContent = 'Thinking...';
  setRunning(true);
  timeEl.textContent = '';
  sqlEl.textContent = '-- SQL will appear here --';
  memoryEl.textContent = querycraftEl.textContent = presenterEl.textContent = '—';
  
  const startTime = Date.now();
  
  try {
    const body = { question: q, mode: currentMode };
    if (currentMode === 'ocr_qa' && currentDocId) body.doc_id = currentDocId;
    const res = await fetch('/api/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    let j = null;
    if (!res.ok) {
      try { j = await res.json(); } catch(err){ const t = await res.text(); answerEl.textContent = 'Error: ' + t; setRunning(false); return; }
      if (j.error === 'worker_timeout') { answerEl.textContent = 'Worker timeout: ' + (j.message || 'The worker did not respond in time.'); setRunning(false); return; }
      answerEl.textContent = j.error || j.message || JSON.stringify(j);
      if (j.trace) presenterEl.textContent = j.trace;
      setRunning(false);
      return;
    }
    j = await res.json();
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    
    if (j.answer) {
      // Render markdown/pipe tables as HTML tables for readability
      if(isMarkdownTable(j.answer)){
        const node = parseMarkdownTable(j.answer);
        answerEl.innerHTML = '';
        if(node) answerEl.appendChild(node);
        else answerEl.textContent = j.answer;
      } else {
        // Non-table answers: preserve plain text
        answerEl.textContent = j.answer;
      }
    } else answerEl.textContent = JSON.stringify(j, null, 2);
    
    // Prefer direct sql field from worker, else derive from logs
    if (j.sql) {
      const codeEl = document.getElementById('sql');
      codeEl.textContent = j.sql;
      if(window.Prism && Prism.highlightElement) Prism.highlightElement(codeEl);
    }
    
    if (j.logs) {
      console.log('[DEBUG] runQuery: Processing logs of length:', j.logs.length);
      const sections = splitLogs(j.logs);
      console.log('[DEBUG] runQuery: Sections received:', {
        memory: sections.memory.length,
        querycraft: sections.querycraft.length,
        presenter: sections.presenter.length
      });
      
      const memoryContent = sections.memory.trim() || '—';
      const querycraftContent = sections.querycraft.trim() || '—';
      const presenterContent = sections.presenter.trim() || '';
      
      console.log('[DEBUG] runQuery: Setting memory content length:', memoryContent.length);
      console.log('[DEBUG] runQuery: Setting querycraft content length:', querycraftContent.length);
      console.log('[DEBUG] runQuery: Setting presenter content length:', presenterContent.length);
      
      memoryEl.textContent = memoryContent;
      querycraftEl.textContent = querycraftContent;
      
      if(presenterContent){ 
        console.log('[DEBUG] runQuery: Rendering presenter trace');
        const node = renderPresenterTrace(presenterContent); 
        presenterEl.innerHTML = ''; 
        presenterEl.appendChild(node); 
      } else { 
        console.log('[DEBUG] runQuery: No presenter content, setting dash');
        presenterEl.textContent = '—'; 
      }
      if (!j.sql) {
        const sql = findSQL(j.logs) || '';
        if (sql) { 
          const codeEl = document.getElementById('sql'); 
          codeEl.textContent = sql; 
          if(window.Prism && Prism.highlightElement) Prism.highlightElement(codeEl); 
        }
      }
    } else {
      console.log('[DEBUG] runQuery: No logs in response');
    }
    
    statusEl.textContent = 'Done';
    timeEl.textContent = `${elapsed}s`;
    setRunning(false);
  } catch (e) {
    answerEl.textContent = 'Error: ' + String(e);
    setRunning(false);
  }
}

askBtn.onclick = () => runQuery(questionEl.value.trim());

recalcBtn.onclick = () => {
  const q = questionEl.value.trim();
  if (!q) return alert('Please enter a question');
  const normalized = q.replace(/^(?:re-execute\s+)+/i, '');
  const prefixed = 're-execute ' + normalized;
  // do not modify the visible textarea; send the prefixed query directly
  runQuery(prefixed);
};

// Enter key support
questionEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    runQuery(questionEl.value.trim());
  }
});

