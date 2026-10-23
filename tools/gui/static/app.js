'use strict';
const $ = id => document.getElementById(id);
const form = $('settings');
let token, saved, current, baseline, reasoning, busy = false, folderTarget, parentFolder;
let previewTimer;
try { document.documentElement.dataset.theme = localStorage.getItem('kyojin.theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch {}
$('theme').onclick = () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('kyojin.theme', theme); } catch {}
};
function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
async function api(path, body) {
  const response = await fetch('/api/' + path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Launcher-Token': token}, body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
function config() {
  const values = {};
  for (const field of form.elements) {
    if (!field.name) continue;
    values[field.name] = field.type === 'checkbox' ? field.checked :
      ['port', 'context', 'sessions', 'draft_tokens', 'cache_bits'].includes(field.name) ? Number(field.value) : field.value;
  }
  return values;
}
function dirty() { return JSON.stringify(config()) !== baseline; }
function mark() { $('edited').textContent = dirty() ? 'Unsaved changes' : 'Saved configuration'; }
function familyControls() {
  const family = form.elements.family.value;
  document.querySelectorAll('[data-family]').forEach(el => { el.hidden = !el.dataset.family.split(' ').includes(family); });
  const effort = form.elements.reasoning.value;
  form.elements.reasoning.replaceChildren(...reasoning[family].map(value => new Option(value || 'Model default', value)));
  form.elements.reasoning.value = reasoning[family].includes(effort) ? effort : '';
  form.elements.reasoning.disabled = family === 'qwen' && !form.elements.thinking.checked;
  form.elements.draft_policy.options[0].textContent = family === 'mimo' ? 'DFlash' : 'Mixed (MTP + n-gram)';
}
function fill(cfg) {
  for (const [key, value] of Object.entries(cfg)) {
    const field = form.elements.namedItem(key);
    if (!field || key === 'reasoning') continue;
    if (field.type === 'checkbox') field.checked = value;
    else field.value = value;
  }
  familyControls();
  form.elements.reasoning.value = cfg.reasoning;
  baseline = JSON.stringify(config()); mark(); preview();
}
function options() {
  $('presets').replaceChildren(...Object.keys(saved.presets).map(name => new Option(name, name)));
  $('presets').value = current;
}
async function refreshConfig() {
  const data = await api('config');
  saved = data; token = data.token; reasoning = data.reasoning; error(data.warning);
  return data;
}
async function action(work) {
  if (busy) return;
  busy = true; error();
  try { await work(); } catch (exc) { error(exc.message); }
  finally { busy = false; }
}
async function preview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try { $('command').textContent = (await api('preview', {config: config()})).command; }
    catch (exc) { $('command').textContent = exc.message; }
  }, 200);
}
form.addEventListener('submit', event => event.preventDefault());
form.addEventListener('input', () => { mark(); preview(); });
form.elements.family.addEventListener('change', () => {
  form.elements.reasoning.value = '';
  form.elements.draft_tokens.value = form.elements.family.value === 'glm' ? 2 : 7;
  form.elements.draft_policy.value = 'mix';
  familyControls(); mark(); preview();
});
form.elements.thinking.addEventListener('change', () => { familyControls(); mark(); preview(); });
$('load').onclick = () => action(async () => {
  if (dirty() && !confirm('Discard unsaved changes and load the selected preset?')) return;
  current = $('presets').value; fill(saved.presets[current]);
});
async function save(name) {
  await api('save', {name, config: config()});
  await refreshConfig(); current = name; options(); fill(saved.presets[name]);
}
$('save').onclick = () => action(() => save(current));
$('save-as').onclick = () => action(async () => {
  const name = prompt('Name for the new preset:');
  if (!name || !name.trim()) return;
  if (Object.hasOwn(saved.presets, name.trim())) throw new Error('That preset already exists. Use Save to update it.');
  await save(name.trim());
});
$('delete').onclick = () => action(async () => {
  if (!confirm(`Delete saved preset “${current}”? Unsaved changes will be discarded. Model files are kept.`)) return;
  await api('delete', {name: current}); const data = await refreshConfig();
  current = data.selected; options(); fill(saved.presets[current]);
});
$('start').onclick = () => action(async () => {
  if (!form.reportValidity()) return;
  await api('start', {config: config()}); await poll();
});
$('stop').onclick = () => action(async () => { await api('stop', {}); await poll(); });
$('exit').onclick = () => action(async () => {
  if (!confirm('Stop the engine and exit the launcher?')) return;
  await api('exit', {}); $('status').textContent = 'Launcher closed. You can close this tab.';
  clearTimeout(pollTimer); $('start').disabled = $('stop').disabled = true;
});
$('copy-log').onclick = () => action(() => navigator.clipboard.writeText($('logs').textContent));
let pollTimer;
async function poll() {
  clearTimeout(pollTimer);
  try {
    const data = await api('status');
    $('badge').textContent = data.state; $('status').textContent = data.message;
    $('live-dot').className = 'live-dot ' + data.state;
    const running = !['stopped', 'exited'].includes(data.state);
    $('start').disabled = running; $('stop').disabled = !running;
    $('api').textContent = data.api_url || '—';
    $('running').textContent = data.running && running ? data.running.model_id : '—';
    $('progress').hidden = !Number.isFinite(data.progress);
    if (Number.isFinite(data.progress)) $('progress').value = data.progress;
    const log = $('logs'), atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
    log.textContent = data.logs.join('\n') || 'No engine output yet.';
    if (atBottom) log.scrollTop = log.scrollHeight;
  } catch {
    $('badge').textContent = 'Disconnected'; $('status').textContent = 'Launcher unavailable. Check its terminal.';
    $('start').disabled = $('stop').disabled = true;
  }
  pollTimer = setTimeout(poll, 1500);
}
async function folders(path) {
  try {
    const data = await api('folders' + (path ? '?path=' + encodeURIComponent(path) : ''));
    $('folder-path').value = data.path; parentFolder = data.parent;
    $('browse-message').textContent = data.model ? 'Model config.json found.' : 'Open an EXL3 model pack folder.';
    $('browse-use').disabled = !data.model;
    $('folders').replaceChildren(...data.folders.map(folder => {
      const button = document.createElement('button'); button.textContent = '▸ ' + folder.name;
      button.onclick = () => folders(folder.path); return button;
    }));
  } catch (exc) { $('browse-message').textContent = exc.message; $('browse-use').disabled = true; }
}
document.querySelectorAll('[data-browse]').forEach(button => button.onclick = async () => {
  folderTarget = button.dataset.browse; $('browser').showModal();
  await folders(form.elements.namedItem(folderTarget).value);
});
$('browse-go').onclick = () => folders($('folder-path').value);
$('folder-path').onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); folders(event.target.value); } };
$('browse-up').onclick = () => folders(parentFolder);
$('browse-close').onclick = () => $('browser').close();
$('browse-use').onclick = () => {
  form.elements.namedItem(folderTarget).value = $('folder-path').value;
  $('browser').close(); mark(); preview();
};
window.addEventListener('beforeunload', event => { if (dirty()) { event.preventDefault(); event.returnValue = ''; } });
action(async () => {
  const data = await refreshConfig(); current = data.selected; options(); fill(data.defaults);
  fill({...data.defaults, ...data.presets[current]}); await poll();
});
