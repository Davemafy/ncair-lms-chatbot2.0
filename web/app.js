const form = document.querySelector('#chatForm');
const prompt = document.querySelector('#prompt');
const messages = document.querySelector('#messages');
const welcome = document.querySelector('#welcome');
const conversation = document.querySelector('#conversation');
const send = document.querySelector('#sendBtn');

function scrollDown() { requestAnimationFrame(() => conversation.scrollTo({ top: conversation.scrollHeight, behavior: 'smooth' })); }
function escapeHtml(value) { return value.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]); }
function renderMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1 ↗</a>')
    .replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
}
function message(role, html) {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  row.innerHTML = role === 'assistant' ? `<div class="msg-icon">AI</div><div class="bubble">${html}</div>` : `<div class="bubble">${html}</div>`;
  messages.appendChild(row); scrollDown(); return row;
}
async function ask(text) {
  if (!text.trim()) return;
  welcome.hidden = true; message('user', escapeHtml(text)); prompt.value = ''; prompt.style.height = 'auto'; send.disabled = true;
  const typing = message('assistant', '<div class="typing"><span></span><span></span><span></span></div>');
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'The assistant could not respond.');
    typing.querySelector('.bubble').innerHTML = `<p>${renderMarkdown(result.answer)}</p><span class="source-chip">${escapeHtml(result.tool)}</span>`;
  } catch (error) {
    typing.querySelector('.bubble').innerHTML = `<p>${escapeHtml(error.message)}</p><span class="source-chip">Service unavailable</span>`;
  } finally { send.disabled = false; prompt.focus(); scrollDown(); }
}
form.addEventListener('submit', event => { event.preventDefault(); ask(prompt.value); });
prompt.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
prompt.addEventListener('input', () => { prompt.style.height = 'auto'; prompt.style.height = `${Math.min(prompt.scrollHeight, 120)}px`; });
document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => ask(button.dataset.prompt)));
document.querySelector('#newChat').addEventListener('click', () => { messages.innerHTML = ''; welcome.hidden = false; prompt.value = ''; closeMenu(); });
const sidebar = document.querySelector('#sidebar'); const scrim = document.querySelector('#scrim');
function closeMenu() { sidebar.classList.remove('open'); scrim.classList.remove('show'); }
document.querySelector('#menuBtn').addEventListener('click', () => { sidebar.classList.add('open'); scrim.classList.add('show'); });
scrim.addEventListener('click', closeMenu);
