const form = document.querySelector('#chatForm');
const prompt = document.querySelector('#prompt');
const messages = document.querySelector('#messages');
const welcome = document.querySelector('#welcome');
const conversation = document.querySelector('#conversation');
const send = document.querySelector('#sendBtn');
const apiBaseUrl = String(window.NCAIR_API_URL || '').replace(/\/+$/, '');

const previewKnowledge = [
  { test: /attendance|75%|miss class/i, answer: 'You need at least **75% attendance** to pass each cohort. Passing the assessments does not override the attendance requirement; attendance below 75% means retaking that cohort.' },
  { test: /course|curriculum|learn|pathway|cohort/i, answer: 'The course pathway is **Python Beginners → Python Advanced → Data Science Beginners → Data Science Advanced → Product Design Beginners → Product Design Advanced → Product Development → Embedded Systems**.' },
  { test: /logbook|siwes/i, answer: 'SIWES interns must physically present their logbooks for signing at NCAIR Headquarters **every two weeks**.' },
  { test: /location|address|where.*ncair/i, answer: 'NCAIR is at **Plot 790, Alimoh-Abu Street, behind VIO Yard, Wuye District, Abuja**. Physical onboarding takes place at the 50-Seater Hall in the e-Government Building.' },
  { test: /password|email|invitation|spam|locked|upload|document/i, answer: 'Check Spam, Junk or Trash for a missing invitation. Passwords need at least 8 characters with one letter and number. Uploads must be PDF or PNG and under 2 MB.' },
  { test: /sign.?in|login|lms|portal|open/i, answer: '[Open the NCAIR LMS sign-in page](https://lms.ncair.nitda.gov.ng/intern/signin).' },
  { test: /start|new|onboard|register|step/i, answer: 'New interns complete registration physically with facilitators in the PSIN 50-Seater Hall after orientation. Registration closes at **5:00 PM** on the registration day.' },
];

function previewAnswer(text) {
  const match = previewKnowledge.find(item => item.test.test(text));
  return match?.answer || 'I could not find enough information in the current official guide. Please confirm this with your NCAIR facilitator.';
}

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
  welcome.hidden = true; message('user', escapeHtml(text)); prompt.value = ''; prompt.style.height = 'auto'; prompt.style.overflowY = 'hidden'; send.disabled = true;
  const typing = message('assistant', '<div class="typing"><span></span><span></span><span></span></div>');
  try {
    const response = await fetch(`${apiBaseUrl}/api/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text }) });
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) throw new Error('BACKEND_UNAVAILABLE');
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'The assistant could not respond.');
    typing.querySelector('.bubble').innerHTML = `<p>${renderMarkdown(result.answer)}</p><span class="source-chip">${escapeHtml(result.tool)}</span>`;
  } catch (error) {
    const answer = previewAnswer(text);
    typing.querySelector('.bubble').innerHTML = `<p>${renderMarkdown(answer)}</p><span class="source-chip">Official guide · preview mode</span>`;
  } finally { send.disabled = false; prompt.focus(); scrollDown(); }
}
form.addEventListener('submit', event => { event.preventDefault(); ask(prompt.value); });
prompt.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
prompt.addEventListener('input', () => {
  prompt.style.height = 'auto';
  const height = Math.min(prompt.scrollHeight, 120);
  prompt.style.height = `${height}px`;
  prompt.style.overflowY = prompt.scrollHeight > 120 ? 'auto' : 'hidden';
});
document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => ask(button.dataset.prompt)));
document.querySelector('#newChat').addEventListener('click', () => { messages.innerHTML = ''; welcome.hidden = false; prompt.value = ''; closeMenu(); });
const sidebar = document.querySelector('#sidebar'); const scrim = document.querySelector('#scrim');
function closeMenu() { sidebar.classList.remove('open'); scrim.classList.remove('show'); }
document.querySelector('#menuBtn').addEventListener('click', () => { sidebar.classList.add('open'); scrim.classList.add('show'); });
scrim.addEventListener('click', closeMenu);
