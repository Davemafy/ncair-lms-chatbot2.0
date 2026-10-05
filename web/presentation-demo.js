
(() => {
  const params = new URLSearchParams(window.location.search);
  if (params.get('demo') !== '1') return;

  document.body.classList.add('demo-mode');

  let paused = false;
  let cancelled = false;
  let running = false;
  let current = 0;

  const DEMO = [
    {
      title: 'LMS sign-in',
      q: 'Where do I sign in to the LMS?',
      stages: ['Understanding navigation request', 'Selecting a verified LMS destination', 'Checking the configured URL'],
      trace: 'language: english  ·  tool: get_portal_link  ·  action: login',
      a: '[Open the NCAIR LMS sign-in page](https://lms.ncair.nitda.gov.ng/intern/signin).',
      source: 'Verified portal link',
      dwell: 23000
    },
    {
      title: 'Onboarding step 2',
      q: "I'm on step 2. What should I do?",
      stages: ['Reading the requested onboarding step', 'Validating the step number', 'Returning stored guidance'],
      trace: 'language: english  ·  tool: get_step_guidance  ·  step: 2',
      a: '**Step 2 — Register physically:** complete registration with the facilitators in the **PSIN 50-Seater Hall**.',
      source: 'Official onboarding guidance',
      dwell: 24000
    },
    {
      title: 'Attendance policy',
      q: 'What is the attendance requirement to pass a cohort?',
      stages: ['Interpreting a policy question', 'Searching official NCAIR evidence', 'Verifying the supported answer'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  evidence: attendance policy',
      a: 'A minimum of **75% attendance** is required to pass a cohort. Passing assessments does not override low attendance.',
      source: 'Official guide · Attendance and grading',
      dwell: 27000
    },
    {
      title: 'Trap: distinction + 60%',
      q: 'I scored distinction in every assessment but my attendance is 60%. I still pass, right?',
      stages: ['Checking the claim instead of matching trigger words', 'Retrieving the attendance rule', 'Verifying the contradiction'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  class: trap',
      a: 'No. Even with strong assessment results, attendance below **75%** causes an **automatic retake** of that cohort.',
      source: 'Official guide · Attendance and grading',
      dwell: 28000
    },
    {
      title: 'Returning intern',
      q: "I'm a returning intern, so I should create a fresh account and onboard again, correct?",
      stages: ['Interpreting returning-intern intent', 'Searching official procedures', 'Checking whether re-onboarding is required'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  evidence: returning interns',
      a: 'No. Returning interns **do not need to onboard again**. They should sign in with their existing email and password.',
      source: 'Official guide · Returning interns',
      dwell: 26000
    },
    {
      title: 'Password requirements',
      q: 'What are the LMS password requirements?',
      stages: ['Classifying a troubleshooting question', 'Searching official onboarding guidance', 'Checking the exact requirement'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  evidence: password requirements',
      a: 'Passwords must be at least **8 characters** long and contain at least **one letter and one number**.',
      source: 'Official guide · Registration troubleshooting',
      dwell: 23000
    },
    {
      title: 'Course progression',
      q: 'List the official course progression.',
      stages: ['Searching the documented curriculum', 'Ordering the retrieved course sequence', 'Verifying against the official guide'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  evidence: course progression',
      a: '**Python Beginners → Python Advanced → Data Science Beginners → Data Science Advanced → Product Design Beginners → Product Design Advanced → Product Development → Embedded Systems**.',
      source: 'Official guide · Course progression',
      dwell: 29000
    },
    {
      title: 'NCAIR location',
      q: "What is NCAIR's physical address?",
      stages: ['Searching official facility information', 'Checking the address evidence', 'Preparing a grounded response'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  evidence: NCAIR location',
      a: 'NCAIR is at **Plot 790, Alimoh-Abu Street, behind VIO Yard, Wuye District, Abuja, Nigeria**.',
      source: 'Official guide · NCAIR location',
      dwell: 23000
    },
    {
      title: 'Hausa navigation',
      q: 'Ina zan shiga asusun LMS dina?',
      stages: ['Detecting Hausa', 'Understanding the requested LMS destination', 'Checking the verified sign-in URL'],
      trace: 'language: hausa  ·  tool: get_portal_link  ·  action: login',
      a: 'Ga shafin shiga LMS: [lms.ncair.nitda.gov.ng/intern/signin](https://lms.ncair.nitda.gov.ng/intern/signin).',
      source: 'Verified portal link · Hausa',
      dwell: 27000
    },
    {
      title: 'Igbo onboarding',
      q: 'Gịnị ka m ga-eme na step 2?',
      stages: ['Detecting Igbo', 'Reading the explicit onboarding step', 'Returning verified step guidance'],
      trace: 'language: igbo  ·  tool: get_step_guidance  ·  step: 2',
      a: '**Step 2:** gaa PSIN **50-Seater Hall** ka i mezue registration gị na facilitators.',
      source: 'Official onboarding guidance · Igbo',
      dwell: 27000
    },
    {
      title: 'Upload trap',
      q: 'Can I upload a JPG file that is 4 MB as my ID document?',
      stages: ['Checking the upload claim', 'Retrieving document constraints', 'Verifying format and size'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  class: trap',
      a: 'No. Uploaded ID documents or letters must be **PDF or PNG** and must be **under 2 MB**.',
      source: 'Official guide · Upload requirements',
      dwell: 25000
    },
    {
      title: 'Undocumented stipend',
      q: 'How much monthly stipend does NCAIR pay SIWES interns?',
      stages: ['Searching official NCAIR sources', 'Checking evidence coverage', 'Applying the unsupported-question rule'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  supported: false',
      a: 'I could not verify a monthly SIWES stipend from the official NCAIR LMS guide. Please confirm this with an NCAIR facilitator.',
      source: 'Safe refusal · No verified evidence',
      dwell: 29000
    },
    {
      title: 'Undocumented lunch',
      q: 'What food is served for lunch at NCAIR on Fridays?',
      stages: ['Searching the official guide', 'Finding no supporting evidence', 'Refusing to invent an answer'],
      trace: 'language: english  ·  tool: search_ncair_knowledge_base  ·  supported: false',
      a: 'I could not verify that from the official NCAIR LMS guide. Please confirm it with an NCAIR facilitator.',
      source: 'Safe refusal · No verified evidence',
      dwell: 26000
    },
    {
      title: 'Support page',
      q: 'I need the official NCAIR support page.',
      stages: ['Understanding a navigation request', 'Selecting the configured support destination', 'Validating the URL'],
      trace: 'language: english  ·  tool: get_portal_link  ·  action: support',
      a: '[Open the official NCAIR support page](https://ncair.nitda.gov.ng/contact/).',
      source: 'Verified portal link',
      dwell: 22000
    }
  ];

  const wait = async (ms) => {
    let left = ms;
    while (left > 0 && !cancelled) {
      if (paused) {
        await new Promise(resolve => setTimeout(resolve, 120));
        continue;
      }
      const slice = Math.min(120, left);
      await new Promise(resolve => setTimeout(resolve, slice));
      left -= slice;
    }
  };

  const historyLabel = document.querySelector('.history-item.active span:last-child');
  const headerTitle = document.querySelector('.header-title');
  const status = document.createElement('span');
  status.className = 'demo-status';
  status.textContent = 'Presentation replay';
  headerTitle.appendChild(status);

  const controls = document.createElement('div');
  controls.className = 'demo-controls';
  controls.innerHTML = '<span class="demo-progress">00 / ' + String(DEMO.length).padStart(2, '0') + '</span><button type="button" data-action="pause">Pause</button><button type="button" data-action="restart">Restart</button>';
  document.body.appendChild(controls);
  const progress = controls.querySelector('.demo-progress');
  const pauseButton = controls.querySelector('[data-action="pause"]');

  controls.addEventListener('click', event => {
    const action = event.target?.dataset?.action;
    if (action === 'pause') {
      paused = !paused;
      pauseButton.textContent = paused ? 'Resume' : 'Pause';
      status.textContent = paused ? 'Replay paused' : 'Presentation replay';
    }
    if (action === 'restart') {
      cancelled = true;
      setTimeout(() => {
        cancelled = false;
        paused = false;
        pauseButton.textContent = 'Pause';
        run();
      }, 180);
    }
  });

  const addStage = text => {
    const row = document.createElement('div');
    row.className = 'demo-stage';
    row.innerHTML = '<span class="demo-spinner"></span><span></span>';
    row.querySelector('span:last-child').textContent = text;
    messages.appendChild(row);
    scrollDown();
    return row;
  };

  const addTrace = text => {
    const row = document.createElement('div');
    row.className = 'demo-trace';
    const parts = text.split('  ·  ');
    row.innerHTML = parts.map((part, index) => index === 0 ? '<strong>' + escapeHtml(part) + '</strong>' : escapeHtml(part)).join('  ·  ');
    messages.appendChild(row);
    scrollDown();
  };

  const typePrompt = async text => {
    prompt.value = '';
    for (const char of text) {
      if (cancelled) return;
      prompt.value += char;
      prompt.style.height = 'auto';
      prompt.style.height = Math.min(prompt.scrollHeight, 120) + 'px';
      await wait(18 + Math.random() * 17);
    }
    await wait(450);
  };

  const typeAssistant = async (text, source) => {
    const row = message('assistant', '<p></p><span class="source-chip"></span>');
    const paragraph = row.querySelector('p');
    const chip = row.querySelector('.source-chip');
    chip.textContent = source;
    let shown = '';
    for (const char of text) {
      if (cancelled) return;
      shown += char;
      paragraph.innerHTML = renderMarkdown(shown);
      scrollDown();
      await wait(7 + Math.random() * 6);
    }
  };

  const run = async () => {
    if (running) cancelled = true;
    await new Promise(resolve => setTimeout(resolve, 160));
    running = true;
    cancelled = false;
    current = 0;

    messages.innerHTML = '';
    welcome.hidden = true;
    prompt.value = '';
    prompt.disabled = true;
    send.disabled = true;
    document.querySelectorAll('.suggestion').forEach(button => button.disabled = true);

    message('assistant', '<p>I\'ll run through a few real benchmark-style requests: navigation, onboarding, policies, multilingual queries, traps and safe refusals.</p><span class="source-chip">Presentation replay · scripted from project evidence</span>');
    await wait(2200);

    for (let i = 0; i < DEMO.length && !cancelled; i += 1) {
      current = i;
      const item = DEMO[i];
      progress.textContent = String(i + 1).padStart(2, '0') + ' / ' + String(DEMO.length).padStart(2, '0');
      status.textContent = item.title;
      if (historyLabel) historyLabel.textContent = item.title;

      await typePrompt(item.q);
      if (cancelled) break;
      message('user', escapeHtml(item.q));
      prompt.value = '';
      prompt.style.height = 'auto';

      let stage = null;
      for (const stageText of item.stages) {
        if (stage) stage.remove();
        stage = addStage(stageText);
        await wait(850 + Math.random() * 420);
      }
      if (stage) stage.remove();

      addTrace(item.trace);
      await wait(650);
      await typeAssistant(item.a, item.source);
      await wait(item.dwell);
    }

    if (!cancelled) {
      message('assistant', '<p>That completes the presentation replay. The normal assistant is available now for a fresh question.</p><span class="source-chip">Demo complete · live UI restored</span>');
      progress.textContent = 'DONE';
      status.textContent = 'Replay complete';
      prompt.disabled = false;
      send.disabled = false;
      document.querySelectorAll('.suggestion').forEach(button => button.disabled = false);
      prompt.placeholder = 'Ask a fresh question…';
      prompt.focus();
      scrollDown();
    }

    running = false;
  };

  setTimeout(run, 900);
})();
