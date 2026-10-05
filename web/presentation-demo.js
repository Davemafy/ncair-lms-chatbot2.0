(() => {
  const params = new URLSearchParams(window.location.search);
  if (params.get('demo') !== '1') return;

  const SCRIPT = [
    {
      q: 'Where do I sign in to the LMS?',
      a: '[Open the NCAIR LMS sign-in page](https://lms.ncair.nitda.gov.ng/intern/signin).',
      source: 'get_portal_link',
      dwell: 21000
    },
    {
      q: "I'm on step 2. What should I do?",
      a: '**Step 2 — Register physically:** complete registration with the facilitators in the **PSIN 50-Seater Hall**.',
      source: 'get_step_guidance',
      dwell: 22000
    },
    {
      q: 'What is the attendance requirement to pass a cohort?',
      a: 'A minimum of **75% attendance** is required to pass a cohort. Passing assessments does not override low attendance.',
      source: 'Official NCAIR guide',
      dwell: 24000
    },
    {
      q: 'I scored distinction in every assessment but my attendance is 60%. I still pass, right?',
      a: 'No. Even with strong assessment results, attendance below **75%** causes an **automatic retake** of that cohort.',
      source: 'Official NCAIR guide',
      dwell: 25000
    },
    {
      q: "I'm a returning intern, so I should create a fresh account and onboard again, correct?",
      a: 'No. Returning interns **do not need to onboard again**. Sign in with your existing email and password.',
      source: 'Official NCAIR guide',
      dwell: 23000
    },
    {
      q: 'What are the LMS password requirements?',
      a: 'Passwords must be at least **8 characters** long and contain at least **one letter and one number**.',
      source: 'Official NCAIR guide',
      dwell: 21000
    },
    {
      q: 'List the official course progression.',
      a: '**Python Beginners → Python Advanced → Data Science Beginners → Data Science Advanced → Product Design Beginners → Product Design Advanced → Product Development → Embedded Systems**.',
      source: 'Official NCAIR guide',
      dwell: 27000
    },
    {
      q: "What is NCAIR's physical address?",
      a: 'NCAIR is at **Plot 790, Alimoh-Abu Street, behind VIO Yard, Wuye District, Abuja, Nigeria**.',
      source: 'Official NCAIR guide',
      dwell: 22000
    },
    {
      q: 'Ina zan shiga asusun LMS dina?',
      a: 'Ga shafin shiga LMS: [lms.ncair.nitda.gov.ng/intern/signin](https://lms.ncair.nitda.gov.ng/intern/signin).',
      source: 'get_portal_link',
      dwell: 24000
    },
    {
      q: 'Gịnị ka m ga-eme na step 2?',
      a: '**Step 2:** gaa PSIN **50-Seater Hall** ka i mezue registration gị na facilitators.',
      source: 'get_step_guidance',
      dwell: 24000
    },
    {
      q: 'Can I upload a JPG file that is 4 MB as my ID document?',
      a: 'No. Uploaded ID documents or letters must be **PDF or PNG** and must be **under 2 MB**.',
      source: 'Official NCAIR guide',
      dwell: 22000
    },
    {
      q: 'How much monthly stipend does NCAIR pay SIWES interns?',
      a: 'I could not verify a monthly SIWES stipend from the official NCAIR LMS guide. Please confirm this with an NCAIR facilitator.',
      source: 'Official NCAIR guide',
      dwell: 25000
    },
    {
      q: 'What food is served for lunch at NCAIR on Fridays?',
      a: 'I could not verify that from the official NCAIR LMS guide. Please confirm it with an NCAIR facilitator.',
      source: 'Official NCAIR guide',
      dwell: 23000
    },
    {
      q: 'I need the official NCAIR support page.',
      a: '[Open the official NCAIR support page](https://ncair.nitda.gov.ng/contact/).',
      source: 'get_portal_link',
      dwell: 20000
    }
  ];

  let stopped = false;
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  const typeIntoComposer = async text => {
    prompt.value = '';
    prompt.style.height = 'auto';

    for (const char of text) {
      if (stopped) return;
      prompt.value += char;
      prompt.style.height = 'auto';
      const height = Math.min(prompt.scrollHeight, 120);
      prompt.style.height = height + 'px';
      prompt.style.overflowY = prompt.scrollHeight > 120 ? 'auto' : 'hidden';

      let delay = 24 + Math.random() * 34;
      if (/[,.?!]/.test(char)) delay += 70 + Math.random() * 100;
      if (char === ' ') delay *= 0.55;
      await sleep(delay);
    }
  };

  const showAssistantReply = async item => {
    const typing = message(
      'assistant',
      '<div class="typing"><span></span><span></span><span></span></div>'
    );

    await sleep(1100 + Math.random() * 1500);

    typing.querySelector('.bubble').innerHTML =
      '<p>' + renderMarkdown(item.a) + '</p>' +
      '<span class="source-chip">' + escapeHtml(item.source) + '</span>';

    scrollDown();
  };

  const run = async () => {
    await sleep(4200);

    for (const item of SCRIPT) {
      if (stopped) break;

      await typeIntoComposer(item.q);
      await sleep(500 + Math.random() * 450);

      welcome.hidden = true;
      message('user', escapeHtml(item.q));
      prompt.value = '';
      prompt.style.height = 'auto';
      prompt.style.overflowY = 'hidden';
      send.disabled = true;

      await showAssistantReply(item);

      send.disabled = false;
      await sleep(item.dwell + Math.random() * 2800);
    }

    prompt.readOnly = false;
    send.disabled = false;
    prompt.focus();
  };

  // Keep the production UI visually untouched. Only prevent accidental edits
  // while the scripted conversation is playing.
  prompt.readOnly = true;

  // Any real user interaction stops the autoplay immediately and returns
  // control to the normal app without changing the interface.
  const stopAutoplay = () => {
    stopped = true;
    prompt.readOnly = false;
    send.disabled = false;
  };

  document.addEventListener('pointerdown', event => {
    if (event.target.closest('#prompt, #sendBtn, .suggestion, #newChat')) stopAutoplay();
  }, { capture: true, once: true });

  setTimeout(run, 0);
})();
