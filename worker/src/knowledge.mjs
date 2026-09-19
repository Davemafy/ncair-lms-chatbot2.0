export const PORTAL_URLS = {
  main: "https://lms.ncair.nitda.gov.ng",
  login: "https://lms.ncair.nitda.gov.ng/intern/signin",
  profile: "https://lms.ncair.nitda.gov.ng/intern/profile",
  courses: "https://lms.ncair.nitda.gov.ng/intern/courses",
  support: "https://ncair.nitda.gov.ng/contact/",
};

const KNOWLEDGE = [
  {
    source: "Official guide · NCAIR location",
    keywords: ["ncair", "location", "address", "where", "wuye", "orientation", "onboarding", "facility"],
    text: "NCAIR is a special-purpose vehicle of NITDA under the Federal Ministry of Communications, Innovation and Digital Economy. Its physical address is Plot 790, Alimoh-Abu Street, behind VIO Yard, Wuye District, Abuja. New-intern physical registration takes place after orientation at the 50-Seater Hall in the e-Government Building (PSIN).",
  },
  {
    source: "Official guide · Course progression",
    keywords: ["course", "courses", "curriculum", "pathway", "progression", "cohort", "python", "data", "design", "development", "embedded"],
    text: "The official progression is Python Beginners, Python Advanced, Data Science Beginners, Data Science Advanced, Product Design Beginners, Product Design Advanced, Product Development, then Embedded Systems.",
  },
  {
    source: "Official guide · Intern rules",
    keywords: ["siwes", "nysc", "logbook", "cds", "intern", "signing", "weeks", "simultaneously"],
    text: "First-cohort SIWES interns register for Python Beginners and Product Design Beginners simultaneously. NYSC interns select their courses and CDS day on the registration form; the selected CDS day grants an official class excuse. SIWES interns must physically present their logbooks at NCAIR Headquarters for signing every two weeks.",
  },
  {
    source: "Official guide · Attendance and grading",
    keywords: ["attendance", "class", "classes", "75", "pass", "passing", "retake", "assessment", "grading", "schedule", "unlock"],
    text: "Classes run three times weekly for about two to three hours per session. At least 75% attendance is required to pass a cohort. Passing assessments or the final project does not override low attendance: attendance below 75% causes an automatic retake. Passing both attendance and assessments unlocks the next course under My Learning → Courses.",
  },
  {
    source: "Official guide · Registration troubleshooting",
    keywords: ["registration", "register", "locked", "invitation", "email", "spam", "junk", "password", "upload", "document", "pdf", "png", "2mb"],
    text: "Physical registration closes at 5:00 PM on registration day, and controls show Locked outside the window. For a missing invitation, check Spam, Junk or Trash and move the email to Inbox before selecting Complete Registration. Passwords require at least eight characters with at least one letter and one number. Uploaded IDs or letters must be PDF or PNG and under 2 MB.",
  },
  {
    source: "Official guide · Returning interns",
    keywords: ["returning", "existing", "account", "signin", "sign", "login", "failed", "reregister", "retake"],
    text: "Returning interns do not onboard again. They sign in with their existing email and password at https://lms.ncair.nitda.gov.ng/intern/signin. To retake a failed course, open Courses and select Re-register.",
  },
];

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "can", "do", "for", "how", "i", "in", "is", "it",
  "me", "my", "of", "on", "or", "the", "to", "what", "when", "where", "with",
]);

function tokens(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
}

export function searchKnowledgeBase(query, limit = 3) {
  const queryTokens = new Set(tokens(query));
  const ranked = KNOWLEDGE.map((entry) => {
    const keywordMatches = entry.keywords.filter((keyword) => queryTokens.has(keyword)).length;
    const textTokens = new Set(tokens(entry.text));
    const textMatches = [...queryTokens].filter((token) => textTokens.has(token)).length;
    return { ...entry, score: keywordMatches * 3 + textMatches };
  })
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  if (!ranked.length) {
    return {
      evidence: "The official guide does not contain enough information to answer this question.",
      sources: ["Official NCAIR LMS onboarding guide"],
    };
  }

  return {
    evidence: ranked.map((entry) => `[${entry.source}]\n${entry.text}`).join("\n\n"),
    sources: ranked.map((entry) => entry.source),
  };
}

export function getPortalLink(action) {
  const key = Object.hasOwn(PORTAL_URLS, action) ? action : "main";
  const labels = {
    main: "NCAIR LMS",
    login: "NCAIR LMS sign-in page",
    profile: "intern profile page",
    courses: "courses page",
    support: "NCAIR support page",
  };
  return {
    answer: `[Open the ${labels[key]}](${PORTAL_URLS[key]}).`,
    evidence: "Verified NCAIR portal registry",
  };
}

export function getStepGuidance(step) {
  const steps = {
    1: "**Step 1 — Attend orientation:** New interns begin with the physical orientation at NCAIR.",
    2: "**Step 2 — Register physically:** Complete registration with the facilitators in the PSIN 50-Seater Hall.",
    3: "**Step 3 — Complete your LMS account:** Use the invitation email, create a valid password and upload the requested document in PDF or PNG under 2 MB.",
    4: "**Step 4 — Begin your assigned courses:** Confirm your cohort and course selection in the LMS. Registration closes at 5:00 PM on registration day.",
  };
  const numericStep = Number(step);
  if (steps[numericStep]) {
    return { answer: steps[numericStep], evidence: "Official NCAIR onboarding sequence" };
  }
  return {
    answer: Object.values(steps).join("\n\n"),
    evidence: "Official NCAIR onboarding sequence",
  };
}
