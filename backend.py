/* ══════════════════════════════════════════
   AGROAI — app.js
   ══════════════════════════════════════════ */
'use strict';

/* ─── TOAST ─── */
function showToast(msg, type = 'info') {
  let t = document.getElementById('_toast');
  if (!t) {
    t = document.createElement('div');
    t.id = '_toast';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.className   = 'toast toast-' + type;
  t.style.display = 'block';
  clearTimeout(t._tmr);
  t._tmr = setTimeout(() => { t.style.display = 'none'; }, 3200);
}

const API = 'http://localhost:8000';

/* ─── DISEASE DATA ─── */
const DISEASES = [
  { key:'Tomato_Bacterial_spot',     label:'Bacterial Spot',         severity:'High',
    symptoms:'Small dark water-soaked lesions on leaves and fruit surfaces.',
    treatment:'Apply copper-based bactericides. Remove infected debris. Avoid overhead irrigation.',
    prevention:'Use certified disease-free seeds. Practice 2-year crop rotation.' },
  { key:'Tomato_Early_blight',       label:'Early Blight',           severity:'Medium',
    symptoms:'Concentric dark rings forming a target pattern on older leaves.',
    treatment:'Apply chlorothalonil or mancozeb fungicide every 7 to 10 days.',
    prevention:'Rotate crops annually. Remove lower foliage. Mulch around base.' },
  { key:'Tomato_Late_blight',        label:'Late Blight',            severity:'Critical',
    symptoms:'Large irregular water-soaked grey-green lesions; white mould on underside.',
    treatment:'Apply metalaxyl or cymoxanil fungicide immediately. Destroy infected plants.',
    prevention:'Avoid overhead watering. Plant resistant varieties. Monitor humidity.' },
  { key:'Tomato_Leaf_Mold',          label:'Leaf Mold',              severity:'Medium',
    symptoms:'Yellow patches on upper leaf surface; olive-green mould on underside.',
    treatment:'Apply mancozeb or chlorothalonil. Improve greenhouse ventilation.',
    prevention:'Reduce relative humidity below 85%. Space plants adequately.' },
  { key:'Tomato_Septoria_leaf_spot', label:'Septoria Leaf Spot',     severity:'Medium',
    symptoms:'Small circular spots with dark borders and pale grey centres.',
    treatment:'Apply copper fungicide. Remove heavily infected leaves promptly.',
    prevention:'Mulch soil. Avoid wetting foliage during irrigation.' },
  { key:'Tomato_Spider_mites',       label:'Spider Mites',           severity:'Low',
    symptoms:'Fine yellow stippling on leaves; fine webbing on leaf undersides.',
    treatment:'Apply miticide or neem oil. Increase ambient humidity.',
    prevention:'Regular scouting. Introduce predatory mites as biocontrol.' },
  { key:'Tomato_Target_Spot',        label:'Target Spot',            severity:'Medium',
    symptoms:'Bulls-eye concentric ring lesions on leaves and stems.',
    treatment:'Apply azoxystrobin or fluxapyroxad. Improve field drainage.',
    prevention:'Remove plant debris after harvest. Avoid dense canopy.' },
  { key:'Tomato_Yellow_Leaf_Curl_Virus', label:'Yellow Leaf Curl Virus', severity:'Critical',
    symptoms:'Upward leaf curling, yellowing margins, stunted plant growth.',
    treatment:'No chemical cure. Remove and destroy infected plants immediately.',
    prevention:'Control whitefly populations. Use insect-proof nets and resistant varieties.' },
  { key:'Tomato_Mosaic_Virus',       label:'Tomato Mosaic Virus',    severity:'High',
    symptoms:'Mosaic light-dark green patterns on leaves; distortion and stunting.',
    treatment:'No cure. Remove infected plants. Disinfect all tools with bleach solution.',
    prevention:'Use virus-free certified seeds. Wash hands before handling plants.' },
  { key:'Tomato_healthy',            label:'Healthy',                severity:'None',
    symptoms:'No disease symptoms detected. Plant appears healthy.',
    treatment:'No treatment required.',
    prevention:'Continue regular monitoring, balanced fertilisation and irrigation.' },
];

const CLASS_PERF = [
  ['Bacterial Spot',94.1,95.3],['Early Blight',94.4,95.7],['Late Blight',93.2,94.1],
  ['Leaf Mold',91.5,92.8],['Septoria Leaf Spot',92.7,93.4],['Spider Mites',90.3,91.6],
  ['Target Spot',89.8,90.5],['Yellow Leaf Curl Virus',96.5,51.2],['Mosaic Virus',86.2,96.6],
  ['Healthy',99.7,99.5],
];

const SEV_CLASS = { None:'badge-none', Low:'badge-low', Medium:'badge-medium', High:'badge-high', Critical:'badge-critical' };
const SEV_COLOR = { None:'#16a34a', Low:'#ca8a04', Medium:'#ea580c', High:'#dc2626', Critical:'#9d174d' };

/* ─── SESSION ─── */
let currentUser = JSON.parse(sessionStorage.getItem('agroai_user') || 'null');
function setUser(u)  { currentUser = u; sessionStorage.setItem('agroai_user', JSON.stringify(u)); }
function clearUser() { currentUser = null; sessionStorage.removeItem('agroai_user'); }

/* ─── NAVIGATION ─── */
const PROTECTED = ['home','detect','model','results','about'];

function goPage(pageId) {
  if (PROTECTED.includes(pageId) && !currentUser) {
    showToast('Please login to access this page.', 'info');
    pageId = 'login';
  }
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tb-btn').forEach(b => b.classList.remove('active'));
  const target = document.getElementById('page-' + pageId);
  if (target) target.classList.add('active');
  const btn = document.querySelector(`.tb-btn[data-page="${pageId}"]`);
  if (btn) btn.classList.add('active');
  if (pageId === 'results') loadHistory();
  if (pageId === 'about')   initPerfBars();
  if (pageId === 'model')   initModelChart();
  window.scrollTo(0, 0);
}

document.querySelectorAll('.tb-btn').forEach(btn => {
  btn.addEventListener('click', () => goPage(btn.dataset.page));
});

/* ─── TOPBAR ─── */
function updateTopbar() {
  const nav   = document.getElementById('tb-nav');
  const guest = document.getElementById('tb-guest');
  const user  = document.getElementById('tb-user');
  const label = document.getElementById('tb-username-label');
  if (currentUser) {
    nav.style.display   = 'flex';
    guest.style.display = 'none';
    user.style.display  = 'flex';
    label.textContent   = currentUser.username;
  } else {
    nav.style.display   = 'none';
    guest.style.display = 'flex';
    user.style.display  = 'none';
  }
}

/* ─── LOGOUT ─── */
function logout() {
  clearUser();
  updateTopbar();
  clearImage();
  goPage('login');
  setTimeout(() => showToast('You have been successfully logged out.', 'success'), 200);
}

/* ─── SIGNUP ─── */
async function doSignup() {
  const username = document.getElementById('signup-username').value.trim();
  const email    = document.getElementById('signup-email').value.trim();
  const password = document.getElementById('signup-password').value;
  const confirm  = document.getElementById('signup-confirm').value;
  const errBox   = document.getElementById('signup-error');
  const okBox    = document.getElementById('signup-success');
  const btn      = document.getElementById('signup-btn');

  errBox.style.display = 'none';
  okBox.style.display  = 'none';

  if (!username || !email || !password || !confirm) { showAlert(errBox,'Please fill in all fields.'); return; }
  if (password !== confirm) { showAlert(errBox,'Passwords do not match.'); return; }
  if (password.length < 6) { showAlert(errBox,'Password must be at least 6 characters.'); return; }

  btn.disabled = true; btn.textContent = 'Creating account...';
  try {
    const res  = await fetch(`${API}/api/signup`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ username, email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Signup failed.');
    showAlert(okBox, 'Account created! Redirecting to login...');
    setTimeout(() => { clearSignupForm(); goPage('login'); }, 1500);
  } catch(err) { showAlert(errBox, err.message); }
  finally { btn.disabled = false; btn.textContent = 'Create Account'; }
}

function clearSignupForm() {
  ['signup-username','signup-email','signup-password','signup-confirm'].forEach(id => {
    document.getElementById(id).value = '';
  });
  document.getElementById('signup-error').style.display   = 'none';
  document.getElementById('signup-success').style.display = 'none';
}

/* ─── LOGIN ─── */
let _loginBusy = false;

async function doLogin() {
  if (_loginBusy) return;

  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;
  const errBox   = document.getElementById('login-error');
  const okBox    = document.getElementById('login-success');
  const btn      = document.getElementById('login-btn');

  errBox.style.display = 'none';
  if (okBox) okBox.style.display = 'none';

  if (!username || !password) { showAlert(errBox,'Please enter your username and password.'); return; }

  _loginBusy = true; btn.disabled = true; btn.textContent = 'Signing in...';
  try {
    const res  = await fetch(`${API}/api/login`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Invalid username or password.');
    setUser({ username: data.username, email: data.email });
    updateTopbar();
    document.getElementById('login-username').value = '';
    document.getElementById('login-password').value = '';
    errBox.style.display = 'none';
    goPage('home');
  } catch(err) { showAlert(errBox, err.message); }
  finally { _loginBusy = false; btn.disabled = false; btn.textContent = 'Sign In'; }
}

function showAlert(el, msg) { el.textContent = msg; el.style.display = 'block'; }

/* ─── ENTER KEY LOGIN ─── */
document.getElementById('login-password').addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin();
});

/* ─── DISEASE TABLE ─── */
function renderDiseaseTable() {
  const wrap = document.getElementById('disease-table');
  if (!wrap) return;
  wrap.innerHTML = DISEASES.map(d => `
    <div class="drow">
      <span class="drow-name">${d.label}</span>
      <span class="badge ${SEV_CLASS[d.severity]}">${d.severity}</span>
    </div>
  `).join('');
}

/* ─── FILE UPLOAD ─── */
const uploadZone  = document.getElementById('upload-zone');
const fileInput   = document.getElementById('file-input');
const previewImg  = document.getElementById('preview-img');
const placeholder = document.getElementById('upload-placeholder');
const clearBtn    = document.getElementById('clear-btn');

/*
  FIX — Double file-dialog bug:
  Previously, the zone had a click listener calling fileInput.click(), AND the
  "Browse Files" button had an inline onclick also calling fileInput.click().
  Clicking the button triggered BOTH handlers → dialog opened twice.

  Solution:
  - Remove the inline onclick from the button in HTML (now just id="browse-btn").
  - The button's listener calls e.stopPropagation() so the zone click does NOT fire.
  - The zone click listener checks if the button was the target and bails early.
  Result: exactly ONE file dialog opens regardless of where the user clicks.
*/
const browseBtn = document.getElementById('browse-btn');
if (browseBtn) {
  browseBtn.addEventListener('click', e => {
    e.stopPropagation(); // prevent zone click from also firing
    fileInput.click();
  });
}

uploadZone.addEventListener('click', e => {
  // If the click originated from the browse button, it is already handled above.
  if (browseBtn && browseBtn.contains(e.target)) return;
  fileInput.click();
});

fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

let _currentFile = null;

function handleFile(file) {
  _currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    placeholder.classList.add('hidden');
    previewImg.classList.remove('hidden');
    clearBtn.style.display = 'block';
    runDetection(file);
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  fileInput.value = '';
  previewImg.src  = '';
  _currentFile    = null;
  previewImg.classList.add('hidden');
  placeholder.classList.remove('hidden');
  clearBtn.style.display = 'none';
  document.getElementById('result-placeholder').classList.remove('hidden');
  document.getElementById('result-output').classList.add('hidden');
  document.getElementById('loading-box').classList.add('hidden');
}

/* ─── DETECTION ─── */
async function runDetection(file) {
  document.getElementById('result-placeholder').classList.add('hidden');
  document.getElementById('result-output').classList.add('hidden');
  document.getElementById('loading-box').classList.remove('hidden');

  try {
    const form = new FormData();
    form.append('file', file);
    const res  = await fetch(`${API}/api/predict`, { method:'POST', body:form });
    const data = await res.json();
    const info   = DISEASES.find(d => d.label === data.disease) || DISEASES[9];
    const result = { disease:info, confidence:data.confidence, severity:data.severity, annotatedUrl: data.annotated_url || null };
    showResult(result);
    await saveDetection(result);
  } catch(err) {
    document.getElementById('loading-box').classList.add('hidden');
    document.getElementById('result-output').innerHTML = `
      <div class="alert alert-error">
        Cannot connect to backend. Make sure
        <code>python start.py</code> is running and the server is on port 8000.
      </div>`;
    document.getElementById('result-output').classList.remove('hidden');
  }
}

function showResult({ disease, confidence, severity, annotatedUrl }) {
  document.getElementById('loading-box').classList.add('hidden');
  const sev   = severity || disease.severity;
  const pct   = Math.round(confidence * 100);
  const color = SEV_COLOR[sev] || '#2166c4';
  const annotatedHtml = annotatedUrl
    ? `<img src="${annotatedUrl}" class="annotated-img" alt="Annotated leaf" />`
    : '';
  document.getElementById('result-output').innerHTML = `
    ${annotatedHtml}
    <div class="result-block" style="border-left-color:${color};">
      <div class="result-name">${disease.label}</div>
      <div class="result-conf">
        Confidence: <strong>${pct}%</strong>
        &nbsp;
        <span class="badge ${SEV_CLASS[sev]}">${sev}</span>
      </div>
      <div class="conf-wrap">
        <div class="conf-fill" id="conf-bar" style="background:${color};"></div>
      </div>
      <div class="result-label">Symptoms</div>
      <div class="result-value">${disease.symptoms}</div>
      <div class="result-label">Treatment</div>
      <div class="result-value">${disease.treatment}</div>
      <div class="result-label">Prevention</div>
      <div class="result-value" style="margin-bottom:0">${disease.prevention}</div>
    </div>
  `;
  document.getElementById('result-output').classList.remove('hidden');
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const bar = document.getElementById('conf-bar');
    if (bar) bar.style.width = pct + '%';
  }));
}

/* ─── HISTORY ─── */
async function saveDetection({ disease, confidence, severity }) {
  if (!currentUser) return;
  try {
    await fetch(`${API}/api/save-detection`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        username: currentUser.username,
        disease:  disease.label,
        confidence: confidence,
        severity: severity || disease.severity,
      }),
    });
  } catch(_) {}
}

async function loadHistory() {
  if (!currentUser) return;
  try {
    const res  = await fetch(`${API}/api/history/${currentUser.username}`);
    const data = await res.json();
    renderHistory(data.history || []);
  } catch(_) { renderHistory([]); }
}

function renderHistory(rows) {
  const tbody = document.getElementById('history-body');
  ['m-total','m-diseased','m-healthy','m-avg'].forEach(id => {
    document.getElementById(id).textContent = id === 'm-avg' ? '—' : '0';
  });
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-row">No detections yet. Go to Detect to analyse a leaf.</td></tr>';
    return;
  }
  const total    = rows.length;
  const healthy  = rows.filter(r => r.disease === 'Healthy').length;
  const diseased = total - healthy;
  const avg      = rows.reduce((s,r) => s + r.confidence, 0) / total;
  document.getElementById('m-total').textContent    = total;
  document.getElementById('m-diseased').textContent = diseased;
  document.getElementById('m-healthy').textContent  = healthy;
  document.getElementById('m-avg').textContent      = (avg * 100).toFixed(1) + '%';
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.disease}</strong></td>
      <td>${(r.confidence * 100).toFixed(1)}%</td>
      <td><span class="badge ${SEV_CLASS[r.severity] || 'badge-none'}">${r.severity}</span></td>
      <td style="font-size:13px;color:#94a3b8">${r.timestamp}</td>
    </tr>
  `).join('');
}

async function clearHistory() {
  if (!currentUser) return;
  if (!confirm('Clear all your detection history?')) return;
  try {
    await fetch(`${API}/api/history/${currentUser.username}`, { method:'DELETE' });
    renderHistory([]);
  } catch(_) {}
}

/* ─── MODEL CHART ─── */
let _chartInstance = null;
function initModelChart() {
  const canvas = document.getElementById('model-chart');
  if (!canvas) return;
  if (_chartInstance) { _chartInstance.destroy(); _chartInstance = null; }
  _chartInstance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['YOLOv8\n(Proposed)', 'SVM', 'Random\nForest', 'Decision\nTree'],
      datasets: [{
        label: 'Accuracy (%)',
        data: [96.7, 83.1, 78.4, 71.2],
        backgroundColor: [
          'rgba(33,102,196,0.85)',
          'rgba(100,116,139,0.6)',
          'rgba(100,116,139,0.6)',
          'rgba(100,116,139,0.6)',
        ],
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y}%` } }
      },
      scales: {
        y: {
          min: 60, max: 100,
          ticks: { callback: v => v + '%', font: { family:'Poppins', size:11 }, color:'#94a3b8' },
          grid: { color:'rgba(0,0,0,0.05)' },
        },
        x: {
          ticks: { font: { family:'Poppins', size:12, weight:'600' }, color:'#475569' },
          grid: { display: false },
        }
      }
    }
  });
}

/* ─── PERF BARS ─── */
function initPerfBars() {
  const wrap = document.getElementById('perf-bars');
  if (!wrap || wrap.dataset.loaded) return;
  wrap.innerHTML = CLASS_PERF.map(([name, prec, rec]) => `
    <div class="perf-row">
      <div class="perf-head">
        <span class="perf-name">${name}</span>
        <span class="perf-vals">Precision ${prec}% &middot; Recall ${rec}%</span>
      </div>
      <div class="perf-bg">
        <div class="perf-fill" data-width="${((prec+rec)/2).toFixed(1)}"></div>
      </div>
    </div>
  `).join('');
  wrap.dataset.loaded = 'true';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    wrap.querySelectorAll('.perf-fill').forEach(b => { b.style.width = b.dataset.width + '%'; });
  }));
}

/* ─── FORGOT PASSWORD ─── */
let _forgotStep  = 1;
let _forgotEmail = '';

function resetForgotForm() {
  _forgotStep = 1; _forgotEmail = '';
  const fe = document.getElementById('forgot-email');
  if (fe) { fe.value = ''; fe.disabled = false; }
  const np = document.getElementById('forgot-newpw');     if (np) np.value = '';
  const cp = document.getElementById('forgot-confirmpw'); if (cp) cp.value = '';
  const eg = document.getElementById('forgot-error');     if (eg) eg.style.display = 'none';
  const og = document.getElementById('forgot-success');   if (og) og.style.display = 'none';
  const ng = document.getElementById('new-pw-group');     if (ng) ng.style.display = 'none';
  const cg = document.getElementById('confirm-pw-group'); if (cg) cg.style.display = 'none';
  const btn = document.getElementById('forgot-btn');      if (btn) btn.textContent = 'Verify Email';
}

async function doForgot() {
  const errBox = document.getElementById('forgot-error');
  const okBox  = document.getElementById('forgot-success');
  const btn    = document.getElementById('forgot-btn');
  errBox.style.display = 'none'; okBox.style.display = 'none';

  if (_forgotStep === 1) {
    const email = document.getElementById('forgot-email').value.trim();
    if (!email || !email.includes('@')) { showAlert(errBox,'Please enter a valid email.'); return; }
    btn.disabled = true; btn.textContent = 'Verifying...';
    try {
      const res  = await fetch(`${API}/api/verify-email`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Email not found.');
      _forgotEmail = email; _forgotStep = 2;
      showAlert(okBox, 'Email verified. Please enter your new password.');
      document.getElementById('new-pw-group').style.display     = 'block';
      document.getElementById('confirm-pw-group').style.display = 'block';
      document.getElementById('forgot-email').disabled = true;
      btn.textContent = 'Reset Password';
    } catch(err) { showAlert(errBox, err.message); }
    finally { btn.disabled = false; if (_forgotStep === 1) btn.textContent = 'Verify Email'; }

  } else {
    const np = document.getElementById('forgot-newpw').value;
    const cp = document.getElementById('forgot-confirmpw').value;
    if (!np || !cp)  { showAlert(errBox,'Please fill both password fields.'); return; }
    if (np !== cp)   { showAlert(errBox,'Passwords do not match.'); return; }
    if (np.length<6) { showAlert(errBox,'Password must be at least 6 characters.'); return; }
    btn.disabled = true; btn.textContent = 'Resetting...';
    try {
      const res  = await fetch(`${API}/api/reset-password`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ email: _forgotEmail, new_password: np }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Reset failed.');
      showAlert(okBox,'Password reset successfully! Redirecting to login...');
      setTimeout(() => { resetForgotForm(); goPage('login'); }, 2000);
    } catch(err) { showAlert(errBox, err.message); }
    finally { btn.disabled = false; if (_forgotStep === 2) btn.textContent = 'Reset Password'; }
  }
}

/* ─── INIT ─── */
renderDiseaseTable();
updateTopbar();
if (currentUser) { goPage('home'); }
else             { goPage('login'); }
