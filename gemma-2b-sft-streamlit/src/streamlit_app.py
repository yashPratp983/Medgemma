"""
Medical Kiosk AI — Streamlit Frontend
API: https://yash983-medical-kiosk-api.hf.space

Endpoints:
  POST /api/v1/chat          JSON {pulse_rate, oxygen_level, temperature, text}
                              → {treatment_plan}          ← only field returned

  POST /api/v1/classify-image  multipart {image, pulse_rate, oxygen_level, temperature}
                              → {predicted_disease, treatment_plan}
"""

import requests
import streamlit as st
import streamlit.components.v1 as components

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BASE_URL  = "https://yash983-medical-kiosk-api.hf.space"
CHAT_URL  = f"{BASE_URL}/api/v1/chat"
IMAGE_URL = f"{BASE_URL}/api/v1/classify-image"

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Kiosk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #f0f6ff; }

  .bubble-user {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    padding: 11px 17px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 80px;
    font-size: 0.93rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(37,99,235,.28);
  }
  .bubble-meta-user { text-align: right; font-size: 0.73rem; color: #64748b; margin: 0 3px 1px; }

  .transcript-box {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 10px;
    padding: 10px 14px;
    font-style: italic;
    color: #475569;
    min-height: 42px;
    font-size: 0.9rem;
    margin: 8px 0;
  }
  .transcript-ready {
    border-color: #2563eb;
    background: #eff6ff;
    color: #1d4ed8;
    font-style: normal;
  }

  .input-panel {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 14px rgba(0,0,0,.07);
    margin-top: 10px;
  }

  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  section[data-testid="stSidebar"] { background: #0f172a; }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ────────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "voice_transcript": "",
    "voice_result": None,
    "voice_result_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def vitals_ok(p, o, t):
    return bool(p.strip() and o.strip() and t.strip())


def call_chat(text: str, pulse: str, oxygen: str, temp: str) -> dict:
    payload = {
        "pulse_rate":   pulse,
        "oxygen_level": oxygen,
        "temperature":  temp,
        "text":         text,
    }
    r = requests.post(CHAT_URL, json=payload, timeout=None)
    r.raise_for_status()
    return r.json()


def call_classify_image(image_bytes: bytes, filename: str,
                        pulse: str, oxygen: str, temp: str) -> dict:
    files = {"image": (filename, image_bytes, "image/jpeg")}
    data  = {"pulse_rate": pulse, "oxygen_level": oxygen, "temperature": temp}
    r = requests.post(IMAGE_URL, files=files, data=data, timeout=90)
    r.raise_for_status()
    return r.json()


# ─── CARD: Chat result  (only treatment_plan is returned) ─────────────────────
def render_chat_card(result: dict):
    """Render diagnosis result using native Streamlit widgets — no iframes."""
    treatment = result.get("treatment_plan") or "No treatment plan returned."
    with st.container(border=True):
        st.markdown("#### 🧪 Diagnosis & Treatment Plan")
        st.caption("TREATMENT PLAN")
        st.info(treatment, icon="💊")


# ─── CARD: Image-classify result  (predicted_disease + treatment_plan) ────────
def render_image_card(result: dict):
    """Render image classification result using native Streamlit widgets."""
    disease    = result.get("predicted_disease") or "Unknown"
    treatment  = result.get("treatment_plan")    or "No treatment plan returned."
    confidence = result.get("confidence")

    with st.container(border=True):
        st.markdown("#### 🧪 Skin Lesion Classification")
        col_d, col_c = st.columns([2, 1])
        with col_d:
            st.caption("PREDICTED DISEASE")
            st.markdown(
                f"<span style='background:#eff6ff;border:1px solid #bfdbfe;"
                f"border-radius:20px;padding:5px 14px;font-size:.9rem;"
                f"color:#1d4ed8;font-weight:700;'>🦠 {disease}</span>",
                unsafe_allow_html=True,
            )
        if confidence is not None:
            pct = int(float(confidence) * 100)
            with col_c:
                st.metric("Confidence", f"{pct}%")
            st.progress(pct / 100)

        st.caption("TREATMENT PLAN")
        st.info(treatment, icon="💊")


# ─── STT Widget (deployment-safe) ────────────────────────────────────────────
# Uses components.html() (one-way, no local asset serving) for the recorder UI.
# The recognised transcript is copied to clipboard by JS; the user pastes it
# into the st.text_area below — zero dependency on declare_component.
def render_stt_widget():
    """Render the voice recorder iframe. Returns nothing — transcript is read
    from the st.text_area that sits below it in the Voice tab."""
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
      *{box-sizing:border-box;margin:0;padding:0;}
      body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent;}
      #btn{
        width:100%;display:flex;align-items:center;justify-content:center;
        gap:10px;padding:12px 20px;
        background:linear-gradient(135deg,#2563eb,#1d4ed8);
        color:#fff;border:none;border-radius:12px;
        font-size:15px;font-weight:600;cursor:pointer;
        box-shadow:0 3px 10px rgba(37,99,235,.35);transition:all .2s;
      }
      #btn:hover{filter:brightness(1.08);transform:translateY(-1px);}
      #btn.rec{background:linear-gradient(135deg,#dc2626,#b91c1c);animation:pulse 1.3s infinite;}
      @keyframes pulse{
        0%{box-shadow:0 0 0 0 rgba(220,38,38,.55);}
        70%{box-shadow:0 0 0 12px rgba(220,38,38,0);}
        100%{box-shadow:0 0 0 0 rgba(220,38,38,0);}
      }
      #status{font-size:12px;color:#64748b;text-align:center;margin-top:8px;}
      #live{
        font-size:13px;color:#1d4ed8;background:#eff6ff;
        border-radius:8px;padding:8px 12px;margin-top:8px;
        min-height:32px;display:none;
      }
      #copy-btn{
        display:none;width:100%;margin-top:8px;padding:8px;
        background:#10b981;color:#fff;border:none;border-radius:8px;
        font-size:13px;font-weight:600;cursor:pointer;
      }
      #copy-btn:hover{filter:brightness(1.08);}
      .err{color:#ef4444!important;}
    </style>
    </head><body>
    <button id="btn" onclick="toggle()">🎤 Start Recording</button>
    <div id="status">Record your symptoms, then copy the transcript below ↓</div>
    <div id="live"></div>
    <button id="copy-btn" onclick="copyText()">📋 Copy transcript to clipboard</button>
    <script>
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      let rec=null, going=false, finalText="";
      if(!SR){
        document.getElementById("btn").disabled=true;
        document.getElementById("status").textContent="⚠️ Not supported — use Chrome or Edge";
        document.getElementById("status").className="err";
      } else {
        rec=new SR(); rec.lang="en-US"; rec.interimResults=true;
        rec.continuous=false; rec.maxAlternatives=1;
        rec.onstart=()=>{
          finalText="";
          document.getElementById("btn").textContent="🔴 Stop Recording";
          document.getElementById("btn").className="rec";
          document.getElementById("status").textContent="Listening… speak now";
          document.getElementById("live").style.display="block";
          document.getElementById("live").textContent="";
          document.getElementById("copy-btn").style.display="none";
        };
        rec.onresult=e=>{
          let interim="",fin="";
          for(let i=e.resultIndex;i<e.results.length;i++){
            if(e.results[i].isFinal) fin+=e.results[i][0].transcript;
            else interim+=e.results[i][0].transcript;
          }
          if(fin) finalText=fin;
          document.getElementById("live").textContent=(finalText||interim)+(interim?"…":"");
        };
        rec.onend=()=>{
          going=false;
          document.getElementById("btn").textContent="🎤 Start Recording";
          document.getElementById("btn").className="";
          if(finalText){
            document.getElementById("status").textContent="✅ Done — copy transcript then paste below ↓";
            document.getElementById("live").textContent=finalText;
            document.getElementById("copy-btn").style.display="block";
          } else {
            document.getElementById("status").textContent="No speech detected — try again";
          }
        };
        rec.onerror=e=>{
          going=false;
          document.getElementById("btn").textContent="🎤 Start Recording";
          document.getElementById("btn").className="";
          const msgs={"not-allowed":"⛔ Microphone permission denied","no-speech":"No speech detected","network":"Network error"};
          document.getElementById("status").textContent=msgs[e.error]||("Error: "+e.error);
        };
      }
      function toggle(){if(!rec)return; if(going){rec.stop();going=false;}else{rec.start();going=true;}}
      function copyText(){
        if(!finalText) return;
        navigator.clipboard.writeText(finalText).then(()=>{
          document.getElementById("copy-btn").textContent="✅ Copied!";
          setTimeout(()=>{ document.getElementById("copy-btn").textContent="📋 Copy transcript to clipboard"; },2000);
        }).catch(()=>{
          // Fallback for browsers that block clipboard in iframes
          const ta=document.createElement("textarea");
          ta.value=finalText; document.body.appendChild(ta);
          ta.select(); document.execCommand("copy"); document.body.removeChild(ta);
          document.getElementById("copy-btn").textContent="✅ Copied!";
          setTimeout(()=>{ document.getElementById("copy-btn").textContent="📋 Copy transcript to clipboard"; },2000);
        });
      }
    </script>
    </body></html>"""
    components.html(html, height=175, scrolling=False)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Medical Kiosk AI")
    st.markdown("*Symptom Diagnosis + Skin Analysis*")
    st.markdown("---")
    st.markdown("### 📊 Patient Vitals")
    st.caption("Required for all diagnoses")

    pulse  = st.text_input("❤️ Pulse Rate (bpm)",     value="72",   placeholder="e.g. 72")
    oxygen = st.text_input("🫁 Oxygen Level (SpO2%)",  value="98",   placeholder="e.g. 98")
    temp   = st.text_input("🌡️ Temperature (°F)",      value="98.6", placeholder="e.g. 98.6")

    if vitals_ok(pulse, oxygen, temp):
        st.success("✅ Vitals set")
    else:
        st.error("⚠️ Fill all vitals")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.voice_transcript = ""
        st.rerun()

    st.markdown("---")
    st.markdown("**API**")
    st.code("POST /api/v1/chat\nPOST /api/v1/classify-image", language=None)
    st.caption(f"Base: `{BASE_URL}`")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# 🏥 Medical Kiosk Assistant")
st.markdown("Describe symptoms, upload a skin lesion image, or speak your query — and receive an AI-powered diagnosis.")
st.markdown("---")

tab_chat, tab_image, tab_camera, tab_voice = st.tabs(
    ["💬 Chat Diagnosis", "🖼️ Upload Image", "📷 Take Photo", "🎤 Voice Input"]
)


# ════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT CHAT
# /api/v1/chat → { treatment_plan }
# ════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px; color:#94a3b8;">
          <div style="font-size:2.8rem;">💬</div>
          <div style="font-size:1rem; margin-top:8px;">
            Describe your symptoms and the AI will diagnose.<br>
            Make sure your <b>vitals are set</b> in the sidebar.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        for item in st.session_state.chat_history:
            st.markdown('<div class="bubble-meta-user">You 👤</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble-user">{item["text"]}</div>', unsafe_allow_html=True)
            if item.get("result"):
                # Chat returns only treatment_plan
                render_chat_card(item["result"])
            elif item.get("error"):
                st.error(item["error"])

    st.markdown("---")
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_area(
            "Describe your symptoms:",
            placeholder="e.g. I have a high fever, sore throat, and body aches for the past 2 days…",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "📤 Send & Diagnose", use_container_width=True, type="primary"
        )

    if submitted:
        if not vitals_ok(pulse, oxygen, temp):
            st.warning("⚠️ Please fill in all vitals in the sidebar before diagnosing.")
        elif not user_text.strip():
            st.warning("Please enter your symptoms.")
        else:
            with st.spinner("🤔 Analysing symptoms…"):
                try:
                    result = call_chat(user_text.strip(), pulse, oxygen, temp)
                    st.session_state.chat_history.append({"text": user_text.strip(), "result": result})
                except Exception as e:
                    st.session_state.chat_history.append({"text": user_text.strip(), "error": f"API Error: {e}"})
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — SKIN IMAGE ANALYSIS
# /api/v1/classify-image → { predicted_disease, treatment_plan }
# ════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown("### 🔬 Skin Lesion Classifier")
    st.caption("Upload a JPEG photo of the skin area.")

    col_upload, col_preview = st.columns([1.4, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload skin lesion image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        img_bytes = uploaded.read() if uploaded else None

    with col_preview:
        if img_bytes:
            st.image(img_bytes, caption="Preview", use_container_width=True)
        else:
            st.markdown("""
            <div style="border:2px dashed #cbd5e1;border-radius:12px;
                        height:180px;display:flex;align-items:center;
                        justify-content:center;color:#94a3b8;font-size:.9rem;">
              📷 Image preview
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔬 Classify & Diagnose", use_container_width=True,
                 type="primary", disabled=(img_bytes is None)):
        if not vitals_ok(pulse, oxygen, temp):
            st.warning("⚠️ Please fill in all vitals in the sidebar.")
        else:
            with st.spinner("🔬 Running skin disease classifier…"):
                try:
                    result = call_classify_image(img_bytes, uploaded.name, pulse, oxygen, temp)
                    st.success("Classification complete!")
                    render_image_card(result)
                except requests.exceptions.HTTPError as e:
                    st.error(f"API returned an error: {e.response.status_code} — {e.response.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    if img_bytes is None:
        st.info("📎 Upload an image to enable the classify button.")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — CAMERA CAPTURE
# ════════════════════════════════════════════════════════════════════
with tab_camera:
    st.markdown("### 📷 Take a Photo")
    st.caption("Use your device camera to capture a skin lesion.")

    cam_img = st.camera_input("Point camera at the skin area and click capture",
                              label_visibility="collapsed")

    if cam_img:
        cam_bytes = cam_img.getvalue()
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.image(cam_bytes, caption="Captured photo", use_container_width=True)
        with col_b:
            st.markdown("**Photo captured!** ✅")
            st.markdown("Vitals will be taken from the sidebar.")
            st.markdown(f"- ❤️ Pulse: `{pulse}` bpm")
            st.markdown(f"- 🫁 SpO2: `{oxygen}%`")
            st.markdown(f"- 🌡️ Temp: `{temp}°F`")

        st.markdown("---")

        if st.button("🔬 Classify Captured Photo", use_container_width=True, type="primary"):
            if not vitals_ok(pulse, oxygen, temp):
                st.warning("⚠️ Please fill in all vitals in the sidebar.")
            else:
                with st.spinner("🔬 Classifying skin disease from photo…"):
                    try:
                        result = call_classify_image(
                            cam_bytes, "camera_capture.jpg", pulse, oxygen, temp
                        )
                        st.success("Classification complete!")
                        render_image_card(result)
                    except requests.exceptions.HTTPError as e:
                        st.error(f"API error {e.response.status_code}: {e.response.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
    else:
        st.info("📸 Click **Take photo** above to capture from your camera.")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — VOICE INPUT
# ════════════════════════════════════════════════════════════════════
with tab_voice:
    st.markdown("### 🎤 Voice Symptom Input")
    st.caption("Uses your browser's built-in **Web Speech API** (Chrome / Edge). "
               "Record → copy the transcript → paste it below → send.")

    # ── 1. Recorder widget (components.html — no local asset serving) ──────────
    render_stt_widget()

    st.markdown("---")

    # ── 2. Paste area — user pastes the copied transcript here ─────────────────
    pasted = st.text_area(
        "📋 Paste your transcript here:",
        value=st.session_state.voice_transcript,
        placeholder="Paste the copied transcript, or type your symptoms manually…",
        height=90,
        key="voice_paste_area",
    )
    # Keep session state in sync so Clear works
    st.session_state.voice_transcript = pasted

    current = pasted.strip()

    col_send, col_clear = st.columns(2)
    with col_send:
        send_voice = st.button(
            "📤 Send to Diagnosis API",
            use_container_width=True,
            type="primary",
            disabled=not bool(current),
        )
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.voice_transcript = ""
            st.session_state.voice_result = None
            st.session_state.voice_result_text = ""
            st.rerun()

    if send_voice:
        if not vitals_ok(pulse, oxygen, temp):
            st.warning("⚠️ Please fill in all vitals in the sidebar.")
        else:
            with st.spinner("🤔 Diagnosing from voice transcript…"):
                try:
                    result = call_chat(current, pulse, oxygen, temp)
                    # Store result in session state — do NOT rerun, render directly below
                    st.session_state.voice_result = result
                    st.session_state.voice_result_text = current
                    st.session_state.chat_history.append({
                        "text": f"🎤 {current}",
                        "result": result,
                    })
                    st.session_state.voice_transcript = ""
                except Exception as e:
                    st.error(f"API Error: {e}")

    # Always render the stored result so it survives any rerun
    if st.session_state.get("voice_result"):
        st.success("Diagnosis complete!")
        st.markdown(f"**Your symptoms:** *{st.session_state.voice_result_text}*")
        render_chat_card(st.session_state.voice_result)