# Cursor Project Instructions

## Overview
This project combines a **frontend (from Figma)** and a **backend (built on macOS, now running on Windows)**.  
The system must work seamlessly on **Windows** with **Python** (for backend/ML model) and **Node.js** (for frontend tooling).

---

## Environment Setup Rules

### 1. Python Setup
- Always create or activate a Python virtual environment before running backend code.
- Use the following commands:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- Install dependencies only inside the virtual environment:
  ```bash
  pip install -r requirements.txt
  ```
- Backend must run through the main entry file (for example: `app.py` or `server.py`).

---

### 2. Node.js / Frontend Setup
- Ensure **Node.js** and **npm** (or **yarn**) are available.
- Before running or building the frontend, install dependencies:
  ```bash
  npm install
  ```
- If Figma export is React or TypeScript:
  - Use `npm run start` to serve the frontend.
  - If it’s pure HTML/JS/CSS, just ensure static files are linked correctly.

---

### 3. Cross-Platform Guidelines
- Remove or ignore macOS system files:
  - `.DS_Store`
  - `.localized`
  - `.AppleDouble`
  - `.Trashes`
- Do not modify core code files unless explicitly instructed.
- Avoid absolute macOS-style paths.

---

### 4. Cursor Behavior Rules
- **Always read all project files** before performing major edits.
- **Never delete or overwrite** existing source code.
- **Ask before refactoring or restructuring directories.**
- **Preserve comments, documentation, and progress.**
- **Follow these steps before execution:**
  1. Activate the Python virtual environment.
  2. Start backend server.
  3. Verify that the frontend can connect to backend endpoints.

---

### 5. Integration Expectations
- Frontend communicates with backend via REST API (HTTP requests).
- All integration must remain modular and easy to debug.
- Log key interactions (requests/responses) for debugging.

---

### 6. Target Platform
- Primary OS: Windows 10/11
- Secondary OS: macOS (compatibility only)
- Browser Target: Chromium-based (Edge/Chrome)
