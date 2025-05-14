# 🧠 AI-Powered Project Bootstrapper (PoC)

This is a **proof-of-concept** tool that uses **Large Language Models (LLMs)**, **GitHub MCP**, and **Terminal MCP** to turn a plain-English software idea into a ready-to-explore project folder — complete with directory structure and cloned repositories from GitHub.

> ✨ Describe your project in plain English — let the system scaffold it for you.

---

## 🚀 What It Does

1. **Input**: You describe a project idea (e.g., *“Video chat app with Go backend and WebRTC frontend”*).
2. **Decomposition**: An LLM breaks the idea into components (e.g., signaling, media, UI).
3. **Repo Discovery**: Uses **GitHub MCP** to search for existing, relevant open-source repositories.
4. **Setup**: Uses **Terminal MCP** to:

   * Create a clean directory structure
   * Clone selected repositories into their respective folders
   * Optionally generate README or manifest files

---

## 📁 Example Output

```
video-call-app/
├── backend/
│   └── go-webrtc-server/        # Cloned GitHub repo
├── frontend/
│   └── htmx-video-ui/           # Cloned GitHub repo
├── signaling/
│   └── ws-signaling-server/     # Cloned GitHub repo
└── README.md
```

---

## 🧠 Technologies

* **LLMs** (OpenAI, Ollama, etc.): For reasoning, decomposition, and repo mapping
* **GitHub MCP**: Programmatic GitHub search and metadata extraction
* **Terminal MCP**: Executes filesystem and Git operations for setup

---

## 🛠 Requirements

* Access to an LLM (API or local)
* GitHub MCP access and credentials
* Terminal MCP enabled locally
* Node.js or Python (depending on the script)

---

## 🧪 How to Use (PoC)

1. Clone this repository:

   ```bash
   git clone https://github.com/vish419/project-bootstrapper.git
   cd project-bootstrapper
   ```
2. Run the bootstrap script:

   ```bash
   node bootstrap.js
   ```
3. Enter your project description:

   ```
   > Collaborative drawing app with voice chat using WebRTC and Go
   ```
4. The system will:

   * Break the idea into components
   * Suggest repositories
   * Set up the directory structure and clone repos

---

## 🎯 Example Prompts

* "Real-time quiz app with Firebase and Svelte"
* "REST API gateway in Go with authentication and rate limiting"
* "Slack bot that syncs with Trello boards"

---

## 🧱 Project Status

**Stage**: Proof of Concept
This project is actively being developed to test automation of:

* Repository selection
* Directory structuring
* Intelligent project bootstrapping

---

## 📅 Roadmap

* [ ] Automatic repo ranking and validation
* [ ] CLI config and command options
* [ ] Custom template fallbacks when no repo is found
* [ ] Optional Docker setup for cloned components
* [ ] Integration with project management tools (Trello, GitHub Projects)

---

## 🤝 Contributing

Contributions, ideas, and pull requests are welcome!

---

## 📜 License

MIT License — see [`LICENSE`](./LICENSE)

---

## 👤 Author

Maintained by [@vish419](https://github.com/vish419)

> *Bootstrap real-world code with real-world repositories — from idea to implementation.*
