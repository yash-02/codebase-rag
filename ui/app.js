const repoPath = document.querySelector("#repoPath");
const indexBtn = document.querySelector("#indexBtn");
const indexStatus = document.querySelector("#indexStatus");
const askForm = document.querySelector("#askForm");
const askBtn = document.querySelector("#askBtn");
const question = document.querySelector("#question");
const topK = document.querySelector("#topK");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const sourceEmpty = document.querySelector("#sourceEmpty");

function setMessage(node, text, type) {
  node.className = "message" + (type ? " " + type : "");
  node.textContent = text;
}

function setBusy(button, busy, text) {
  button.disabled = busy;
  button.textContent = busy ? text : button.dataset.label;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatInlineMarkdown(text) {
  return escapeHtml(text).replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderAnswerMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let inCodeBlock = false;
  let codeLines = [];
  let inList = false;
  let listType = null;
  let inTable = false;

  function closeList() {
    if (!inList) return;
    html.push(listType === "ol" ? "</ol>" : "</ul>");
    inList = false;
    listType = null;
  }

  function openList(type) {
    if (inList && listType === type) return;
    closeList();
    html.push(type === "ol" ? "<ol>" : "<ul>");
    inList = true;
    listType = type;
  }

  function closeTable() {
    if (!inTable) return;
    html.push("</tbody></table>");
    inTable = false;
  }

  for (const line of lines) {
    if (line.startsWith("```")) {
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCodeBlock = false;
      } else {
        closeList();
        closeTable();
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    if (!line.trim()) {
      closeList();
      closeTable();
      continue;
    }

    const heading = line.match(/^##\s+(.+)$/);
    if (heading) {
      closeList();
      closeTable();
      html.push(`<h2>${formatInlineMarkdown(heading[1])}</h2>`);
      continue;
    }

    if (/^\|.+\|$/.test(line.trim())) {
      const cells = line
        .trim()
        .slice(1, -1)
        .split("|")
        .map((cell) => cell.trim());
      const isDivider = cells.every((cell) => /^:?-{3,}:?$/.test(cell));
      if (isDivider) continue;

      closeList();
      if (!inTable) {
        html.push("<table><tbody>");
        inTable = true;
      }

      html.push(`<tr>${cells.map((cell) => `<td>${formatInlineMarkdown(cell)}</td>`).join("")}</tr>`);
      continue;
    }

    closeTable();

    const bullet = line.match(/^-\s+(.+)$/);
    if (bullet) {
      openList("ul");
      html.push(`<li>${formatInlineMarkdown(bullet[1])}</li>`);
      continue;
    }

    const numbered = line.match(/^\d+\.\s+(.+)$/);
    if (numbered) {
      openList("ol");
      html.push(`<li>${formatInlineMarkdown(numbered[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${formatInlineMarkdown(line)}</p>`);
  }

  closeList();
  closeTable();
  if (inCodeBlock) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  return html.join("");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    const message = data.detail ? JSON.stringify(data.detail) : "Request failed";
    throw new Error(message);
  }
  return data;
}

indexBtn.dataset.label = indexBtn.textContent;
askBtn.dataset.label = askBtn.textContent;

indexBtn.addEventListener("click", async () => {
  setBusy(indexBtn, true, "Indexing...");
  setMessage(indexStatus, "Parsing files, building the call graph, and storing embeddings.", "");
  try {
    const data = await postJson("/index", { repo_path: repoPath.value.trim() || "./test_repo" });
    setMessage(indexStatus, `Indexed ${data.chunks_indexed} chunks.`, "ok");
  } catch (err) {
    setMessage(indexStatus, err.message, "err");
  } finally {
    setBusy(indexBtn, false);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) {
    answer.className = "answer empty";
    answer.textContent = "Enter a question first.";
    return;
  }

  setBusy(askBtn, true, "Thinking...");
  answer.className = "answer";
  answer.textContent = "Retrieving code context and generating an answer...";
  sources.innerHTML = "";
  sourceEmpty.hidden = false;
  sourceEmpty.textContent = "Waiting for answer.";

  try {
    const data = await postJson("/ask", {
      question: text,
      top_k: Number(topK.value),
    });
    answer.innerHTML = renderAnswerMarkdown(data.answer || "No answer returned.");
    sources.innerHTML = "";
    for (const source of data.sources || []) {
      const item = document.createElement("li");
      item.textContent = source;
      sources.appendChild(item);
    }
    sourceEmpty.hidden = Boolean((data.sources || []).length);
    sourceEmpty.textContent = "No sources returned.";
  } catch (err) {
    answer.className = "answer empty";
    answer.textContent = err.message;
  } finally {
    setBusy(askBtn, false);
  }
});
