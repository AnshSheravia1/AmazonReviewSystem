const el = (id) => document.getElementById(id);

async function checkHealth() {
  const badge = el("health");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    badge.textContent = `● ${data.books_loaded.toLocaleString()} books loaded`;
    badge.className = "health ok";
  } catch (err) {
    badge.textContent = "● API unreachable";
    badge.className = "health error";
  }
}

async function analyzeSentiment() {
  const text = el("sentiment-text").value;
  const resultBox = el("sentiment-result");
  const button = el("sentiment-submit");

  button.disabled = true;
  resultBox.hidden = false;
  resultBox.innerHTML = "Analyzing&hellip;";

  try {
    const res = await fetch("/sentiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      resultBox.innerHTML = `<span class="error">${escapeHtml(data.detail || "Request failed.")}</span>`;
      return;
    }

    resultBox.innerHTML = `
      <span class="label ${data.sentiment}">${data.sentiment}</span>
      &mdash; compound score: <strong>${data.compound_score.toFixed(4)}</strong>
    `;
  } catch (err) {
    resultBox.innerHTML = `<span class="error">Could not reach the API.</span>`;
  } finally {
    button.disabled = false;
  }
}

let searchTimer = null;
let selectedBookId = null;

function debounceSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(runSearch, 250);
}

async function runSearch() {
  const q = el("book-search").value.trim();
  const list = el("book-suggestions");

  if (!q) {
    list.hidden = true;
    list.innerHTML = "";
    return;
  }

  try {
    const res = await fetch(`/books?q=${encodeURIComponent(q)}&limit=8`);
    const books = await res.json();

    list.innerHTML = "";
    if (books.length === 0) {
      list.hidden = true;
      return;
    }

    for (const book of books) {
      const li = document.createElement("li");
      li.innerHTML = `${escapeHtml(book.title)} <span class="genre">${escapeHtml(book.genre)}</span>`;
      li.addEventListener("click", () => selectBook(book));
      list.appendChild(li);
    }
    list.hidden = false;
  } catch (err) {
    list.hidden = true;
  }
}

function selectBook(book) {
  selectedBookId = book.book_id;
  el("selected-title").textContent = book.title;
  el("selected-id").textContent = `(${book.genre})`;
  el("selected-book").hidden = false;
  el("book-suggestions").hidden = true;
  el("book-search").value = book.title;
  el("recommend-result").innerHTML = "";
}

async function getRecommendations() {
  if (!selectedBookId) return;

  const topN = el("top-n").value;
  const button = el("recommend-submit");
  const box = el("recommend-result");

  button.disabled = true;
  box.innerHTML = "Loading&hellip;";

  try {
    const res = await fetch(`/recommend/${encodeURIComponent(selectedBookId)}?top_n=${topN}`);
    const data = await res.json();

    if (!res.ok) {
      box.innerHTML = `<span class="error">${escapeHtml(data.detail || "Request failed.")}</span>`;
      return;
    }

    if (data.length === 0) {
      box.innerHTML = "<p class=\"muted\">No similar books found.</p>";
      return;
    }

    box.innerHTML = "";
    for (const rec of data) {
      const div = document.createElement("div");
      div.className = "rec-item";
      div.innerHTML = `
        <span class="title">${escapeHtml(rec.title)} <span class="muted">(${escapeHtml(rec.genre)})</span></span>
        <span class="score">${rec.similarity_score.toFixed(2)}</span>
      `;
      box.appendChild(div);
    }
  } catch (err) {
    box.innerHTML = `<span class="error">Could not reach the API.</span>`;
  } finally {
    button.disabled = false;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

el("sentiment-submit").addEventListener("click", analyzeSentiment);
el("book-search").addEventListener("input", debounceSearch);
el("recommend-submit").addEventListener("click", getRecommendations);

document.addEventListener("click", (e) => {
  if (!el("book-suggestions").contains(e.target) && e.target !== el("book-search")) {
    el("book-suggestions").hidden = true;
  }
});

checkHealth();
