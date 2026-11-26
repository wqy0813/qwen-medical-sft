const labels = {
  fixed: "选错 → 选对",
  format: "无法解析 → 选对",
  regressed: "选对 → 未答对",
  both_correct: "两次都答对",
  both_missed: "两次都未答对",
};

const element = id => document.getElementById(id);
const text = (tag, className, value) => {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = value;
  return node;
};

function answerCard(label, output, option, gold) {
  const card = text("div", "answer", "");
  const correct = option === gold;
  const verdict = option ? (correct ? "答对" : "选错") : "无法解析，计错";
  card.append(
    text("h4", "", label),
    text("pre", "", output || "（空输出）"),
    text("p", correct ? "correct" : "miss", option ? option + " · " + verdict : verdict),
  );
  return card;
}

function showQuestion(row) {
  const detail = element("detail");
  detail.replaceChildren();
  if (!row) {
    detail.append(text("p", "", "没有匹配的题目。试试其他 ID 或变化类型。"));
    return;
  }
  const title = text("div", "case-head", "");
  title.append(text("h3", "", "题目 #" + row.id), text("span", "badge " + row.outcome, labels[row.outcome]));
  const metadata = "上游划分 " + row.source_split + " · 标准答案 " + row.gold + " · 选项 " + row.allowed_options;
  detail.append(title, text("p", "case-meta", metadata));

  if (row.question) {
    const question = text("p", "stem", row.question);
    question.lang = "en";
    detail.append(question);
    const options = text("ul", "options", "");
    options.lang = "en";
    for (const [letter, content] of Object.entries(row.options)) {
      options.append(text("li", letter === row.gold ? "gold" : "", letter + ". " + content));
    }
    detail.append(options);
  }
  const answers = text("div", "answer-grid", "");
  answers.append(
    answerCard("Base 原始输出", row.base_output, row.base_answer, row.gold),
    answerCard("SFT 原始输出", row.sft_output, row.sft_answer, row.gold),
  );
  detail.append(answers);
}

function start(data) {
  for (const version of ["base", "sft"]) {
    const summary = data.metrics[version];
    const rate = summary.normalized_accuracy * 100;
    element(version + "-rate").textContent = rate.toFixed(2) + "%";
    element(version + "-bar").style.width = rate + "%";
    element(version + "-detail").textContent = summary.normalized_correct + " / " + summary.n
      + " 答对 · " + summary.invalid_count + " 条无法解析";
  }
  const delta = (data.metrics.sft.normalized_accuracy - data.metrics.base.normalized_accuracy) * 100;
  element("delta").textContent = (delta >= 0 ? "+" : "") + delta.toFixed(2) + " 个百分点";

  const filter = element("outcome-filter");
  const query = element("query");
  const list = element("question-list");
  const previous = element("previous");
  const next = element("next");
  const pageSize = 6;
  let page = 0;
  let selected = data.featured.fixed;

  function select(row) {
    selected = row?.id;
    for (const button of list.querySelectorAll("button")) {
      button.setAttribute("aria-current", String(Number(button.dataset.id) === selected));
    }
    showQuestion(row);
  }

  function render() {
    const id = query.value.trim();
    const matches = data.rows.filter(row =>
      (filter.value === "all" || row.outcome === filter.value) && (!id || String(row.id) === id));
    const pages = Math.ceil(matches.length / pageSize);
    const visible = matches.slice(page * pageSize, (page + 1) * pageSize);
    list.replaceChildren();
    element("match-count").textContent = matches.length + " 道题";
    element("page-number").textContent = pages ? (page + 1) + " / " + pages : "—";
    previous.disabled = page === 0;
    next.disabled = page + 1 >= pages;

    for (const row of visible) {
      const button = text("button", "topic", "");
      button.type = "button";
      button.dataset.id = row.id;
      button.append(text("span", "", "#" + row.id), text("span", "", labels[row.outcome]));
      button.addEventListener("click", () => select(row));
      list.append(button);
    }
    select(visible.find(row => row.id === selected) || visible[0]);
    for (const button of document.querySelectorAll("[data-outcome]")) {
      button.setAttribute("aria-pressed", String(button.dataset.outcome === filter.value));
    }
  }

  function applyFilters() {
    page = 0;
    render();
  }
  filter.addEventListener("change", applyFilters);
  query.addEventListener("input", applyFilters);
  previous.addEventListener("click", () => { page -= 1; render(); });
  next.addEventListener("click", () => { page += 1; render(); });
  for (const button of document.querySelectorAll("[data-outcome]")) {
    const key = button.dataset.outcome;
    element(key + "-count").textContent = data.counts[key] ?? 0;
    button.addEventListener("click", () => {
      filter.value = key;
      query.value = "";
      selected = data.featured[key];
      applyFilters();
      element("compare").scrollIntoView();
    });
  }
  element("explorer").hidden = false;
  render();
}

fetch("./data.json")
  .then(response => {
    if (!response.ok) throw new Error("Could not load results");
    return response.json();
  })
  .then(start)
  .catch(() => { element("load-error").hidden = false; });
