// A small Markdown renderer, deliberately not a parser library.
//
// The model's answer is untrusted text, so every block is escaped before any
// markup is added: the tags below are the only ones that can reach the page.
// Supported: fenced code with a copy button, inline code, bold, italic, links,
// headings, bullet and numbered lists, blockquotes, paragraphs.

const ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (character) => ESCAPES[character]);
}

/// Only http(s) links become anchors; anything else stays text.
function safeHref(url) {
  return /^https?:\/\//i.test(url) ? url : null;
}

function inline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  html = html.replace(/(^|[\s(])_([^_\n]+)_/g, "$1<em>$2</em>");
  html = html.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (match, label, url) => {
    const href = safeHref(url);
    return href
      ? `<a href="${href}" target="_blank" rel="noreferrer noopener">${label}</a>`
      : match;
  });
  return html;
}

function codeBlock(language, code) {
  const pre = document.createElement("pre");
  if (language) {
    const tag = document.createElement("span");
    tag.className = "code-lang";
    tag.textContent = language;
    pre.append(tag);
  }
  const block = document.createElement("code");
  block.textContent = code;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy-code";
  copy.textContent = "salin";
  copy.addEventListener("click", async () => {
    await copyText(code);
    copy.textContent = "tersalin";
    setTimeout(() => {
      copy.textContent = "salin";
    }, 1400);
  });
  pre.append(block, copy);
  return pre;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}

function paragraph(lines) {
  const node = document.createElement("p");
  node.innerHTML = lines.map(inline).join("<br />");
  return node;
}

function list(items, ordered) {
  const node = document.createElement(ordered ? "ol" : "ul");
  for (const item of items) {
    const li = document.createElement("li");
    li.innerHTML = inline(item);
    node.append(li);
  }
  return node;
}

function heading(level, text) {
  const node = document.createElement(level <= 2 ? "h3" : "h4");
  node.innerHTML = inline(text);
  return node;
}

function quote(lines) {
  const node = document.createElement("blockquote");
  node.innerHTML = lines.map(inline).join("<br />");
  return node;
}

/// Blocks are split on fenced code first, so nothing inside a fence is parsed.
export function render(text) {
  const fragment = document.createDocumentFragment();
  const segments = String(text ?? "").split(/^```/m);

  segments.forEach((segment, index) => {
    if (index % 2 === 1) {
      const newline = segment.indexOf("\n");
      const language = newline === -1 ? segment.trim() : segment.slice(0, newline).trim();
      const code = newline === -1 ? "" : segment.slice(newline + 1).replace(/\n$/, "");
      fragment.append(codeBlock(language, code));
      return;
    }
    appendProse(fragment, segment);
  });

  return fragment;
}

function appendProse(fragment, segment) {
  const lines = segment.split("\n");
  let paragraphLines = [];
  let listItems = [];
  let listOrdered = false;
  let quoteLines = [];

  const flushParagraph = () => {
    if (paragraphLines.length > 0) {
      fragment.append(paragraph(paragraphLines));
      paragraphLines = [];
    }
  };
  const flushList = () => {
    if (listItems.length > 0) {
      fragment.append(list(listItems, listOrdered));
      listItems = [];
    }
  };
  const flushQuote = () => {
    if (quoteLines.length > 0) {
      fragment.append(quote(quoteLines));
      quoteLines = [];
    }
  };
  const flushAll = () => {
    flushParagraph();
    flushList();
    flushQuote();
  };

  for (const line of lines) {
    const trimmed = line.trim();
    const bullet = /^[-*+]\s+(.*)$/.exec(trimmed);
    const numbered = /^\d+[.)]\s+(.*)$/.exec(trimmed);
    const head = /^(#{1,6})\s+(.*)$/.exec(trimmed);
    const quoted = /^>\s?(.*)$/.exec(trimmed);

    if (trimmed === "") {
      flushAll();
    } else if (head) {
      flushAll();
      fragment.append(heading(head[1].length, head[2]));
    } else if (bullet) {
      flushParagraph();
      flushQuote();
      if (listItems.length > 0 && listOrdered) {
        flushList();
      }
      listOrdered = false;
      listItems.push(bullet[1]);
    } else if (numbered) {
      flushParagraph();
      flushQuote();
      if (listItems.length > 0 && !listOrdered) {
        flushList();
      }
      listOrdered = true;
      listItems.push(numbered[1]);
    } else if (quoted) {
      flushParagraph();
      flushList();
      quoteLines.push(quoted[1]);
    } else {
      flushList();
      flushQuote();
      paragraphLines.push(trimmed);
    }
  }
  flushAll();
}
