function baseLabel(labelText) {
  return labelText.replace(/\s+\(cont\.\)$/i, "").trim();
}

function continuationLabel(labelText) {
  return `${baseLabel(labelText)} (cont.)`;
}

function overflowTarget(card) {
  return card.querySelector(".rich-output") || card;
}

function isOverflowing(card) {
  const target = overflowTarget(card);
  return target.scrollHeight > target.clientHeight + 1;
}

function clearCloneContent(clone) {
  clone.querySelectorAll(".meta-row").forEach((node) => node.remove());
  const richOutput = clone.querySelector(".rich-output");
  if (!richOutput) {
    return;
  }

  const summaryGrid = richOutput.querySelector(".summary-grid");
  if (summaryGrid) {
    summaryGrid.innerHTML = "";
    return;
  }

  const formatted = richOutput.querySelector(".formatted-output");
  if (formatted) {
    formatted.innerHTML = "";
    return;
  }

  const treatmentOutput = richOutput.querySelector(".treatment-output");
  if (treatmentOutput) {
    const introParagraphs = treatmentOutput.querySelectorAll(":scope > p");
    introParagraphs.forEach((node) => node.remove());
    const groups = treatmentOutput.querySelector(".treatment-groups");
    if (groups) {
      groups.innerHTML = "";
    }
    return;
  }

  richOutput.innerHTML = "";
}

function createContinuationCard(card) {
  const clone = card.cloneNode(true);
  const label = clone.querySelector(".pill-label");
  if (label) {
    label.textContent = continuationLabel(label.textContent || "");
  }
  clearCloneContent(clone);
  return clone;
}

function prependMovedNode(targetContainer, node) {
  targetContainer.insertBefore(node, targetContainer.firstChild);
}

function moveOverflowContent(card, continuation) {
  const summaryGrid = card.querySelector(".summary-grid");
  const continuationSummaryGrid = continuation.querySelector(".summary-grid");
  if (summaryGrid && continuationSummaryGrid) {
    const items = summaryGrid.querySelectorAll(".summary-item");
    if (!items.length) {
      return false;
    }
    prependMovedNode(continuationSummaryGrid, items[items.length - 1]);
    return true;
  }

  const formatted = card.querySelector(".formatted-output");
  const continuationFormatted = continuation.querySelector(".formatted-output");
  if (formatted && continuationFormatted) {
    const blocks = Array.from(formatted.children);
    if (!blocks.length) {
      return false;
    }
    prependMovedNode(continuationFormatted, blocks[blocks.length - 1]);
    return true;
  }

  const groups = card.querySelector(".treatment-groups");
  const continuationGroups = continuation.querySelector(".treatment-groups");
  if (groups && continuationGroups) {
    const treatmentCards = Array.from(groups.querySelectorAll(":scope > .treatment-group"));
    if (!treatmentCards.length) {
      return false;
    }
    prependMovedNode(continuationGroups, treatmentCards[treatmentCards.length - 1]);
    return true;
  }

  return false;
}

function paginateCard(card, grid) {
  let currentCard = card;

  while (isOverflowing(currentCard)) {
    const continuation = createContinuationCard(currentCard);
    let moved = false;

    while (isOverflowing(currentCard)) {
      const didMove = moveOverflowContent(currentCard, continuation);
      if (!didMove) {
        break;
      }
      moved = true;
    }

    if (!moved) {
      currentCard.classList.add("paper-card-unbounded");
      break;
    }

    grid.insertBefore(continuation, currentCard.nextSibling);
    currentCard = continuation;
  }
}

function paginateResults() {
  const grid = document.querySelector(".results-grid");
  if (!grid || grid.dataset.paginated === "true") {
    return;
  }
  grid.dataset.paginated = "true";

  const initialCards = Array.from(grid.querySelectorAll(":scope > .output-card"));
  initialCards.forEach((card) => paginateCard(card, grid));
}

function schedulePagination() {
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(paginateResults);
  });

  window.setTimeout(() => {
    const grid = document.querySelector(".results-grid");
    if (!grid) {
      return;
    }
    delete grid.dataset.paginated;
    paginateResults();
  }, 120);
}

window.addEventListener("DOMContentLoaded", schedulePagination);
window.addEventListener("load", schedulePagination);
