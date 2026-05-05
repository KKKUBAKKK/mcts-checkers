/* global game state */
let state = null;       // server game state
let selected = null;    // {r, c} of selected piece
let humanColor = 1;
let thinking = false;

const $ = (s) => document.querySelector(s);
const boardEl = $("#board");
const statusEl = $("#status-text");
const container = $("#board-container");

/* ── helpers ──────────────────────────────────────────────────────────── */

function movesFromCell(r, c) {
  if (!state || !state.legal_moves) return [];
  return state.legal_moves.filter(m => m[0][0] === r && m[0][1] === c);
}

function destinationsFromCell(r, c) {
  /* returns set of "r,c" strings for every possible final square */
  const dests = new Set();
  for (const m of movesFromCell(r, c)) {
    const last = m[m.length - 1];
    dests.add(`${last[0]},${last[1]}`);
  }
  return dests;
}

function findMove(fromR, fromC, toR, toC) {
  const candidates = movesFromCell(fromR, fromC).filter(m => {
    const last = m[m.length - 1];
    return last[0] === toR && last[1] === toC;
  });
  /* if multiple capture paths lead to the same square, pick the longest */
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => b.length - a.length);
  return candidates[0];
}

function playerName(p) { return p === 1 ? "White" : "Black"; }

function renderCaptured() {
  const el = $("#captured");
  el.innerHTML = "";
  const pieces = state.captured || [];
  for (const p of pieces) {
    const pip = document.createElement("span");
    pip.className = "captured-pip " + (p > 0 ? "white" : "black");
    if (Math.abs(p) === 2) pip.classList.add("king");
    el.appendChild(pip);
  }
}

function isHumanTurn() {
  return state && !state.winner && state.current_player === humanColor && !thinking;
}

/* ── rendering ────────────────────────────────────────────────────────── */

function render() {
  if (!state) return;
  boardEl.innerHTML = "";

  const dests = selected ? destinationsFromCell(selected.r, selected.c) : new Set();

  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      const cell = document.createElement("div");
      cell.className = "cell " + ((r + c) % 2 === 0 ? "light" : "dark");
      cell.dataset.r = r;
      cell.dataset.c = c;

      if (selected && selected.r === r && selected.c === c) {
        cell.classList.add("selected");
      }
      if (dests.has(`${r},${c}`)) {
        cell.classList.add("highlight");
      }

      const p = state.board[r][c];
      if (p !== 0) {
        const piece = document.createElement("div");
        piece.className = "piece " + (p > 0 ? "white" : "black");
        if (Math.abs(p) === 2) piece.classList.add("king");
        cell.appendChild(piece);
      }

      cell.addEventListener("click", () => onCellClick(r, c));
      boardEl.appendChild(cell);
    }
  }

  /* captured pieces */
  renderCaptured();

  /* info */
  $("#info-move").textContent = state.move_count;
  $("#info-player").textContent = playerName(state.current_player);

  /* status */
  if (state.winner === 1) statusEl.textContent = "White wins!";
  else if (state.winner === -1) statusEl.textContent = "Black wins!";
  else if (state.winner === 0) statusEl.textContent = "Draw!";
  else if (thinking) statusEl.textContent = "Opponent is thinking…";
  else if (isHumanTurn()) statusEl.textContent = "Your turn (" + playerName(humanColor) + ")";
  else statusEl.textContent = "Opponent's turn";

  statusEl.classList.toggle("thinking", thinking);
}

/* ── interaction ──────────────────────────────────────────────────────── */

function onCellClick(r, c) {
  if (!isHumanTurn()) return;

  const p = state.board[r][c];
  const isOwn = (humanColor === 1 && p > 0) || (humanColor === -1 && p < 0);

  if (isOwn) {
    /* select / toggle */
    if (selected && selected.r === r && selected.c === c) {
      selected = null;
    } else {
      selected = { r, c };
    }
    render();
    return;
  }

  if (!selected) return;

  /* try to move */
  const move = findMove(selected.r, selected.c, r, c);
  if (!move) return;

  selected = null;
  sendMove(move);
}

async function sendMove(move) {
  thinking = true;
  render();
  try {
    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ move }),
    });
    state = await res.json();
    if (state.error) { alert(state.error); thinking = false; render(); return; }
    render();

    /* let AI respond */
    if (!state.winner && state.winner !== 0 && state.current_player !== humanColor) {
      await requestAiMove();
    } else {
      thinking = false;
      render();
    }
  } catch (e) {
    console.error(e);
    thinking = false;
    render();
  }
}

async function requestAiMove() {
  thinking = true;
  container.classList.add("thinking");
  render();
  try {
    const res = await fetch("/api/ai_move", { method: "POST" });
    state = await res.json();
  } catch (e) {
    console.error(e);
  }
  thinking = false;
  container.classList.remove("thinking");
  render();
}

/* ── new game ─────────────────────────────────────────────────────────── */

async function startNewGame() {
  humanColor = parseInt($("#sel-color").value);
  const variant = $("#sel-variant").value;
  const iterations = parseInt($("#inp-iterations").value) || 2000;
  const maxMoves = parseInt($("#inp-max-moves").value) || 200;

  thinking = true;
  selected = null;
  render();

  try {
    const res = await fetch("/api/new_game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        human_color: humanColor,
        variant,
        iterations,
        max_moves: maxMoves,
      }),
    });
    state = await res.json();
  } catch (e) {
    console.error(e);
  }
  thinking = false;
  render();
}

/* ── init ─────────────────────────────────────────────────────────────── */

$("#btn-new-game").addEventListener("click", startNewGame);

(async () => {
  try {
    const res = await fetch("/api/state");
    state = await res.json();
  } catch (e) {
    console.error(e);
  }
  render();
})();
