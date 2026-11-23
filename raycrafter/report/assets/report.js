// The "?" notes of the RayCrafter HTML report (see raycrafter/report/html_report.py).
const note = document.getElementById('explain');
const explanations = JSON.parse(document.getElementById('explanation-data').textContent);
const GAP = 14;                      // space between the card and the note
let activeButton = null;

// Place the note next to the card: to its right or left if there is room, otherwise below or above it,
// so the plot itself stays uncovered and can be read alongside the text.
function placeNote() {
  if (!activeButton) return;
  const card = activeButton.closest('.card');
  const cardBox = card.getBoundingClientRect();
  const noteWidth = note.offsetWidth, noteHeight = note.offsetHeight;
  const roomRight = window.innerWidth - cardBox.right;
  const roomLeft = cardBox.left;
  const roomBelow = window.innerHeight - cardBox.bottom;
  const roomAbove = cardBox.top;

  let left, top;                                      // viewport coordinates
  if (roomRight >= noteWidth + GAP) {
    left = cardBox.right + GAP; top = cardBox.top;
  } else if (roomLeft >= noteWidth + GAP) {
    left = cardBox.left - GAP - noteWidth; top = cardBox.top;
  } else if (roomBelow >= roomAbove) {
    left = cardBox.right - noteWidth; top = cardBox.bottom + GAP;
  } else {
    left = cardBox.right - noteWidth; top = cardBox.top - GAP - noteHeight;
  }
  left = Math.max(8, Math.min(left, window.innerWidth - noteWidth - 8));
  note.style.left = (left + window.scrollX) + 'px';
  note.style.top = (top + window.scrollY) + 'px';
}

function closeNote() {
  note.hidden = true;
  document.querySelectorAll('.card.active').forEach(card => card.classList.remove('active'));
  activeButton = null;
}

// Clicking a "?" shows its note; clicking the same one again hides it. The page is never dimmed or blocked.
document.querySelectorAll('.help').forEach(button => {
  button.addEventListener('click', () => {
    if (activeButton === button) { closeNote(); return; }
    closeNote();
    const entry = explanations[button.dataset.key];
    note.querySelector('h3').textContent = entry.title;
    note.querySelector('.body').innerHTML = entry.paragraphs.map(text => '<p>' + text + '</p>').join('');
    activeButton = button;
    button.closest('.card').classList.add('active');
    note.hidden = false;
    placeNote();
  });
});

note.querySelector('.close').addEventListener('click', closeNote);
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeNote(); });
window.addEventListener('resize', placeNote);
