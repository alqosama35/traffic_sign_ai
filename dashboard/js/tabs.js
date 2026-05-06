const buttons = document.querySelectorAll('.tab-btn');
const panels  = document.querySelectorAll('.tab-panel');

function activate(tabId) {
  buttons.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  panels.forEach(p  => p.classList.toggle('active',  p.id === `tab-${tabId}`));
}

buttons.forEach(btn => btn.addEventListener('click', () => activate(btn.dataset.tab)));

// Activate first tab on load
if (buttons.length > 0) activate(buttons[0].dataset.tab);
