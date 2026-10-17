export function initializeQuickGuide({document, hasSeen, remember, openSection, restore}) {
  const dialog = document.getElementById('quick-guide');
  const welcome = document.getElementById('welcome-guide');
  const steps = [...dialog.querySelectorAll('[data-guide-step]')];
  let step = 0;
  function showStep(index, focus = true) {
    step = index;
    steps.forEach((node, i) => { node.hidden = i !== step; node.querySelector('h1').id = `guide-title-${i}`; });
    dialog.setAttribute('aria-labelledby', `guide-title-${step}`);
    document.getElementById('guide-position').textContent = `${step + 1} of ${steps.length}`;
    document.getElementById('guide-previous').disabled = step === 0;
    document.getElementById('guide-next').hidden = step === steps.length - 1;
    document.getElementById('guide-finish').hidden = step !== steps.length - 1;
    dialog.scrollTop = 0;
    if (focus) steps[step].querySelector('h1').focus();
  }
  function dismiss() { welcome.hidden = true; remember(); }
  function open() {
    showStep(0, false); dialog.showModal(); steps[0].querySelector('h1').focus();
  }
  document.getElementById('open-guide').onclick = open;
  document.getElementById('welcome-guide-open').onclick = open;
  document.getElementById('welcome-dismiss').onclick = dismiss;
  document.getElementById('welcome-restore').onclick = restore;
  document.getElementById('close-guide').onclick = () => dialog.close();
  document.getElementById('guide-finish').onclick = () => dialog.close();
  document.getElementById('guide-previous').onclick = () => showStep(Math.max(0, step - 1));
  document.getElementById('guide-next').onclick = () => showStep(Math.min(steps.length - 1, step + 1));
  dialog.addEventListener('close', dismiss);
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  for (const button of dialog.querySelectorAll('[data-guide-action]')) button.onclick = () => {
    dialog.close(); openSection(button.dataset.guideAction);
  };
  return {open, showWelcome(present) { welcome.hidden = present || hasSeen(); }};
}
