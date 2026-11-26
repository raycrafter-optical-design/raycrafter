// Keeps the report served by RayCrafter up to date (see raycrafter/report/live_server.py): it
// waits to hear that there is a newer report, then reloads the page. A page in a hidden tab is
// reloaded when it is next looked at.
(() => {
  const RETRY_MS = 3000;
  const shown = Number(document.currentScript.dataset.revision); // what this page shows
  const notice = document.getElementById('live-notice');
  const liveText = notice.textContent;
  let latest = shown;

  function reloadIfOutOfDate() {
    if (latest !== shown && !document.hidden) location.reload();
  }

  async function watch() {
    for (;;) {
      try {
        // Answered when the report is no longer at this revision, or after a while.
        const response = await fetch('revision?after=' + latest);
        if (!response.ok) throw new Error(response.statusText);
        latest = Number(await response.text());
        notice.textContent = liveText;
        reloadIfOutOfDate();
      } catch (error) {
        notice.textContent = 'RayCrafter cannot be reached, so this page is no longer updated';
        await new Promise(resolve => setTimeout(resolve, RETRY_MS));
      }
    }
  }

  document.addEventListener('visibilitychange', reloadIfOutOfDate);
  watch();
})();
