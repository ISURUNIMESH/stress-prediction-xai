"use strict";

(function initializeTheme() {
  const root = document.documentElement;
  const toggle = document.getElementById("theme-toggle");
  const storageKey = "stress-theme";

  function selectedTheme() {
    return root.dataset.theme === "dark" ? "dark" : "light";
  }

  function render(theme) {
    const dark = theme === "dark";
    root.dataset.theme = dark ? "dark" : "light";
    if (!toggle) return;
    toggle.setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} mode`);
    toggle.title = `Switch to ${dark ? "light" : "dark"} mode`;
    toggle.querySelector("[aria-hidden='true']").textContent = dark ? "☀" : "☾";
    toggle.querySelector(".theme-label").textContent = dark ? "Light mode" : "Dark mode";
  }

  let preferred = "light";
  try { preferred = localStorage.getItem(storageKey) || "light"; } catch (_error) { /* Storage can be disabled. */ }
  render(preferred);
  if (toggle) {
    toggle.addEventListener("click", () => {
      const next = selectedTheme() === "dark" ? "light" : "dark";
      render(next);
      try { localStorage.setItem(storageKey, next); } catch (_error) { /* Theme works for this page view. */ }
    });
  }
})();
