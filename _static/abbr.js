// Подсказки {abbr}: текст из title переносим в data-tip. Иначе поверх нашей подсказки из custom.css
// через пару секунд вылезает штатная подсказка браузера с тем же текстом.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("abbr[title]").forEach((abbr) => {
    abbr.dataset.tip = abbr.title;
    abbr.setAttribute("aria-label", abbr.title);
    abbr.removeAttribute("title");
  });
});
