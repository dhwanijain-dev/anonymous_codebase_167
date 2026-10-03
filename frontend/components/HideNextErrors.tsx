"use client";

import { useEffect } from "react";

export function HideNextErrors() {
  useEffect(() => {
    const hide = () => {
      const portal = document.querySelector("nextjs-portal");

      if (!portal?.shadowRoot) return;

      const style = document.createElement("style");
      style.id = "hide-next-errors";

      style.textContent = `
        [data-nextjs-dialog-overlay],
        [data-nextjs-dialog-root] {
          display: none !important;
        }

        [data-next-badge] {
          display: none !important;
        }
      `;

      if (!portal.shadowRoot.querySelector("#hide-next-errors")) {
        portal.shadowRoot.appendChild(style);
      }
    };

    const observer = new MutationObserver(hide);

    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
    });

    hide();

    return () => observer.disconnect();
  }, []);

  return null;
}