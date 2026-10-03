'use client'

import { Sun, Moon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useSatQuery } from '@/components/satquery-context'
import { LeftSidebar, MobileSidebarTrigger } from './left-sidebar'
import { ChatShell } from './ChatShell'

function AppShell() {
  const { darkMode, toggleDarkMode } = useSatQuery()

  return (
    <div className={`satquery-shell ${darkMode ? 'dark-shell' : ''}`}>
      {/* Left Sidebar — Navigation, recent chats */}
      {/* BACKEND: Sidebar data (nav items, recent chats) loaded in context from GET /api/chats */}
      <LeftSidebar />

      {/* Main Workspace */}
      <main className="workspace">
        {/* Top bar — mobile brand + theme toggle */}
        

        {/*
          ChatShell replaces the previous direct <HeroSection /> + <Composer />
          pairing. It switches between:
            - HeroSection (globe / categories / suggestions) when there's no
              active conversation, and
            - a scrolling prompt/response thread once a query has been sent,
          while cooperating with LeftSidebar's recent-chats list (saving a
          draft when "New chat" is pressed, resuming a chat when one is
          clicked in the sidebar). See ChatShell.tsx for the small
          SatQueryContext additions this needs.

          RESPONSIVENESS: `.workspace` needs a resolvable height (not
          `height: auto`) for ChatShell's internal `flex: 1; overflow-y: auto`
          thread region to actually scroll instead of pushing the composer
          off screen — e.g.:
            .satquery-shell { display: flex; height: 100dvh; }
            .workspace { flex: 1; min-width: 0; display: flex; flex-direction: column; height: 100dvh; }
          If `.workspace` already does this in your global stylesheet, no
          change needed; this is just a checklist item if the composer isn't
          sticking to the bottom on small screens.
        */}
        <ChatShell />
      </main>

      {/* Right Sidebar — Profile, dataset, tools, examples */}
      {/* BACKEND: Right sidebar data loaded from multiple endpoints (see right-sidebar.tsx) */}
      {/* <RightSidebar /> */}
    </div>
  )
}

/**
 * SatQueryApp — Top-level exported component.
 * Wraps everything in SatQueryProvider for shared state.
 *
 * Usage in page.tsx:
 * ```tsx
 * import { SatQueryApp } from '@/components/satquery-app'
 * export default function Page() {
 *   return <SatQueryApp />
 * }
 * ```
 */
export { AppShell }