'use client'

import {
  MessageCircle,
  Scan,
  Bookmark,
  Database,
  Grid2X2,
  ArrowRight,
  Menu,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet'
import { useSatQuery } from '@/components/satquery-context'

/**
 * BACKEND INTEGRATION: Navigation items
 *
 * API Endpoint: GET /api/navigation
 * Response: { items: { name, icon, path, badge? }[] }
 *
 * Replace this static array with navigation fetched from your API.
 * The icon mapping can be done via a lookup table.
 */
const navItems = [
  { name: 'New Chat', Icon: MessageCircle },
  { name: 'Explore', Icon: Scan },
  { name: 'Saved Analyses', Icon: Bookmark },
  { name: 'Datasets', Icon: Database },
  { name: 'Tools', Icon: Grid2X2 },
]

/**
 * SidebarContent — Inner content shared between desktop sidebar
 * and mobile Sheet drawer. Extracted to avoid duplicating markup.
 */
function SidebarContent() {
  // `activeChat` and `startNewSession` are NOT in the original context —
  // see the "satquery-context additions" notes shared alongside this file.
  // They're read with optional chaining below so this component still
  // compiles/renders even before you've added them, it just won't
  // highlight the active chat or reset the draft until you do.
  const { activeNav, setActiveNav, recentChats, loadChat, activeChat, startNewSession } =
    useSatQuery() as ReturnType<typeof useSatQuery> & {
      activeChat?: { id: string } | null
      startNewSession?: () => void
    }

  const handleNavClick = (name: string) => {
    setActiveNav(name)

    if (name === 'New Chat') {
      /**
       * BACKEND/STATE INTEGRATION: New Chat (sidebar entry point)
       * ---------------------------------------------------------------
       * This mirrors ChatShell's own "New chat" button so both entry
       * points behave identically: whatever draft is currently on
       * screen should be persisted into `recentChats` before it's
       * cleared. Right now that persistence happens inside ChatShell
       * (it owns the in-progress `messages` array), so calling
       * startNewSession() here only clears context-level state
       * (sentQuery/analysisResults/activeChat) — ChatShell reacts to
       * that and clears its local draft too, but won't have a chance to
       * save it first.
       *
       * To make both "New Chat" buttons save-then-reset consistently,
       * migrate the in-progress `messages` array into SatQueryContext
       * (see ChatShell.tsx's top-of-file note) so a single
       * `startNewSession()` in the context can do save + clear in one
       * place, and both this button and ChatShell's button just call it.
       */
      startNewSession?.()
    }
  }

  return (
    <>
      {/* Brand lockup — logo + app name */}
      <div className="brand-lockup">
        <div className="brand-mark"><span /><span /><b>+</b></div>
        <div>
          <div className="brand-name">SatQuery AI</div>
          <div className="brand-tagline">SEE&nbsp;&nbsp; UNDERSTAND&nbsp;&nbsp; ACT</div>
        </div>
      </div>

      <Separator />

      {/* Primary navigation */}
      {/* BACKEND: Navigation items can be fetched from GET /api/navigation */}
      <nav className="primary-nav" aria-label="Primary navigation">
        {navItems.map(({ name, Icon }) => (
          <Button
            key={name}
            variant="ghost"
            className={`nav-item ${activeNav === name ? 'active' : ''}`}
            onClick={() => handleNavClick(name)}
          >
            <Icon data-icon="inline-start" />
            {name}
          </Button>
        ))}
      </nav>

      <Separator />

      {/* Recent chats section */}
      {/* BACKEND: Fetch recent chats from GET /api/chats?limit=5&sort=recent */}
      <div className="recent-section">
        <div className="section-label">RECENT CHATS</div>
        <ScrollArea>
          {recentChats.map((chat) => (
            <button
              className={`recent-chat ${activeChat?.id === chat.id ? 'active' : ''}`}
              key={chat.id}
              onClick={() => {
                /* BACKEND: Load chat from GET /api/chats/:id */
                loadChat(chat)
              }}
            >
              <MessageCircle />
              {chat.title}
            </button>
          ))}
        </ScrollArea>
        {/* BACKEND: Navigate to full chat list or paginate GET /api/chats?page=1 */}
        <button className="view-all">View all <ArrowRight /></button>
      </div>

      <div className="sidebar-footnote">
        EARTH<br />INSIGHTS<br />FOR A BRIGHTER<br />TOMORROW
      </div>
    </>
  )
}

/**
 * LeftSidebar — Desktop sidebar with brand, navigation,
 * recent chats, and decorative footer text.
 *
 * RESPONSIVENESS: this component's own markup is unchanged — visibility
 * across breakpoints (hidden on mobile in favor of MobileSidebarTrigger's
 * Sheet drawer) is expected to be handled by your existing CSS for
 * `.left-sidebar` (e.g. `display: none` under a breakpoint, shown again
 * inside the Sheet via the shared `.left-sidebar` class). No change needed
 * here unless that rule doesn't already exist.
 */
export function LeftSidebar() {
  return (
    <aside className="left-sidebar">
      <SidebarContent />
    </aside>
  )
}

/**
 * MobileSidebarTrigger — Sheet-based sidebar for mobile viewports.
 * Opens as a slide-in drawer from the left containing the same
 * sidebar content as the desktop version.
 */
export function MobileSidebarTrigger() {
  const { isMobileSidebarOpen, setMobileSidebarOpen } = useSatQuery()

  return (
    <Sheet open={isMobileSidebarOpen} onOpenChange={setMobileSidebarOpen}>
  <SheetTrigger>
  <Menu className="h-5 w-5" />
</SheetTrigger>
      <SheetContent side="left" className="p-0 w-[288px] border-r-0">
        <div className="left-sidebar" style={{ borderRight: 'none' }}>
          <SidebarContent />
        </div>
      </SheetContent>
    </Sheet>
  )
}