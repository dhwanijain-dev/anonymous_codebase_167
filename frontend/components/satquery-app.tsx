'use client'

/**
 * BACKEND INTEGRATION: SatQueryApp — Root Application Shell
 *
 * This is the main application component that composes all sub-components
 * within the SatQueryProvider context. The provider manages all shared state.
 *
 * To connect to a backend:
 * 1. Update SatQueryProvider in satquery-context.tsx to fetch initial data
 * 2. Replace mock handlers in each component with actual API calls
 * 3. See individual component files for specific API endpoint comments
 *
 * Architecture:
 * - SatQueryProvider (context) → manages all state
 * - LeftSidebar → navigation, recent chats
 * - Workspace (main) → hero section, composer, results
 * - RightSidebar → profile, dataset, tools, examples
 */

import { SatQueryProvider, useSatQuery } from '@/components/satquery-context'
import { LeftSidebar, MobileSidebarTrigger } from '@/components/left-sidebar'
import { RightSidebar } from '@/components/right-sidebar'
import { HeroSection } from '@/components/hero-section'
import { Composer } from '@/components/composer'
import { Button } from '@/components/ui/button'
import { Moon, Sun } from 'lucide-react'
import { ChatShell } from './ChatShell'
import {AppShell} from './AppShell'
/**
 * AppShell — The inner shell that reads from context.
 * Separated from SatQueryApp so it can access the provider.
 */
// function AppShell() {
//   const { darkMode, toggleDarkMode } = useSatQuery()

//   return (
//     <div className={`satquery-shell ${darkMode ? 'dark-shell' : ''}`}>
//       {/* Left Sidebar — Navigation, recent chats */}
//       {/* BACKEND: Sidebar data (nav items, recent chats) loaded in context from GET /api/chats */}
//       <LeftSidebar />

//       {/* Main Workspace */}
//       <main className="workspace">
//         {/* Top bar — mobile brand + theme toggle */}
//         <header className="topbar">
//           {/* Mobile menu trigger — only visible on small screens */}
//           <MobileSidebarTrigger />
//           <div className="mobile-brand">SatQuery AI</div>
//           {/* BACKEND: Theme preference can be persisted via PUT /api/user/settings { theme } */}
//           <Button
//             variant="ghost"
//             size="icon"
//             onClick={toggleDarkMode}
//             aria-label="Toggle color mode"
//           >
//             {darkMode ? <Sun /> : <Moon />}
//           </Button>
//         </header>

//         {/* Hero content — headline, globe, category cards, suggestions */}
//         <HeroSection />

//         {/* Composer — query input, upload dialog, dataset selector, advanced options */}
//         {/* BACKEND: Query submission handled in Composer via POST /api/analysis/query */}
//         <div className="hero-content" style={{ paddingTop: 0 }}>
//           <Composer />
//         </div>
//       </main>

//       {/* Right Sidebar — Profile, dataset, tools, examples */}
//       {/* BACKEND: Right sidebar data loaded from multiple endpoints (see right-sidebar.tsx) */}
//       {/* <RightSidebar /> */}
//     </div>
//   )
// }

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
export function SatQueryApp() {
  return (
    <SatQueryProvider>
      <AppShell />
    </SatQueryProvider>
  )
}
